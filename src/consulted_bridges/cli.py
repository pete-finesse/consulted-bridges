"""Command line: `uv run cb <command> --help` for each command's options."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def parse_layers(s: str | None, h) -> list[int]:
    if s in (None, "", "band"):
        return h.band
    if s == "all":
        return list(range(h.n_layers))
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def parse_edit(s: str, h):
    """'ablate words=Salzburg,Graz layers=23-30 pos=last'  |  'steer words=Salzburg alpha=-0.5 ...'  |
    'swap words=Salzburg,Budapest s=1 ...'. pos: last | all | -3 | 'from:<text>' | 'span:<text>'
    (quote values with spaces). lens=logit uses unembedding directions instead of J-lens ones."""
    from .interventions import Edit

    kind, *kv = shlex.split(s)
    d = dict(x.split("=", 1) for x in kv)
    pos = d.get("pos", "last")
    pos = int(pos) if pos.lstrip("-").isdigit() else pos
    return Edit(kind, [w.replace("_", " ") for w in d["words"].split(",")], parse_layers(d.get("layers"), h), pos,
                alpha=float(d.get("alpha", 0)), strength=float(d.get("s", 1)), lens=d.get("lens", "jlens"))


def prompt_text(args, h) -> str:
    from .text import chat

    if args.text is not None:
        return args.text
    return chat(h, args.user, args.prefill or "", args.system)


def add_prompt(p):
    p.add_argument("--user", help="user message (chat prompt)")
    p.add_argument("--prefill", default="", help="start of the assistant's reply")
    p.add_argument("--system", default=None)
    p.add_argument("--text", default=None, help="a raw plain-text prompt instead of chat")


def show_summary(run_dir, band, full: bool = False, only: tuple = ()):
    """Print the key same-question tables (all of them with full=True); all go to summary_*.csv."""
    from .silent_dissent.summarize import summarize

    out = summarize(run_dir, band)
    cols = ["n_facts", "n_bridges", "mean", "mean_bridge_once", "ci_lo", "ci_hi"]
    key = {
        "contrasts": lambda d: d[d.contrast == "same_country - bridge"],
        "conditions": lambda d: d[d.pair == "answer_written - filler"],
        "ablate": lambda d: d,
        "probs": lambda d: d[d.measure == "p_full"],
    }
    for name, df in out.items():
        if df.empty or (not full and name not in key) or (only and name not in only):
            continue
        d = df if full else key[name](df)
        print(f"\n== summary_{name}.csv{'' if full else ' (key rows)'} ==")
        print(d.round(3).to_string(index=False))
    print(f"\nband layers {band[0]}-{band[-1]}; score = log10(rank+1), lower = ranks better. "
          f"All tables: {run_dir}/summary_*.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cb", description=__doc__)
    ap.add_argument("--config", default="tiny", help="tiny | qwen35_4b | qwen36_27b | path to a .yaml")
    ap.add_argument("--allow-download", action="store_true", help="permit downloading the model / lens")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: from the config)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("readout", help="ranks of words at every position and layer, both lenses")
    add_prompt(p)
    p.add_argument("--words", required=True, help="comma-separated, e.g. Salzburg,Graz,Vienna,Tokyo")
    p.add_argument("--start", default=None, help="'all positions' summaries start at this text")
    p.add_argument("--edit", action="append", default=[], help="apply an edit first (see intervene)")
    p.add_argument("--out", default=None, help="write the full position x layer table here (.csv)")

    p = sub.add_parser("intervene", help="answer probabilities under ablate / steer / swap edits")
    add_prompt(p)
    p.add_argument("--candidates", required=True, help="e.g. Austria,France,Hungary")
    p.add_argument("--edit", action="append", default=[], help=parse_edit.__doc__)
    p.add_argument("--generate", action="store_true", help="also show the greedy answer under each edit")
    p.add_argument("--out", default=None)

    p = sub.add_parser("batch", help="readouts over many facts and conditions -> tidy CSVs")
    p.add_argument("--conditions", default="q_alone,country_alone,subject_alone,answer_written,filler,"
                   "np_agree,np_pressure,np_hidden,paper_agree,paper_pressure,paper_hidden")
    p.add_argument("--per-bridge", type=int, default=3)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--uids", default=None, help="comma-separated uids (e.g. thf:91051) instead of sampling")
    p.add_argument("--known", default=None, help="known.csv from `sdh known`: keep facts the model knows")
    p.add_argument("--paper-fmt", default="plain", choices=["plain", "chat"])
    p.add_argument("--same-country", default="random", choices=["random", "frequent"],
                   help="same-country control: a random other bridge city, or the most frequent one")
    p.add_argument("--keep-leaky", action="store_true", help="keep facts whose subject names the bridge")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)

    p = sub.add_parser("ablate", help="batch removal test: ablate bridge vs controls, P(answer)")
    p.add_argument("--condition", default="q_alone")
    p.add_argument("--layers", default="band")
    p.add_argument("--where", default="question", choices=["question", "last"])
    p.add_argument("--roles", default="bridge,same_country,capital,random")
    p.add_argument("--lens", default="jlens")
    p.add_argument("--per-bridge", type=int, default=3)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--uids", default=None)
    p.add_argument("--known", default=None)
    p.add_argument("--paper-fmt", default="plain", choices=["plain", "chat"])
    p.add_argument("--same-country", default="random", choices=["random", "frequent"],
                   help="same-country control: a random other bridge city, or the most frequent one")
    p.add_argument("--keep-leaky", action="store_true", help="keep facts whose subject names the bridge")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)

    p = sub.add_parser("summarize", help="same-question summary tables of a batch / ablate run")
    p.add_argument("run_dir")
    p.add_argument("--band", default=None, help="layers, e.g. 23-30 (default: the config's band)")

    p = sub.add_parser("known", help="which facts the model knows (the paper's 3-prompt screen)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", required=True)

    p = sub.add_parser("mozart", help="the worked example: Peter's Neuronpedia Mozart checks")
    p.add_argument("--peer", default="France")
    p.add_argument("--layers", default="band")
    p.add_argument("--out", default=None)

    args = ap.parse_args(argv)
    from .load import load

    h = load(args.config, args.allow_download, args.device)
    cname = Path(args.config).stem

    if args.cmd == "readout":
        from .readout import readout
        from .text import offsets

        text = prompt_text(args, h)
        ro = readout(h, text, args.words.split(","), edits=[parse_edit(e, h) for e in args.edit])
        start = 0
        if args.start:
            at = text.rfind(args.start)
            start = next((i for i, (a, b) in enumerate(offsets(h, text)) if b > at), 0)
        print("pieces:", ro.pieces)
        print("\nRank at the last position (token %r), per layer:" % ro.tokens[-1])
        print(ro.last().to_string(index=False))
        print(f"\nSummary (band {h.band[0]}-{h.band[-1]}):")
        print(ro.summary(h.band, start).to_string(index=False))
        if args.out:
            ro.long().to_csv(args.out, index=False)
            print("wrote", args.out)

    elif args.cmd == "intervene":
        from .interventions import compare, generate

        text = prompt_text(args, h)
        edits = [parse_edit(e, h) for e in args.edit]
        runs = {e.label: [e] for e in edits}
        df = compare(h, text, args.candidates.split(","), runs)
        print(df.to_string(index=False))
        if args.generate:
            for label, es in {"baseline": [], **runs}.items():
                print(f"{label:45s} -> {generate(h, text, es)!r}")
        if args.out:
            df.to_csv(args.out, index=False)

    elif args.cmd in ("batch", "ablate"):
        from .silent_dissent import batch, facts as F

        pool = F.load()
        known = None
        if args.known:
            k = pd.read_csv(args.known)
            known = set(k[k.known].uid)
        chosen = F.select(pool, args.per_bridge, args.limit, args.seed,
                          args.uids.split(",") if args.uids else None, known)
        items = batch.prepare(h, chosen, pool, args.seed, args.same_country, args.keep_leaky, verbose=True)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        meta = {"config": args.config, "cmd": args.cmd, "n_facts": len(items), "band": h.band,
                **{k: v for k, v in vars(args).items() if k not in ("cmd",)}}
        (out / f"{args.cmd}_run.json").write_text(json.dumps(meta, indent=1, default=str))
        if args.cmd == "batch":
            batch.run_readouts(h, items, args.conditions.split(","), out, args.paper_fmt)
        else:
            batch.run_ablations(h, items, args.condition, parse_layers(args.layers, h), out,
                                args.roles.split(","), args.where, args.paper_fmt, args.lens)
        show_summary(out, h.band, only=("ablate",) if args.cmd == "ablate" else ("contrasts", "conditions", "probs"))

    elif args.cmd == "summarize":
        show_summary(args.run_dir, parse_layers(args.band, h), full=True)

    elif args.cmd == "known":
        from .silent_dissent.known import known_facts

        df = known_facts(h, args.limit, args.batch_size)
        df.to_csv(args.out, index=False)
        print(f"{df.known.sum()} of {len(df)} facts known; wrote {args.out}")

    elif args.cmd == "mozart":
        from .silent_dissent.mozart import run

        run(h, args.out or f"results/mozart_{cname}", args.peer, parse_layers(args.layers, h))


if __name__ == "__main__":
    sys.exit(main())
