"""Spider legs: can we change the answer away from "eight" by editing J-space words other than "spider"?

Prompt (chat, no prefill): "How many legs do the animals that spin webs have? Answer in one word."
The model must silently think "spider" (never written) to answer "Eight".

1. Baseline: the greedy answer and the top next tokens.
2. Discovery: the J-lens top words at every position (from "How many" on) and layer, and a tally
   of which words show up most, so we know what is in "J-space" here.
3. Tracked words: ranks of spider, eight, six, insect, ... at the last position and over positions.
4. Edits on everything except spider: ablate / swap / steer words like eight, web, arachnid, legs,
   at several layer ranges, over the question's positions. For each edit: the greedy answer,
   P(Eight), P(8), the winning candidate, and whether "spider" is still in J-space afterwards
   (if it is, the answer changed without removing spider). Also the cosine between each edited
   word's direction and spider's, to show we did not secretly edit spider.

Run on the pod:  uv run python scripts/spider_legs.py [--config qwen35_4b] [--out results/spider_qwen35_4b]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch


from consulted_bridges.interventions import Edit, Edit as _Edit, answer_probs, directions, generate  # noqa: E402
from consulted_bridges.load import load  # noqa: E402
from consulted_bridges.readout import readout  # noqa: E402
from consulted_bridges.text import chat, resolve_positions  # noqa: E402
from silent_dissent.model import get_layers, layer_output  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)

USER = "How many legs do the animals that spin webs have? Answer in one word."
CANDIDATES = ["Eight", "Six", "Four", "Two", "Ten", "Seven", "Zero", "8", "6"]
FORMATS = ("{}", "**{}")          # the reply starts right after the chat template, no leading space
TRACK = ["spider", "spiders", "arachnid", "eight", "six", "four", "two", "insect", "insects", "web", "webs",
         "silk", "legs", "ant", "bee", "bird", "nest", "octopus", "crab", "wings", "spin"]
SPIDER = ["spider", "spiders"]    # never edited
POS = "from:How many"             # the user's question and the template tokens after it


def probs(h, text, edits=None):
    r = answer_probs(h, text, CANDIDATES, edits, formats=FORMATS)
    tot = 0.0
    for c, v in r["cands"].items():
        v["p_full"] = sum(v.get(f"p[{f}]", 0.0) for f in FORMATS)
        tot += v["p_full"]
    for v in r["cands"].values():
        v["p_share"] = v["p_full"] / (tot or 1.0)
    return r


def edits_for(L: int) -> dict[str, list[Edit]]:
    """Every edit leaves spider/spiders alone."""
    def Edit(kind, words, layers, pos, **kw):  # clip layers to the model (tiny has 4)
        return _Edit(kind, words, [l for l in layers if l < L] or [L - 1], pos, **kw)

    ranges = {"early": list(range(4, 14)), "mid": list(range(12, 22)), "late": list(range(20, 30)),
              "all": list(range(0, L))}
    runs: dict[str, list[Edit]] = {}

    def add(name, make):
        for rn, layers in ranges.items():
            runs[f"{name} [{rn}]"] = make(layers)

    eight = ["eight", "Eight", "8"]
    add("ablate eight/Eight/8", lambda ls: [Edit("ablate", eight, ls, POS)])
    for new in ["six", "four", "two", "ten"]:
        add(f"swap eight->{new}", lambda ls, new=new: [Edit("swap", ["eight", new], ls, POS),
                                                        Edit("swap", ["Eight", new.capitalize()], ls, POS)])
    add("ablate web/webs", lambda ls: [Edit("ablate", ["web", "webs"], ls, POS)])
    add("swap web->nest", lambda ls: [Edit("swap", ["web", "nest"], ls, POS), Edit("swap", ["webs", "nests"], ls, POS)])
    add("swap web->hive", lambda ls: [Edit("swap", ["web", "hive"], ls, POS), Edit("swap", ["webs", "hives"], ls, POS)])
    add("ablate arachnid", lambda ls: [Edit("ablate", ["arachnid", "arachnids"], ls, POS)])
    add("swap arachnid->insect", lambda ls: [Edit("swap", ["arachnid", "insect"], ls, POS)])
    add("swap legs->wings", lambda ls: [Edit("swap", ["legs", "wings"], ls, POS)])
    add("swap eight->six + arachnid->insect",
        lambda ls: [Edit("swap", ["eight", "six"], ls, POS), Edit("swap", ["Eight", "Six"], ls, POS),
                    Edit("swap", ["arachnid", "insect"], ls, POS)])
    # steering: a push toward a word, over a few layers only (it adds up across layers)
    for word in ["six", "insect", "bird", "four"]:
        for a in [0.05, 0.1, 0.2]:
            for rn, layers in {"mid": list(range(14, 20)), "late": list(range(22, 28))}.items():
                runs[f"steer +{word} a={a} [{rn}]"] = [Edit("steer", [word], layers, POS, alpha=a)]
    for a in [0.05, 0.1, 0.2]:
        for rn, layers in {"mid": list(range(14, 20)), "late": list(range(22, 28))}.items():
            runs[f"steer -eight a={a} [{rn}]"] = [Edit("steer", ["eight"], layers, POS, alpha=-a)]
    return runs


@torch.no_grad()
def capture(h, text):
    blocks = get_layers(h.model)
    ids = h.encode(text)
    store = {}
    hs = [b.register_forward_hook(lambda _m, _i, o, l=l: store.__setitem__(l, layer_output(o)[0].detach()))
          for l, b in enumerate(blocks)]
    try:
        h.model(input_ids=torch.tensor([ids], device=h.device))
    finally:
        for x in hs:
            x.remove()
    return ids, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen35_4b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    h = load(args.config)
    out = Path(args.out or f"results/spider_{Path(args.config).stem}")
    out.mkdir(parents=True, exist_ok=True)
    L = h.n_layers
    global BAND
    BAND = [l for l in range(12, 30) if l < L] or h.band
    text = chat(h, USER, "")
    print("PROMPT:", repr(text))

    # 1. baseline
    base = probs(h, text)
    print("\n== 1. Baseline ==")
    print("greedy answer:", repr(generate(h, text, max_new_tokens=8)))
    print("top next tokens:", ", ".join(f"{t!r} {p:.3f}" for t, p in base["top"]))
    print("candidates p_full:", {c: round(v["p_full"], 4) for c, v in base["cands"].items()})

    # 2. discovery
    ids, store = capture(h, text)
    start = resolve_positions(h, text, POS)[0]
    toks = [h.tok.decode([t]) for t in ids]
    lens = h.lenses["jlens"]
    rows, tally = [], Counter()
    for l in range(L):
        lg = lens.logits(store[l][start:], l).float()
        top = lg.topk(args.topk, dim=-1).indices.cpu().numpy()
        for pi, row in enumerate(top):
            words = [h.tok.decode([int(t)]) for t in row]
            rows.append({"pos": start + pi, "token": toks[start + pi], "layer": l,
                         **{f"top{k}": w for k, w in enumerate(words)}})
            if 8 <= l < L - 1:
                tally.update({w.strip().lower() for w in words if w.strip()})
    disc = pd.DataFrame(rows)
    disc.to_csv(out / "discovery_top_words.csv", index=False)
    print(f"\n== 2. Discovery: J-lens top-{args.topk} words, positions from {toks[start]!r} on ==")
    print("Most frequent words (layers 8-%d, all those positions; count = position x layer cells):" % (L - 2))
    print(", ".join(f"{w}:{c}" for w, c in tally.most_common(90)))
    show = [l for l in (10, 14, 18, 22, 26, 30) if l < L]
    print("\nTop 6 words per position at layers", show)
    for p in range(start, len(ids)):
        cells = []
        for l in show:
            r = disc[(disc.pos == p) & (disc.layer == l)].iloc[0]
            cells.append(f"L{l}: " + "|".join(str(r[f"top{k}"]).strip() for k in range(6)))
        print(f"{p:3d} {toks[p]!r:12s} " + "   ".join(cells))

    # 3. tracked words
    ro = readout(h, text, TRACK, lenses=["jlens", "logit"], dedupe=False)
    summ = ro.summary(BAND, start)
    summ.to_csv(out / "tracked_words.csv", index=False)
    print("\n== 3. Tracked words (rank 0 = top; band 12-29) ==")
    print(summ[summ.lens == "jlens"].to_string(index=False))
    print("\nWhere 'spider' ranks best (J-lens):")
    print(ro.where_best("spider", top=8).to_string(index=False))
    print("\nWhere 'eight' ranks best (J-lens):")
    print(ro.where_best("eight", top=8).to_string(index=False))

    # 4. edits
    runs = edits_for(L)
    CL = [l for l in (8, 16, 24) if l < L] or [L - 1]
    sp_dirs = {l: directions(h, Edit("ablate", SPIDER, [l]), l) for l in CL}
    res = []
    print(f"\n== 4. Edits (positions: {POS!r}; spider never edited) ==")
    for name, edits in runs.items():
        try:
            r = probs(h, text, edits)
        except ValueError as err:          # a word with no usable token piece
            print(f"{name:42s} -> skipped: {err}")
            continue
        ans = generate(h, text, edits, max_new_tokens=6)
        c = r["cands"]
        win = max(c, key=lambda k: c[k]["p_full"])
        ro2 = readout(h, text, ["spider", "eight"], lenses=["jlens"], edits=edits, dedupe=False)
        s2 = ro2.summary(BAND, start).set_index("word")
        # max |cos| between any edited word's directions and spider's, at layers 8/16/24
        cos = 0.0
        for e in edits:
            for l in CL:
                for D in directions(h, Edit("ablate", e.words, [l], lens=e.lens), l):
                    for S in sp_dirs[l]:
                        cos = max(cos, float((D.to(S.device) @ S.T).abs().max()))
        row = {"run": name, "answer": ans.strip(), "winner": win,
               "P_Eight": c["Eight"]["p_full"], "P_8": c["8"]["p_full"], "P_winner": c[win]["p_full"],
               "spider_rank_last_best": s2.loc["spider", "last_best_rank"],
               "spider_rank_anypos_best": s2.loc["spider", "allpos_best_rank"],
               "eight_rank_last_best": s2.loc["eight", "last_best_rank"],
               "max_cos_with_spider": round(cos, 3),
               "top_tokens": ", ".join(f"{t!r} {p:.2f}" for t, p in r["top"][:3])}
        res.append(row)
        print(f"{name:42s} -> {ans.strip()!r:14s} P(Eight)={row['P_Eight']:.3f} P(8)={row['P_8']:.3f} "
              f"win={win}:{row['P_winner']:.2f} spider_rank(last/any)={row['spider_rank_last_best']}/"
              f"{row['spider_rank_anypos_best']} cos={row['max_cos_with_spider']}")
        sys.stdout.flush()
    df = pd.DataFrame(res)
    df.to_csv(out / "edits.csv", index=False)
    flips = df[~df.answer.str.lower().str.contains("eight|8", regex=True)]
    print(f"\n== Answer changed away from eight in {len(flips)} of {len(df)} edits ==")
    print(flips[["run", "answer", "P_Eight", "spider_rank_last_best", "spider_rank_anypos_best",
                 "max_cos_with_spider"]].to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
