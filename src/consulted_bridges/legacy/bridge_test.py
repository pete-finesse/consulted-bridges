"""Does the model consult a hidden bridge word to decide its answer? A general version of spider_bridge.py.

For a one-word-answer chat prompt whose middle step is never written (spider; Salzburg):
0. Baseline answer and the bridge's J-lens ranks; which other words are about as strongly present.
A. Remove the bridge's J-lens direction in sliding 6-layer windows (positions: question onward),
   next to removing controls: hand-picked ones (e.g. other cities of the same country) and words
   chosen automatically to be about as strongly present as the bridge.
B. Swap the bridge for alternatives with a known answer (e.g. Salzburg -> Munich => Germany), with
   swaps that should NOT change the answer (Salzburg -> Innsbruck => still Austria).
C. Swap the controls for the same alternative: if those also change the answer, swaps just plant.
D. The same bridge edits on the question's own tokens vs the template tokens after it.

  uv run python scripts/bridge_test.py --spec mozart [--config qwen35_4b]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch


from consulted_bridges.interventions import Edit, answer_probs, directions, generate  # noqa: E402
from consulted_bridges.load import load  # noqa: E402
from consulted_bridges.readout import readout  # noqa: E402
from consulted_bridges.text import chat, resolve_positions  # noqa: E402
from silent_dissent.model import get_layers, layer_output  # noqa: E402

pd.set_option("display.width", 250)

SPECS = {
    "mozart": dict(
        user="In which country is the city where Wolfgang Amadeus Mozart was born? Answer in one word.",
        bridge=["Salzburg"], answer="Austria",
        answers={"Austria": ["Austria", "Österreich"], "Germany": ["Germany"], "France": ["France"],
                 "Hungary": ["Hungary"], "Italy": ["Italy"], "Switzerland": ["Switzerland"],
                 "Czech": ["Czech", "Czechia"], "Poland": ["Poland"], "Spain": ["Spain"], "Japan": ["Japan"]},
        swaps={"Munich": "Germany", "Paris": "France", "Budapest": "Hungary", "Milan": "Italy",
               "Zurich": "Switzerland", "Prague": "Czech", "Krakow": "Poland", "Madrid": "Spain",
               "Osaka": "Japan", "Innsbruck": "Austria", "Graz": "Austria", "Linz": "Austria"},
        controls=["Vienna", "Graz", "Innsbruck", "Tokyo", "Mozart", "music", "composer", "Wolfgang"],
        plant="Budapest",
        q_span="In which country is the city where Wolfgang Amadeus Mozart was born? Answer in one word.",
        q_start="In which"),
    "mozart_direct": dict(   # shortcut check: no city in the question at all
        user="Which country was Wolfgang Amadeus Mozart from? Answer in one word.",
        bridge=["Salzburg"], answer="Austria",
        answers={"Austria": ["Austria"], "Germany": ["Germany"], "France": ["France"], "Hungary": ["Hungary"],
                 "Italy": ["Italy"], "Switzerland": ["Switzerland"], "Czech": ["Czech"]},
        swaps={"Munich": "Germany", "Budapest": "Hungary", "Milan": "Italy", "Innsbruck": "Austria"},
        controls=["Vienna", "Graz", "Tokyo", "music", "composer"],
        plant="Budapest",
        q_span="Which country was Wolfgang Amadeus Mozart from? Answer in one word.", q_start="Which country"),
    "spider": dict(
        user="How many legs do the animals that spin webs have? Answer in one word.",
        bridge=["spider", "spiders"], answer="8",
        answers={"8": ["Eight", "8"], "6": ["Six", "6"], "4": ["Four", "4"], "2": ["Two", "2"],
                 "10": ["Ten", "10"], "0": ["Zero", "None", "0"]},
        swaps={"bee": "6", "dog": "4", "cat": "4", "ant": "6", "bird": "2", "crab": "10", "octopus": "8"},
        controls=["animals", "legs", "humans", "dogs", "sentence", "word"],
        plant="bird",
        q_span="How many legs do the animals that spin webs have? Answer in one word.", q_start="How many"),
}
FORMATS = ("{}", "**{}")
BLOCK = {"answer", "word", "one"}   # auto-control words to skip besides the bridge/answer


@torch.no_grad()
def capture(h, text):
    blocks = get_layers(h.model)
    store = {}
    hs = [b.register_forward_hook(lambda _m, _i, o, l=l: store.__setitem__(l, layer_output(o)[0].detach()))
          for l, b in enumerate(blocks)]
    try:
        h.model(input_ids=torch.tensor([h.encode(text)], device=h.device))
    finally:
        for x in hs:
            x.remove()
    return store


@torch.no_grad()
def strength(h, store, word, layers, pos):
    """Max over positions of |component on the word's unit J-lens direction| / |h|, mean over layers."""
    v = []
    for l in layers:
        D = directions(h, Edit("ablate", [word], [l]), l)[0].to(store[l].device)
        H = store[l][pos].float()
        v.append(((H @ D.T).abs().max(-1).values / H.norm(dim=-1)).max().item())
    return sum(v) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="mozart", choices=list(SPECS))
    ap.add_argument("--config", default="qwen35_4b")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-auto", type=int, default=8, help="automatic strength-matched controls")
    args = ap.parse_args()
    S = SPECS[args.spec]
    h = load(args.config)
    L = h.n_layers
    clip = lambda ls: [l for l in ls if l < L] or [L - 1]  # noqa: E731
    out = Path(args.out or f"results/bridge_{args.spec}_{Path(args.config).stem}")
    out.mkdir(parents=True, exist_ok=True)
    text = chat(h, S["user"], "")
    ALL, Q, TPL = f"from:{S['q_start']}", f"span:{S['q_span']}", "from:<|im_end|>"
    flat = {f: a for a, forms in S["answers"].items() for f in forms}
    cand = list(flat)

    def classify(ans):
        a = ans.replace("<|im_end|>", "").strip(" .*\n").lower()
        for f, k in sorted(flat.items(), key=lambda x: -len(x[0])):
            if a.startswith(f.lower()):
                return k
        return "other"

    rows = []

    def run(section, name, edits, expect=None):
        try:
            r = answer_probs(h, text, cand, edits, formats=FORMATS)
        except ValueError as err:
            print(f"[{section}] {name:48s} -> skipped: {err}")
            return
        p = {}
        for f, k in flat.items():
            p[k] = p.get(k, 0.0) + sum(r["cands"][f].get(f"p[{fm}]", 0.0) for fm in FORMATS)
        ans = generate(h, text, edits, max_new_tokens=6).replace("<|im_end|>", "").strip()
        got = classify(ans)
        row = {"section": section, "run": name, "answer": ans, "class": got, "expect": expect,
               "P_answer": p[S["answer"]], "P_expect": p.get(expect) if expect else None}
        rows.append(row)
        e = f" expect={expect} P={row['P_expect']:.2f}" if expect else ""
        print(f"[{section}] {name:48s} -> {ans!r:16s} P({S['answer']})={row['P_answer']:.2f}{e}")
        sys.stdout.flush()

    print("PROMPT:", repr(text))
    run("base", "no edit", [])

    # 0. readout + strength-matched controls
    store = capture(h, text)
    allpos = resolve_positions(h, text, ALL)
    mid = clip(range(12, 22))
    ro = readout(h, text, S["bridge"] + S["controls"], lenses=["jlens"], dedupe=False)
    print("\nJ-lens ranks (best over positions from the question on; band 12-29):")
    print(ro.summary(clip(range(12, 30)), allpos[0])[["word", "last_best_rank", "allpos_best_rank",
                                                         "allpos_band_logrank"]].to_string(index=False))
    print("\nWhere the bridge ranks best:")
    print(ro.where_best(S["bridge"][0], top=6).to_string(index=False))
    tally = Counter()
    lens = h.lenses["jlens"]
    for l in clip(range(8, 24)):
        top = lens.logits(store[l][allpos], l).float().topk(20, dim=-1).indices
        for row_ in top.tolist():
            tally.update({d.strip() for d in (h.tok.decode([t]) for t in row_) if d.startswith(" ")})
    skip = {w.lower() for w in S["bridge"] + list(flat) + S["controls"] + list(S["swaps"])} | BLOCK
    names = [n.lower() for n in skip]
    pool = [w for w, _ in tally.most_common(200) if w.isalpha() and len(w) > 3 and w.lower() not in skip
            and not any(n.startswith(w.lower()) or w.lower().startswith(n) for n in names)][:40]
    sb = strength(h, store, S["bridge"][0], mid, allpos)
    st = {}
    for w in pool + S["controls"]:
        try:
            st[w] = strength(h, store, w, mid, allpos)
        except ValueError:
            pass
    auto = sorted([w for w in pool if w in st], key=lambda w: abs(st[w] - sb))[: args.n_auto]
    print(f"\nStrength (fraction of |h|, layers 12-21): bridge {S['bridge'][0]}={sb:.3f}; "
          + ", ".join(f"{w}={st[w]:.3f}" for w in S["controls"] + auto if w in st))
    controls = S["controls"] + auto
    pd.Series({S["bridge"][0]: sb, **st}).to_csv(out / "strength.csv")

    bridge_ablate = lambda ls, pos: [Edit("ablate", S["bridge"], clip(ls), pos)]  # noqa: E731
    bridge_swap = lambda o, ls, pos: [Edit("swap", [S["bridge"][0], o], clip(ls), pos)]  # noqa: E731

    print("\n== A. Remove, sliding 6-layer windows (question onward) ==")
    for s in range(0, L, 3):
        win = list(range(s, min(s + 6, L)))
        tag = f"L{win[0]}-{win[-1]}"
        run("A", f"remove {S['bridge'][0]} {tag}", bridge_ablate(win, ALL))
        for c in controls:
            run("A", f"control remove {c} {tag}", [Edit("ablate", [c], clip(win), ALL)])
        if s + 6 >= L:
            break

    ranges = {"mid 12-21": mid, "8-23": clip(range(8, 24)), "all": list(range(L))}
    print("\n== B. Swap the bridge for alternatives with a known answer ==")
    for rn, ls in ranges.items():
        for o, exp in S["swaps"].items():
            run("B", f"swap {S['bridge'][0]}->{o} [{rn}]", bridge_swap(o, ls, ALL), exp)

    print(f"\n== C. Swap controls -> {S['plant']} (does a swap just plant it?) ==")
    exp_plant = S["swaps"].get(S["plant"])
    for rn, ls in ranges.items():
        for c in controls:
            run("C", f"swap {c}->{S['plant']} [{rn}] (strength {st.get(c, float('nan')):.3f})",
                [Edit("swap", [c, S["plant"]], clip(ls), ALL)], exp_plant)

    print("\n== D. Question tokens vs template tokens ==")
    for pn, pos in {"question": Q, "template": TPL}.items():
        for rn, ls in ranges.items():
            run("D", f"remove {S['bridge'][0]} [{rn}] @{pn}", bridge_ablate(ls, pos))
            run("D", f"swap {S['bridge'][0]}->{S['plant']} [{rn}] @{pn}", bridge_swap(S["plant"], ls, pos), exp_plant)

    df = pd.DataFrame(rows)
    df.to_csv(out / "edits.csv", index=False)
    a = df[df.section == "A"]
    a = a.assign(target=a.run.str.replace(r" L\d+-\d+$", "", regex=True).str.replace("control remove ", "")
                 .str.replace("remove ", ""), win=a.run.str.extract(r"(L\d+-\d+)$")[0])
    grid = a.pivot_table(index="target", columns="win", values="P_answer", sort=False)
    grid = grid[sorted(grid.columns, key=lambda c: int(c[1:].split("-")[0]))]
    print(f"\n== Summary A: P({S['answer']}) after removing each word, per layer window ==")
    print(grid.round(2).to_string())
    b = df[df.section == "B"]
    print(f"\nB: swaps that should change the answer: {((b['class'] != S['answer']) & (b.expect != S['answer'])).sum()}"
          f" of {(b.expect != S['answer']).sum()} moved away from {S['answer']}; "
          f"{(b['class'] == b.expect).sum()} of {len(b)} gave exactly the expected answer")
    print("wrote", out)


if __name__ == "__main__":
    main()
