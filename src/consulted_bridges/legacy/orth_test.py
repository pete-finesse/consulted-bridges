"""Remove only the part of a bridge's J-lens direction that is not shared with the answer.

"Salzburg minus Austria": Salzburg's direction with the Austria direction projected out. If the answer
still collapses, the bridge matters beyond being a copy of the answer. Same for each control
("Vienna minus Austria"), and with the strongest same-kind word excluded too ("Salzburg minus
Austria and Vienna": what is specific to Salzburg among Austrian cities).

  uv run python scripts/orth_test.py --spec mozart mozart_direct spider
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


from consulted_bridges.legacy.bridge_test import FORMATS, SPECS  # noqa: E402
from consulted_bridges.interventions import Edit, answer_probs, directions, generate  # noqa: E402
from consulted_bridges.load import load  # noqa: E402
from consulted_bridges.text import chat  # noqa: E402

pd.set_option("display.width", 250)

EXTRA = {   # answer directions to exclude, and the same-kind word to also exclude
    "mozart": dict(exclude=["Austria"], same_kind="Vienna",
                   controls=["Vienna", "Graz", "Innsbruck", "Tokyo", "Wolfgang", "music"]),
    "mozart_direct": dict(exclude=["Austria"], same_kind="Vienna",
                          controls=["Vienna", "Graz", "Tokyo", "Wolfgang", "music"]),
    "spider": dict(exclude=["eight", "Eight"], same_kind="insects",
                   controls=["insects", "animals", "dogs", "legs", "webs"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", default=["mozart", "mozart_direct", "spider"])
    ap.add_argument("--config", default="qwen35_4b")
    args = ap.parse_args()
    h = load(args.config)
    L = h.n_layers
    clip = lambda ls: [l for l in ls if l < L] or [L - 1]  # noqa: E731
    for name in args.spec:
        S, X = SPECS[name], EXTRA[name]
        out = Path(f"results/orth_{name}_{Path(args.config).stem}")
        out.mkdir(parents=True, exist_ok=True)
        text = chat(h, S["user"], "")
        ALL = f"from:{S['q_start']}"
        forms = S["answers"][S["answer"]]

        def p_ans(edits):
            r = answer_probs(h, text, forms, edits, formats=FORMATS)
            return sum(v.get(f"p[{f}]", 0.0) for v in r["cands"].values() for f in FORMATS)

        base = p_ans([])
        b = S["bridge"][0]
        # how much of each direction survives the projection (1 = nothing shared with the answer)
        with torch.no_grad():
            kept = {}
            for w in [b] + X["controls"]:
                vals = []
                for l in clip([12, 18, 24]):
                    D = directions(h, Edit("ablate", [w], [l]), l)[0][:1]
                    E = torch.cat(directions(h, Edit("ablate", X["exclude"], [l]), l))
                    Q, _ = torch.linalg.qr(E.T)
                    vals.append(float((D - (D @ Q) @ Q.T).norm()))
                kept[w] = sum(vals) / len(vals)
        print(f"\n######## {name}: {S['user']!r}  baseline P({S['answer']}) = {base:.2f}")
        print("Share of each direction NOT shared with the answer (norm after projection):",
              ", ".join(f"{w}={v:.2f}" for w, v in kept.items()))
        targets = {f"{b} (full)": lambda ls: [Edit("ablate", S["bridge"], ls, ALL)],
                   f"{b} minus answer": lambda ls: [Edit("ablate_excl", [b] + X["exclude"], ls, ALL)],
                   f"{b} minus answer+{X['same_kind']}":
                       lambda ls: [Edit("ablate_excl", [b] + X["exclude"] + [X["same_kind"]], ls, ALL)]}
        for c in X["controls"]:
            targets[f"{c} (full)"] = lambda ls, c=c: [Edit("ablate", [c], ls, ALL)]
            targets[f"{c} minus answer"] = lambda ls, c=c: [Edit("ablate_excl", [c] + X["exclude"], ls, ALL)]
        wins = {f"L{s}-{min(s + 5, L - 1)}": clip(range(s, s + 6)) for s in range(0, L - 3, 3)}
        wins["mid 12-21"] = clip(range(12, 22))
        rows = []
        for tname, make in targets.items():
            for wn, ls in wins.items():
                p = p_ans(make(ls))
                rows.append({"target": tname, "win": wn, "P": p})
            ans = generate(h, text, make(clip(range(12, 22))), max_new_tokens=5).replace("<|im_end|>", "").strip()
            rows.append({"target": tname, "win": "answer@mid", "P": ans})
            print(f"  {tname:34s} mid 12-21: P={rows[-2]['P']:.2f} answer={ans!r}")
            sys.stdout.flush()
        df = pd.DataFrame(rows)
        df.to_csv(out / "orth.csv", index=False)
        grid = df[df.win != "answer@mid"].pivot_table(index="target", columns="win", values="P", sort=False,
                                                      aggfunc="first")
        grid = grid[[c for c in wins]]
        print(f"\nP({S['answer']}) after removal (baseline {base:.2f}):")
        print(grid.astype(float).round(2).to_string())
        pd.Series(kept).to_csv(out / "kept_share.csv")
    print("wrote results/orth_*")


if __name__ == "__main__":
    main()
