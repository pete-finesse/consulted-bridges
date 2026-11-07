"""Stricter specificity check: remove only the part of the bridge NOT shared with ANY of its same-kind controls.

For each question, in the layers before the answer forms (best window):
  left         share of the bridge's direction outside the controls' span (1 = nothing shared)
  all controls drop from removing every control together (their whole shared span)
  bridge only  drop from removing the bridge minus the span of all controls
If "bridge only" stays high, the model uses something only this bridge has.

  uv run python scripts/spec_check2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch


from consulted_bridges.legacy.bridge_card3 import Card  # noqa: E402
from consulted_bridges.questions.core import SPECS  # noqa: E402
from consulted_bridges.interventions import Edit, directions  # noqa: E402
from consulted_bridges.load import load  # noqa: E402
from consulted_bridges.legacy.spec_check import CHECKS  # noqa: E402


def main():
    h = load("qwen35_4b")
    rows = []
    for name, ctrls in CHECKS.items():
        S = SPECS[name]
        c = Card(h, name, S)
        ctrls = [w for w in ctrls if c.one_token(w)]
        forms = S["answers"][S["answer"]]
        base = c.P(c.text, forms)
        pa = [c.drop(base, c.P(c.text, forms, c.rm(S["answer_dirs"], w))) for w in c.wins]
        onset = c.wins[next(i for i, d in enumerate(pa) if d >= 0.5)][0]
        early = [w for w in c.wins if w[-1] < onset]
        b = S["bridge"]

        def best(make):
            return max((c.drop(base, c.P(c.text, forms, make(w))), w) for w in early)

        bfull, wb = best(lambda w: c.rm(b, w))
        allc, wa = best(lambda w: c.rm(ctrls, w))
        bonly, wo = best(lambda w: [Edit("ablate_excl", [bb] + ctrls, w, c.ALL) for bb in b])
        with torch.no_grad():
            left = []
            for l in sorted({l for w in early for l in w}):
                D = directions(h, Edit("ablate", [b[0]], [l]), l)[0][:1].float()
                E = torch.cat(directions(h, Edit("ablate", ctrls, [l]), l)).float()
                Q, _ = torch.linalg.qr(E.T)
                left.append(float((D - (D @ Q) @ Q.T).norm()))
        left = sum(left) / len(left)
        f = lambda w: f"layers {w[0]}–{w[-1]}"  # noqa: E731
        rows.append(dict(question=name, bridge=b[0], answer=S["answer"], controls=" + ".join(ctrls), left=round(left, 3),
                         bridge_drop=round(bfull, 3), all_controls_drop=round(allc, 3), bridge_only_drop=round(bonly, 3),
                         specificity_strict=round(bonly / bfull, 3) if bfull else None))
        print(f"{name}: remove {b[0]!r} ({f(wb)}): drop {bfull:.2f} | remove {' + '.join(ctrls)} together ({f(wa)}): "
              f"drop {allc:.2f} | {b[0]!r} minus all of them ({f(wo)}): drop {bonly:.2f} | share of {b[0]!r} left: {left:.2f}",
              flush=True)
    pd.DataFrame(rows).to_csv("results/spec_check2_qwen35_4b.csv", index=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
