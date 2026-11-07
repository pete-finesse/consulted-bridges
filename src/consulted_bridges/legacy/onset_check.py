"""Is the answer onset pulled early by overlap with the bridge?

The answer onset is the first 6-layer window where removing the answer's direction kills the answer.
If the answer's direction overlaps the bridge's (Austria with Salzburg), removing "Austria" also damages
the bridge, so the answer can die because the bridge was hit, not because the answer had formed.
Fix: also find the onset from the answer's OWN part, "Austria minus Salzburg".

Per window: drop when removing the bridge, the whole answer direction, and the answer minus the bridge.

  uv run python scripts/onset_check.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


from consulted_bridges.legacy.bridge_card3 import Card  # noqa: E402
from consulted_bridges.interventions import Edit  # noqa: E402
from consulted_bridges.load import load  # noqa: E402

S2 = importlib.import_module("consulted_bridges.questions.v2").SPECS
S3 = importlib.import_module("consulted_bridges.questions.core").SPECS
SC = importlib.import_module("consulted_bridges.questions.ideas.culture").SPECS
QUESTIONS = {"B2_mozart": S2["B2_mozart"], "A2_christ_redeemer": S2["A2_christ_redeemer"],
             "A2_cn_tower": S2["A2_cn_tower"], "B2_messi": S2["B2_messi"], "B2_gates": S2["B2_gates"],
             "X_spider": S3["X_spider"], "C_sport_hockey": SC["C_sport_hockey"]}


def main():
    h = load("qwen35_4b")
    rows = []
    for name, S in QUESTIONS.items():
        c = Card(h, name, S)
        forms = S["answers"][S["answer"]]
        b = S["bridge"]
        base = c.P(c.text, forms)
        prof = []
        for w in c.wins:
            db = c.drop(base, c.P(c.text, forms, c.rm(b, w)))
            da = c.drop(base, c.P(c.text, forms, c.rm(S["answer_dirs"], w)))
            do = c.drop(base, c.P(c.text, forms, [Edit("ablate_excl", [a] + b, w, c.ALL) for a in S["answer_dirs"]]))
            prof.append((w[0], db, da, do))
            rows.append(dict(question=name, window=f"{w[0]}-{w[-1]}", start=w[0], bridge=round(db, 3),
                             answer=round(da, 3), answer_own=round(do, 3)))
        first = lambda k: next((p[0] for p in prof if p[k] >= 0.5), None)  # noqa: E731
        ob, oa, oo = first(1), first(2), first(3)
        line = " ".join(f"L{p[0]}:{p[1]:.2f}/{p[2]:.2f}/{p[3]:.2f}" for p in prof)
        print(f"{name}: bridge onset {ob} | answer onset {oa} | answer-own onset {oo} | "
              f"comes first (old) {ob is not None and oa is not None and ob <= oa - 3} | "
              f"comes first (own) {ob is not None and oo is not None and ob <= oo - 3}", flush=True)
        print(f"   per window bridge/answer/answer-own: {line}", flush=True)
    pd.DataFrame(rows).to_csv("results/onset_check_qwen35_4b.csv", index=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
