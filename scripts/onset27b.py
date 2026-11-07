"""Step 1 of bringing the Qwen3.6-27B scorecard to v4: which questions' answer onset moves when it is found
from the answer's OWN part (its direction minus the bridge's) instead of the whole answer direction?

Uses bridge_card4's Card (depth-scaled windows, raw answer tokens, two-consecutive-window onset rule).
Only questions where the onset moves need a full v4 re-score (step 2); for the rest every number is unchanged.

  uv run python scripts/onset27b.py --config qwen36_27b --spec L_colosseum ... --out results/onset27b.csv
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd


from consulted_bridges.scorecard import TH, Card  # noqa: E402
from consulted_bridges.questions.core import SPECS as S3  # noqa: E402
from consulted_bridges.interventions import Edit  # noqa: E402
from consulted_bridges.load import load  # noqa: E402


def onset(prof, wins):
    i = next((i for i in range(len(prof)) if all(d >= TH["gate"] for d in prof[i:])), None)   # sustained, as bridge_card4
    return None if i is None else wins[i][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen36_27b")
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--out", default="results/onset27b_qwen36_27b.csv")
    args = ap.parse_args()
    specs = dict(S3)
    for m in ("consulted_bridges.questions.ideas.sequence", "consulted_bridges.questions.ideas.language", "consulted_bridges.questions.ideas.science", "consulted_bridges.questions.ideas.culture"):
        specs.update(importlib.import_module(m).SPECS)
    h = load(args.config)
    rows = []
    for name in args.spec:
        S = specs[name]
        c = Card(h, name, S)
        forms = S["answers"][S["answer"]]
        dirs = list(dict.fromkeys(S["answer_dirs"] + ["@" + f for f in forms]))
        b = S["bridge"]
        base = c.P(c.text, forms)
        whole = [c.drop(base, c.P(c.text, forms, c.rm(dirs, w))) for w in c.wins]
        own = [c.drop(base, c.P(c.text, forms, [Edit("ablate_excl", [a] + b, w, c.ALL) for a in dirs])) for w in c.wins]
        ow, oo = onset(whole, c.wins), onset(own, c.wins)
        rows.append(dict(question=name, onset_whole=ow, onset_own=oo, moved=ow != oo,
                         whole=" ".join(f"{d:.2f}" for d in whole), own=" ".join(f"{d:.2f}" for d in own)))
        print(f"[{name}] onset whole {ow} → own part {oo}{'  MOVED' if ow != oo else ''}", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
