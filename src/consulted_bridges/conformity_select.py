"""Pick a new model's conformity questions from its own trimmed solo scorecard, with the rule that picked the
Qwen lists (conformity.QUESTIONS): letter questions where the bridge is needed (need >= 0.5) and comes first.
Writes results/conformity_select_<config>.csv (every letter question, its scorecard numbers and `selected`).

  uv run python -m consulted_bridges.conformity_select --config llama31_8b_it [--solo results/solo_<config>]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from consulted_bridges.conformity import LETTER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--solo", default=None)
    args = ap.parse_args()
    cfg = Path(args.config).stem
    d = args.solo or f"results/solo_{cfg}"
    fs = [f for f in sorted(glob.glob(f"{d}/shard*/cards.csv")) if Path(f).stat().st_size > 1]
    c = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    c["question"] = c.sentence.str.replace(r"^v3_", "", regex=True)
    missing = sorted(set(LETTER) - set(c.question))
    if missing:
        raise SystemExit(f"solo scorecard has no card for {missing}")
    c["selected"] = (c.need >= 0.5) & (c.comes_first == True)  # noqa: E712
    cols = ["question", "selected", "verdict", "verdict_reason", "model_said", "P_base", "need", "comes_first",
            "lead", "answer_onset", "quality"]
    c = c[[x for x in cols if x in c]].sort_values("question")
    c.to_csv(f"results/conformity_select_{cfg}.csv", index=False)
    print(c.verdict.value_counts().to_string())
    print(f"{int(c.selected.sum())} of {len(c)} letter questions selected:", " ".join(c[c.selected].question))


if __name__ == "__main__":
    main()
