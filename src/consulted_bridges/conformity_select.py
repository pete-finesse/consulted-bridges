"""Pick a new model's conformity questions from its own trimmed solo scorecard, with the rule that picked the
Qwen lists (conformity.QUESTIONS): letter questions where the bridge is needed (need >= 0.5) and comes first.
Writes results/conformity_select_<config>.csv (every letter question, its scorecard numbers and `selected`).
--set candidates applies the same rule to the 171 v2 candidates (conformity.CANDIDATE) screened into
results/solo_<config>_candidates, and writes results/conformity_select_<config>_candidates.csv (with family and
dev / confirm split columns); the letter-question files are untouched.

  uv run python -m consulted_bridges.conformity_select --config llama31_8b_it [--solo results/solo_<config>] [--set candidates]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from consulted_bridges.conformity import CANDIDATE, LETTER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--solo", default=None)
    ap.add_argument("--set", default="letters", choices=["letters", "candidates"])
    args = ap.parse_args()
    cfg = Path(args.config).stem
    cand = args.set == "candidates"
    sfx = "_candidates" if cand else ""
    d = args.solo or f"results/solo_{cfg}{sfx}"
    fs = [f for f in sorted(glob.glob(f"{d}/shard*/cards.csv")) if Path(f).stat().st_size > 1]
    c = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    c["question"] = c.sentence.str.replace(r"^v3_", "", regex=True)
    if cand:   # one card per candidate (a re-run shard may repeat one; keep the last)
        c = c[c.question.isin(CANDIDATE)].drop_duplicates("question", keep="last")
    missing = sorted(set(CANDIDATE if cand else LETTER) - set(c.question))
    if missing:
        raise SystemExit(f"solo scorecard has no card for {missing}")
    c["selected"] = (c.need >= 0.5) & (c.comes_first == True)  # noqa: E712
    cols = ["question", "selected", "verdict", "verdict_reason", "model_said", "P_base", "need", "comes_first",
            "lead", "answer_onset", "quality"]
    c = c[[x for x in cols if x in c]].sort_values("question")
    if cand:
        from consulted_bridges.questions.proposed.candidates import family, split
        conf = set(split()["confirm"])
        c.insert(1, "family", c.question.map(family))
        c.insert(2, "split", c.question.map(lambda q: "confirm" if q in conf else "dev"))
    c.to_csv(f"results/conformity_select_{cfg}{sfx}.csv", index=False)
    if cand:   # the confirmation half is reported as counts only (scope.md §6)
        for sp, x in c.groupby("split"):
            print(f"{sp}: {int(x.selected.sum())} of {len(x)} selected")
        return
    print(c.verdict.value_counts().to_string())
    print(f"{int(c.selected.sum())} of {len(c)} letter questions selected:", " ".join(c[c.selected].question))


if __name__ == "__main__":
    main()
