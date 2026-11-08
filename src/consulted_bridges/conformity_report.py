"""Summarise the conformity experiment: merge the shards and print the tables the report uses.
The six calibration questions (the reading thresholds were set on them) are always counted apart.

  uv run python -m consulted_bridges.conformity_report results/conformity_qwen36_27b [more dirs]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd


def merge(d):
    fs = sorted(glob.glob(str(Path(d) / "shard*" / "prompts.csv")))
    p = pd.concat([pd.read_csv(f) for f in fs if Path(f).stat().st_size > 1], ignore_index=True)
    p["set"] = p.calibration.map({True: "calibration", False: "new"})
    return p


def table(p, rows, cols):
    return pd.crosstab([p[r] for r in rows], p[cols], dropna=False).to_string()


def report(d):
    p = merge(d)
    p.to_csv(Path(d) / "prompts_all.csv", index=False)
    print(f"\n=== {d}: {p.question.nunique()} questions, {len(p)} prompts ===")
    for s in ("new", "calibration"):
        q = p[p.set == s]
        if q.empty:
            continue
        print(f"\n--- {s} questions ({q.question.nunique()}) ---")
        a = q[q.condition == "alone"]
        print("\nAlone: labels\n" + a.label.value_counts().to_string())
        g = q[q.condition == "agree"]
        print("\nPeers agree: label by number of peers\n" + table(g, ["peers"], "label"))
        print("\nPeers agree: median need by peers\n" + g.groupby("peers").need.median().round(2).to_string())
        dis = q[q.condition == "disagree"]
        print("\nPeers disagree: state by number of peers\n" + table(dis, ["peers"], "state"))
        print("\nPeers disagree: label by state\n" + table(dis, ["state"], "label"))
        print("\nPeers disagree: median pushback (bridge dLO - noise) by state\n" +
              dis.groupby("state").pushback.median().round(2).to_string())
        both = pd.concat([g, dis])
        if "present" in both:
            both = both.assign(cell=both.present.map({True: "present", False: "not present"}).fillna("no reading"))
            print("\nPresence x use (agree and disagree)\n" + table(both, ["condition", "cell"], "label"))
        c = q[q.condition == "copy"]
        if len(c):
            print("\nCopy-only control: pushback\n" + c[["question", "said", "pushback", "label"]].to_string(index=False))
    return p


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
