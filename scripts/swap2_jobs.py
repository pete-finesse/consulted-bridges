"""Questions for the swap test (docs/swap_test_rules.md): where the bridge is needed and comes first in the saved v4.1
cards (beats noise >= 0.5 and comes first), with each question's saved answer onset. Same files and order as the
question bank (consulted_bridges.bank), plus the close variants.

  uv run python scripts/swap2_jobs.py      -> results/swap2_jobs/<config>.csv
"""
import glob
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
SOURCES = {
    "qwen35_4b": [("bank", ["card4_qwen35_4b/shard*/cards.csv"]), ("variants", ["prop_qwen35_4b/shard*/cards.csv"])],
    "qwen36_27b": [("bank", ["big_qwen36_27b/shard*/cards.csv", "big_raw_qwen36_27b/shard*/cards.csv",
                             "v4_27b/shard*/cards.csv"]), ("variants", ["prop_qwen36_27b/shard*/cards.csv"])],
}

out = R / "swap2_jobs"
out.mkdir(exist_ok=True)
for cfg, groups in SOURCES.items():
    rows = []
    for group, globs in groups:
        fs = [f for g in globs for f in sorted(glob.glob(str(R / g)))]
        c = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates("sentence", keep="last")
        c = c[~c.sentence.str.startswith("v2_")]          # the bank leaves out the round-2 copies
        c = c[(c.beats_noise >= 0.5) & (c.comes_first == True)]  # noqa: E712
        rows.append(c.assign(group=group)[["sentence", "group", "answer", "answer_onset", "quality"]])
    d = pd.concat(rows)
    d["answer_onset"] = d.answer_onset.astype(int)
    d.to_csv(out / f"{cfg}.csv", index=False)
    print(cfg, d.groupby("group").size().to_dict())
