"""Merge scorecard v2 shards into the five tables the results page reads.

  uv run python scripts/card2_page_data.py results/card2_qwen35_4b OUTDIR
"""
import glob
import sys
from pathlib import Path

import pandas as pd

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
load = lambda n: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(src / "shard*" / f"{n}.csv")))
                            if Path(f).stat().st_size > 1], ignore_index=True)
cards, tests, swaps, controls, profile = (load(n) for n in ("cards", "tests", "swaps", "controls", "profile"))
from consulted_bridges.questions.v2 import SPECS  # noqa: E402
order = {k: i for i, k in enumerate(SPECS)}
GROUP = {"E": "E · description riddle", "A1": "A1 · landmark in the capital", "A2": "A2 · landmark in another city",
         "B1": "B1 · born in the capital", "B2": "B2 · famous, born in their own country",
         "B3": "B3 · born abroad", "C1": "C1 · university in a capital", "C2": "C2 · university in another city"}
cards["order"] = cards.sentence.map(order)
cards = cards.sort_values("order")
cards["group_name"] = cards.group.map(GROUP)
# the line Peter wants to read straight off: the needed-early result, or the gate that failed
key = {}
for s, t in tests.groupby("sentence"):
    t = t.sort_values("order")
    t1 = t[t.test.str.startswith("1 ")]
    fails = t[(t.result == "fail") & t.test.str.startswith("Gate")]
    key[s] = (t1.what_happened.iloc[0] if len(t1) else (fails.what_happened.iloc[0] if len(fails) else ""))
cards["key_result"] = cards.sentence.map(key)
cards["verdict"] = cards.verdict.str.capitalize().str.replace("(kind not checked)", "(kind not checked)", regex=False)
keep = ["sentence", "order", "group", "group_name", "predicted", "question", "bridge", "answer", "model_said", "P_base",
        "hop1_question", "hop1_said", "hop2_question", "hop2_said", "answer_onset", "bridge_onset", "needed_early",
        "noise", "beats_noise", "comes_first", "beats_kind", "collateral", "spread_16", "spread_all", "own_part",
        "swaps_follow", "same_answer_keeps", "fallback", "verdict", "verdict_reason", "key_result", "words_skipped"]
cards = cards[[c for c in keep if c in cards]]
cards.to_csv(dst / "cards.csv", index=False)
tests.sort_values(["sentence", "order"]).to_csv(dst / "tests.csv", index=False)
swaps.to_csv(dst / "swaps.csv", index=False)
controls.to_csv(dst / "controls.csv", index=False)
profile.sort_values(["sentence", "start"]).to_csv(dst / "profile.csv", index=False)
print(cards[["sentence", "verdict", "key_result"]].to_string(index=False))
