"""Recompute test 5 (no collateral damage) for finished v2 cards, after the fix that adds
"Answer in one word." to the two unrelated questions. Patches tests.csv and cards.csv in place.
  uv run python scripts/card2_collateral.py results/card2_qwen35_4b
"""
import glob, re, sys
from pathlib import Path
import pandas as pd
from consulted_bridges.legacy.bridge_card2 import Card, UNRELATED, TH, q, fmt_layers
from consulted_bridges.questions.v2 import SPECS
from consulted_bridges.load import load
from consulted_bridges.text import chat
h = load("qwen35_4b")
for d in sorted(glob.glob(sys.argv[1] + "/shard*")):
    tp, cp = Path(d) / "tests.csv", Path(d) / "cards.csv"
    tests, cards = pd.read_csv(tp), pd.read_csv(cp)
    for i, r in tests[tests.test.str.startswith("5 ")].iterrows():
        s = r.sentence
        m = re.search(r"layers (\d+)–(\d+)", r.what_happened)
        lay = list(range(int(m.group(1)), int(m.group(2)) + 1))
        c = Card(h, s, SPECS[s])
        coll, parts = 0.0, []
        for uq, uforms in UNRELATED:
            ut = chat(h, uq, "")
            u0 = c.P(ut, uforms); u1 = c.P(ut, uforms, c.rm(SPECS[s]["bridge"], lay, "from:" + " ".join(uq.split()[:2])))
            dd = c.drop(u0, u1); coll = max(coll, dd)
            parts.append(f"{q(uq)} → P({q(uforms[0])}) {u0:.2f} → {u1:.2f}")
        tests.loc[i, "value"] = round(coll, 3)
        tests.loc[i, "result"] = "pass" if coll <= TH["collateral"] else "fail"
        tests.loc[i, "what_happened"] = f"Same edit ({fmt_layers(lay)}): " + "; ".join(parts) + f". Biggest drop {coll:.2f} (pass ≤ {TH['collateral']})."
        cards.loc[cards.sentence == s, "collateral"] = round(coll, 3)
        v = cards.loc[cards.sentence == s, "verdict"].iloc[0]
        print(f"{s}: collateral {coll:.3f} ({'; '.join(parts)}) verdict {v}", flush=True)
        if coll > TH["collateral"] and v.startswith("CONSULTED"):
            cards.loc[cards.sentence == s, ["verdict", "verdict_reason"]] = ["UNCLEAR", "needed and first, but the edit also damages unrelated questions"]
    tests.to_csv(tp, index=False); cards.to_csv(cp, index=False)
print("ALLDONE", flush=True)
