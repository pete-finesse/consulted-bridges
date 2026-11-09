"""J-lens versus J++ lens on the 37 letter questions (NEXT_EXPERIMENTS 5), Qwen3.5-4B and Qwen3.6-27B.

J-lens numbers come from the question bank (data/artifacts/question_bank/results.csv: full v4.1 scorecard, the
lists in conformity.QUESTIONS); J++ numbers from the trimmed solo scorecard (results/solo_<cfg>_jpp) and its
selection (results/conformity_select_<cfg>_jpp.csv, same rule: need >= 0.5 and the bridge comes first).
Writes results/jpp_compare/solo.csv (one row per question and model) and prints the summary.

    uv run python scripts/jpp_compare.py
"""
import glob
from pathlib import Path

import pandas as pd

from consulted_bridges.conformity import LETTER, QUESTIONS

MODELS = {"qwen35_4b": "Qwen3.5-4B", "qwen36_27b": "Qwen3.6-27B"}
bank = pd.read_csv("data/artifacts/question_bank/results.csv")
out = Path("results/jpp_compare"); out.mkdir(parents=True, exist_ok=True)
rows = []
for cfg, name in MODELS.items():
    fs = [f for f in sorted(glob.glob(f"results/solo_{cfg}_jpp/shard*/cards.csv")) if Path(f).stat().st_size > 1]
    if not fs:
        print(f"{name}: no J++ solo scorecard yet"); continue
    jpp = pd.concat([pd.read_csv(f) for f in fs]).assign(id=lambda d: d.sentence.str.replace(r"^v3_", "", regex=True))
    sel = pd.read_csv(f"results/conformity_select_{cfg}_jpp.csv").set_index("question").selected
    b = bank[bank.model == name].set_index("id")
    for q in LETTER:
        j = jpp[jpp.id == q].iloc[0]
        r = b.loc[q] if q in b.index else None
        rows.append({
            "model": name, "question": q,
            "jl_verdict": None if r is None else r.verdict, "jpp_verdict": j.verdict,
            "jl_need": None if r is None else r.need, "jpp_need": j.need,
            "jl_reach": None if r is None else r.reach, "jpp_reach": j.reach,
            "jl_selected": q in QUESTIONS[cfg], "jpp_selected": bool(sel.get(q, False)),
        })
t = pd.DataFrame(rows)
t.to_csv(out / "solo.csv", index=False)
for name, g in t.groupby("model", sort=False):
    jl, jp = set(g[g.jl_selected].question), set(g[g.jpp_selected].question)
    # the bank stores the reach check as pass / fail / "not reached" (an earlier gate stopped the card);
    # the solo cards store the drop itself (pass = drop >= 0.5, NaN = not reached)
    jl_meas = g.jl_reach.isin(["pass", "fail"]); jp_meas = g.jpp_reach.notna()
    reach_jl = (g.jl_reach == "pass").sum(); reach_jp = (g.jpp_reach >= 0.5).sum()
    print(f"\n== {name}: {len(g)} letter questions")
    print(f"selected for conformity: J-lens {len(jl)}, J++ {len(jp)}, both {len(jl & jp)}")
    print(f"  J++ only: {' '.join(sorted(jp - jl)) or '-'}")
    print(f"  J-lens only: {' '.join(sorted(jl - jp)) or '-'}")
    print(f"edit reaches the bridge (drop >= 0.5): J-lens {reach_jl}/{jl_meas.sum()} checked, "
          f"J++ {reach_jp}/{jp_meas.sum()} checked")
    both_meas = g[jl_meas & jp_meas]
    print(f"  on questions checked under both: J-lens {(both_meas.jl_reach == 'pass').sum()}, "
          f"J++ {(both_meas.jpp_reach >= 0.5).sum()} of {len(both_meas)}")
    both = g.dropna(subset=["jl_need", "jpp_need"])
    print(f"need, median: J-lens {both.jl_need.median():.2f}, J++ {both.jpp_need.median():.2f}; "
          f"correlation {both.jl_need.corr(both.jpp_need):.2f} over {len(both)}")
    print(pd.crosstab(g.jl_verdict.fillna("-"), g.jpp_verdict.fillna("-")).to_string())
