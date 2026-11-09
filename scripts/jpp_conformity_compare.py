"""Conformity headline shares, J-lens versus J++ lens, on Qwen3.5-4B and Qwen3.6-27B (letter questions, the six
calibration questions left out as in the main report). Two views: each lens on its own selected questions, and
both lenses on only the questions both selected (the like-for-like view). Writes results/jpp_compare/conformity.csv.

    uv run python scripts/jpp_conformity_compare.py
"""
import pandas as pd

from consulted_bridges.conformity import CALIBRATION
from consulted_bridges.conformity_stats import relabel, shares

PAIRS = {"Qwen3.6-27B": "qwen36_27b", "Qwen3.5-4B": "qwen35_4b"}
rows = []
for model, cfg in PAIRS.items():
    for wd, tag in (("neutral", "conformity"), ("deferential", "conformity_defer")):
        jl = pd.read_csv(f"results/{tag}_{cfg}/prompts_all.csv")
        jp = pd.read_csv(f"results/{tag}_{cfg}_jpp/prompts_all.csv")
        jl, jp = (p[~p.question.isin(CALIBRATION)] for p in (jl, jp))
        common = set(jl.question) & set(jp.question)
        for view, a, b in (("own questions", jl, jp),
                           ("shared questions", jl[jl.question.isin(common)], jp[jp.question.isin(common)])):
            sa, sb = shares(a, relabel(a)), shares(b, relabel(b))
            for k in sa:
                rows.append(dict(model=model, wording=wd, view=view, share=k,
                                 jlens=f"{sa[k][0]}/{sa[k][1]}", jpp=f"{sb[k][0]}/{sb[k][1]}",
                                 jlens_v=sa[k][0] / sa[k][1] if sa[k][1] else None,
                                 jpp_v=sb[k][0] / sb[k][1] if sb[k][1] else None,
                                 n_questions=f"{a.question.nunique()} / {b.question.nunique()}"))
t = pd.DataFrame(rows)
t.to_csv("results/jpp_compare/conformity.csv", index=False)
pd.set_option("display.width", 200)
for (m, w, v), g in t.groupby(["model", "wording", "view"], sort=False):
    print(f"\n== {m}, {w}, {v} (questions J-lens / J++: {g.n_questions.iloc[0]})")
    print(g[["share", "jlens", "jpp"]].to_string(index=False))
