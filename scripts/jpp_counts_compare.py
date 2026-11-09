"""Count questions on Qwen3.6-27B: J-lens screens (results/counts_qwen36_27b, results/counts_r2_qwen36_27b) against
the J++ screen (results/counts_qwen36_27b_jpp), all trimmed scorecards. For a question in both J-lens rounds (the
anchors), round two is used. Writes results/jpp_compare/counts.csv and prints the summary.

    uv run python scripts/jpp_counts_compare.py
"""
import glob
from pathlib import Path

import pandas as pd


def cards(pattern):
    fs = [f for f in sorted(glob.glob(pattern)) if Path(f).stat().st_size > 1]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


jl = pd.concat([cards("results/counts_qwen36_27b/shard*/cards.csv"),
                cards("results/counts_r2_qwen36_27b/shard*/cards.csv")]).drop_duplicates("sentence", keep="last")
jp = cards("results/counts_qwen36_27b_jpp/shard*/cards.csv")
cols = ["sentence", "bridge", "answer", "model_said", "verdict", "verdict_reason", "reach", "need", "comes_first", "quality"]
t = jl[cols].merge(jp[cols], on=["sentence", "bridge", "answer"], how="outer", suffixes=("_jl", "_jpp"))
t.to_csv("results/jpp_compare/counts.csv", index=False)

short = lambda v: v.fillna("-").str.replace(r" \(KIND NOT CHECKED\)", "", regex=True)
print(f"{len(t)} count questions (J-lens {t.verdict_jl.notna().sum()}, J++ {t.verdict_jpp.notna().sum()})")
print(pd.crosstab(short(t.verdict_jl), short(t.verdict_jpp), rownames=["J-lens"], colnames=["J++"]).to_string())
for name, col in (("J-lens", "reach_jl"), ("J++", "reach_jpp")):
    print(f"edit reaches the bridge (drop >= 0.5), {name}: {(t[col] >= 0.5).sum()} of {t[col].notna().sum()} checked")
cons = lambda v: v.fillna("").str.startswith("CONSULTED")
new = t[cons(t.verdict_jpp) & ~cons(t.verdict_jl)]
lost = t[cons(t.verdict_jl) & ~cons(t.verdict_jpp)]
pd.set_option("display.width", 220)
show = ["sentence", "bridge", "answer", "verdict_jl", "reach_jl", "need_jl", "reach_jpp", "need_jpp", "quality_jpp"]
print(f"\nconsulted with J++ only ({len(new)}):"); print(new[show].to_string(index=False) if len(new) else "-")
print(f"\nconsulted with the J-lens only ({len(lost)}):"); print(lost[show].to_string(index=False) if len(lost) else "-")
print(f"\nconsulted under each: J-lens {cons(t.verdict_jl).sum()}, J++ {cons(t.verdict_jpp).sum()}; answers behind them:")
for name, col in (("J-lens", "verdict_jl"), ("J++", "verdict_jpp")):
    print(f"  {name}:", t[cons(t[col])].groupby(["bridge", "answer"]).size().to_dict())
