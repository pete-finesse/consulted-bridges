"""Conformity on count questions with the J++ lens (Qwen3.6-27B): the new facts J++ made testable (Switzerland 4:
Le Corbusier, Federer; Belgium 3: Herge; Brazil 5, doubtful: Niemeyer, Senna, Coelho) and two references that also
have a J-lens run (Haydn, Jackman). Prints the headline shares per group and wording (the labels of conformity.py),
and one line per prompt for the references under both lenses. Writes results/jpp_compare/counts_conformity.csv.

    uv run python scripts/jpp_counts_conformity.py
"""
import pandas as pd

from consulted_bridges.conformity_stats import relabel, shares

GROUPS = {"Switzerland 4 + Belgium 3 (swaps checked)": ["N_BP_corbusier", "N_BP_federer", "N_BP_herge"],
          "Brazil 5 (doubtful)": ["N_BP_niemeyer", "N_BP_senna", "N_BP_coelho"],
          "references, J++": ["N_BP_haydn", "N_BP_jackman"]}
REFS = GROUPS["references, J++"]
rows, per = [], []
for wd, tag in (("neutral", "conformity_counts"), ("deferential", "conformity_counts_defer")):
    jp = pd.read_csv(f"results/{tag}_qwen36_27b_jpp/prompts_all.csv")
    jl = pd.read_csv(f"results/{tag}_qwen36_27b/prompts_all.csv")
    for g, qs in {**GROUPS, "references, J-lens": REFS}.items():
        p = (jl if g.endswith("J-lens") else jp)
        p = p[p.question.isin(qs)]
        for k, (x, n) in shares(p, relabel(p)).items():
            rows.append(dict(wording=wd, group=g, share=k, value=f"{x}/{n}"))
    for lens, p in (("J-lens", jl), ("J++", jp)):
        q = p[p.question.isin(REFS)]
        per += [dict(wording=wd, lens=lens, **r) for r in
                q[["question", "condition", "peers", "model_said", "state", "label", "need", "pushback"]].to_dict("records")]
t = pd.DataFrame(rows)
t.to_csv("results/jpp_compare/counts_conformity.csv", index=False)
pd.set_option("display.width", 220)
for wd, g in t.groupby("wording", sort=False):
    print(f"\n== {wd}")
    print(g.pivot_table(index="share", columns="group", values="value", aggfunc="first", sort=False).to_string())
r = pd.DataFrame(per)
print("\n== references, one row per prompt")
print(r.pivot_table(index=["wording", "question", "condition", "peers", "model_said", "state"], columns="lens",
                    values="label", aggfunc="first", sort=False).to_string())
