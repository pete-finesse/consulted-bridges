"""Summarise the conformity follow-up (conformity_routes.py): merge shards, print the tables its report uses:
the swap test and hidden-letter prompts first, then the partial copy-block attempt that was stopped.
Calibration questions (the six the reading thresholds were set on) are counted apart.

  uv run python -m consulted_bridges.conformity_routes_report
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from consulted_bridges.conformity_routes import main_dir

ROOT = Path("results/conformity_routes")
RUNS = [("qwen36_27b", "neutral"), ("qwen36_27b", "defer"), ("qwen35_4b", "neutral"), ("qwen35_4b", "defer")]


def merge(d, name):
    fs = [f for f in sorted(glob.glob(str(d / "shard*" / name))) if Path(f).stat().st_size > 1]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def show(title, df):
    print(f"\n{title}\n{df.to_string()}")


def swap_report(cfg, frame):
    d = ROOT / f"swap_{frame}_{cfg}"
    sw, hd, mv = merge(d, "swap.csv"), merge(d, "hidden.csv"), merge(d, "swap_moves.csv")
    if sw.empty:
        return
    sw.to_csv(d / "swap_all.csv", index=False)
    hd.to_csv(d / "hidden_all.csv", index=False)
    mv.to_csv(d / "swap_moves_all.csv", index=False)
    m = pd.read_csv(main_dir(cfg, frame) / "prompts_all.csv")
    print(f"\n==================== SWAP {cfg} {frame}: {len(sw)} prompts, {len(hd)} hidden-letter prompts ====================")
    print(f"baseline P(right) matches the main run: max |diff| {abs(sw.P_right - sw.P_right_main).max():.4f}")
    sw["cell"] = sw.condition + sw.peers.astype(str)
    for s_, cal in (("new", False), ("calibration", True)):
        q = sw[sw.calibration == cal]
        if q.empty:
            continue
        print(f"\n--- {s_} questions ({q.question.nunique()}) ---")
        c = q[q.counts]
        print(f"settings that count (an everyday word readable, none gets through): {len(c)}/{len(q)}")
        show("Swap: targets that follow / informative targets, and median shift, by prompt",
             c.groupby("cell").agg(prompts=("question", "size"), follow=("n_follow", "sum"), targets=("n_targets", "sum"),
                                   share=("follow_share", "mean"), shift=("median_shift", "median")).round(2))
        g = c[c.condition != "alone"]
        show("Swap by main-run label",
             g.groupby(["condition", "main_label"]).agg(prompts=("question", "size"), follow=("n_follow", "sum"),
                                                        targets=("n_targets", "sum"), share=("follow_share", "mean"),
                                                        shift=("median_shift", "median")).round(2))
        show("Controls: everyday words through, source controls following, source max shift (median)",
             q.groupby("cell").agg(through=("everyday_through", "sum"), readable=("everyday_readable", "sum"),
                                   source=("source_follow", "sum"), src_shift=("source_max_shift", "median")).round(2))
        h = hd[hd.calibration == cal] if len(hd) else hd
        if len(h):
            mm = m[m.condition.isin(["alone", "agree"])].copy()
            mm["col"] = mm.condition + mm.peers.astype(int).astype(str)
            piv = mm.pivot_table(index="question", columns="col", values="need").reindex(columns=["alone0", "agree3"])
            j = h.set_index("question")[["model_said", "state", "P_right", "need", "label"]].join(piv)
            show("Hidden letters (3 peers): label", j.label.value_counts().to_frame())
            show("Hidden letters: says the right letter", j.state.value_counts().to_frame())
            show("Hidden letters: median need, against alone and 3 agreeing peers",
                 j[["need", "alone0", "agree3"]].median().round(2).to_frame())
            show("Hidden letters: per question", j.round(2))


def report(cfg, frame):
    d = ROOT / f"{frame}_{cfg}"
    r, hd = merge(d, "routes.csv"), merge(d, "hidden.csv")
    if r.empty:
        return
    r.to_csv(d / "routes_all.csv", index=False)
    hd.to_csv(d / "hidden_all.csv", index=False)
    m = pd.read_csv(main_dir(cfg, frame) / "prompts_all.csv")
    print(f"\n==================== {cfg} {frame}: {len(r)} edited prompts, {len(hd)} hidden-letter prompts ====================")
    print(f"baseline P(right) matches the main run: max |diff| {abs(r.P_right - r.P_right_main).max():.4f}")
    for s, cal in (("new", False), ("calibration", True)):
        a = r[(r.condition == "agree") & (r.calibration == cal)]
        g = r[(r.condition == "disagree") & (r.calibration == cal)]
        print(f"\n--- {s} questions ---")
        if len(a):
            show("Agree: route by main-run label", pd.crosstab(a.main_label, a.route, margins=True))
            show("Agree: route by peers", pd.crosstab(a.peers, a.route))
            show("Agree: medians", a.groupby("main_label")[["P_right", "P_right_block", "copy_need", "need_after_block",
                                                            "both_P_right"]].median().round(2))
        if len(g):
            show("Give-ins: route by main-run label", pd.crosstab(g.main_label, g.route, margins=True))
            show("Give-ins: copy block vs everyday words (median dLO; negative = toward the right letter)",
                 g.groupby("main_label")[["LO", "copy_dLO", "copy_noise_dLO", "LO_block", "pushback_after_block",
                                          "main_pushback"]].median().round(2))
        h = hd[hd.calibration == cal] if len(hd) else hd
        if len(h):
            mm = m[m.condition.isin(["alone", "agree"])].copy()
            mm["col"] = mm.condition + mm.peers.astype(int).astype(str)
            piv = mm.pivot_table(index="question", columns="col", values="need").reindex(columns=["alone0", "agree3"])
            j = h.set_index("question")[["model_said", "state", "P_right", "need", "label"]].join(piv)
            show("Hidden letters (3 peers): label", j.label.value_counts().to_frame())
            show("Hidden letters: says the right letter", j.state.value_counts().to_frame())
            show("Hidden letters: median need, against alone and 3 agreeing peers",
                 j[["need", "alone0", "agree3"]].median().round(2).to_frame())
            show("Hidden letters: per question", j.round(2))


if __name__ == "__main__":
    for cfg, frame in RUNS:
        swap_report(cfg, frame)
    print("\n\n######## Stopped copy-block attempt (partial) ########")
    for cfg, frame in RUNS:
        report(cfg, frame)
