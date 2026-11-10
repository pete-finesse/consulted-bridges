"""Figure 2 of the rewritten paper: how often the answer survives without the bridge, by number of agreeing peers.

"Survives" means the correct answer keeps a probability of at least 0.5 after the bridge is removed (run B).
Every number is read from the 2x2 tables of runs 1 to 3. Run from the repository root:

    .venv/bin/python paper/make_fig2.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "fig2_survival.svg"
BLUE, ORANGE, GREY = "#1f5fae", "#c2410c", "#5b6470"

SERIES = [  # label, run folder, colour, line style, marker, marker fill
    ("Qwen3.6-27B", "v2_run2_qwen36_27b", BLUE, "-", "o", BLUE),
    ("Qwen3.6-27B, new questions", "v2_run3_qwen36_27b", BLUE, "--", "o", "white"),
    ("Qwen3.5-4B", "v2_run1_qwen35_4b", ORANGE, "-", "^", ORANGE),
]
PANELS = [("neutral", "Neutral instruction: “Give your own answer.”"),
          ("defer", "Group-aware instruction: “Taking the other\nagents' answers into account, give your answer.”")]


def shares(run, frame):
    d = pd.read_csv(ROOT / "results" / run / "twobytwo.csv")
    if "calibration" in d:
        d = d[d.calibration == False]  # noqa: E712
    d = d[d.frame == frame]
    out = []
    for cond, k in [("hidden", 3), ("agree", 1), ("agree", 2), ("agree", 3)]:
        s = d[(d.condition == cond) & (d.peers == k)]
        out.append((int((s.P_right_B >= 0.5).sum()), len(s)))
    return out


def main():
    plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8.5,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), sharey=True)
    for ax, (frame, title) in zip(axes, PANELS):
        ax.axvspan(-0.45, 0.5, color="#f1f2f4", lw=0)
        ends = []
        for label, run, col, ls, mk, mfc in SERIES:
            pts = shares(run, frame)
            pct = [100 * a / n for a, n in pts]
            ax.plot([1, 2, 3], pct[1:], ls, color=col, marker=mk, ms=5.5, mfc=mfc, mec=col, lw=1.4, label=label,
                    clip_on=False, zorder=3)
            ax.plot([0], pct[:1], ls="none", marker=mk, ms=5.5, mfc=mfc, mec=col, clip_on=False, zorder=3)
            ends.append((pct[3], f"{pts[3][0]}/{pts[3][1]}", col))
        y_prev = -100
        for y, lab, col in sorted(ends):
            y_lab = max(y, y_prev + 9)
            ax.annotate(lab, (3, y), xytext=(7, 0), textcoords=("offset points"), va="center", fontsize=7.5,
                        color=col, xycoords="data", annotation_clip=False) if y_lab == y else ax.text(
                3.13, y_lab, lab, va="center", fontsize=7.5, color=col)
            y_prev = y_lab
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["3 peers,\nanswers hidden", "1", "2", "3"])
        ax.set_xlim(-0.45, 3.55)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="#e3e5e8", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8.5, loc="left", fontweight="bold")
        ax.text(2, -27, "agreeing peers", ha="center", va="top", fontsize=8.5, color="#222")
        ax.tick_params(length=2.5, color=GREY)
    axes[0].set_ylabel("questions whose answer survives\nwithout the bridge (%)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.2, 1.0), frameon=False, fontsize=7.5, handlelength=2.6)
    fig.subplots_adjust(left=0.1, right=0.965, top=0.83, bottom=0.25, wspace=0.12)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
