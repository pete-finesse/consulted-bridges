"""Draw every paper figure from the run tables. Run from the repository root:

    .venv/bin/python paper/figures/make_figures.py

Writes one SVG and one PNG per figure into paper/figures/. Every plotted number is read from a file named in
SOURCES below (and in README.md); numbers typed in by hand are only those the run reports print as text tables
(the probability-rule labels and the disagree tables), each with its file and table named next to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
R1 = ROOT / "results/v2_run1_qwen35_4b"
R2 = ROOT / "results/v2_run2_qwen36_27b"

# ---- style: plain, greyscale-legible; colour is never the only channel -------------------------------------------
INK, INK2, MUTED, RULE, FAINT = "#0b0b0b", "#52514e", "#8a8984", "#c9c8c2", "#ecebe7"
BLUE, BLUE_D, BLUE_L, ORANGE, GREY = "#2a78d6", "#1c5cab", "#6da7ec", "#eb6834", "#a9a8a2"
SURF = "#ffffff"

plt.rcParams.update({
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "grid.color": FAINT,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "svg.fonttype": "none", "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

MODELS = {"4b": ("Qwen3.5-4B", R1, 20), "27b": ("Qwen3.6-27B", R2, 19)}
WORDING = {"neutral": "neutral", "defer": "group-aware", "deferential": "group-aware"}
SERIES = {  # alone / hidden / agree: colour + marker + line style, so each reads in greyscale
    "alone": dict(color=INK, marker="o", ls="-", label="alone"),
    "hidden": dict(color=BLUE, marker="s", ls="--", label='hidden twin ("?")'),
    "agree": dict(color=ORANGE, marker="^", ls="-", label="peers agree"),
}
READ = {  # the 2x2 readings, in the order scope.md §11 takes them
    "C well below A (leaned on the peers)": ("leaned", ORANGE, INK, "////"),
    "only D collapses (peers the backup)": ("only D", BLUE_L, INK, None),
    "both collapse (bridge needed either way)": ("both", BLUE_D, "#ffffff", None),
    "neither collapses": ("neither", FAINT, INK, None),
    "only B collapses": ("only B", GREY, INK, None),
}
READ_ORDER = list(READ)


def save(fig, name):
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text="", fc=SURF, ec=INK2, lw=0.8, ls="-", size=7.5, color=INK, weight="normal",
        ha="center", hatch=None, r=0.02, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                                ls=ls, hatch=hatch))
    if text:
        tx = x + w / 2 if ha == "center" else x + 0.012
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=size, color=color, weight=weight, **kw)


def arrow(ax, a, b, color=INK2, lw=0.9, ls="-", style="-|>", ms=7, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls, **kw))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    return fig, ax


# ---- Figure 1: each claim and the tests under it (method-map.html §5, plan.md §6) --------------------------------
def fig_claims():
    rows = [  # question, test it rests on, state on 4B, 27B development, 27B confirmation (run 3)
        ("Does the answer need the bridge alone?", "the gate: need and reach", "done", "done", "done"),
        ("Is a readable bridge a used bridge?", "presence (rank) against removal", "done", "done", "not read"),
        ("Does removal still work in group prompts?", "hidden twins, dose-response, B in hidden 2×2",
         "partial", "partial", "done"),
        ("Do agreeing peers weaken the bridge?", "need in log-odds, agree vs. hidden twin",
         "done", "done", "done"),
        ("Are the peers a backup or a replacement?", "the 2×2 (block × removal); the patch", "done", "done", "done"),
        ("Does the bridge pull against wrong peers?", "pull labels; where D's answer goes",
         "partial", "partial", "done"),
        ("When are the peers read?", "patch layer sweep", "partial", "partial", "partial"),
        ("Does the answer follow a swapped bridge?", "swap toward another bridge", "—", "—", "done"),
        ("Do the findings hold on new questions?", "20 frozen confirmation questions", "—", "—", "done"),
    ]
    style = {"done": (BLUE_D, "#ffffff", "done"), "partial": (BLUE_L, INK, "partial"),
             "not read": (SURF, INK2, "not read"), "—": (SURF, MUTED, "not run")}
    fig, ax = blank((7.0, 3.75))
    xs = [0.0, 0.37, 0.73, 0.82, 0.91]
    ws = [0.37, 0.36, 0.09, 0.09, 0.09]
    heads = ["Question", "Test it rests on", "4B", "27B dev", "27B conf."]
    top, rh = 0.93, 0.095
    for x, w, h in zip(xs, ws, heads):
        ax.text(x + (w / 2 if x > 0.7 else 0.004), top + 0.03, h, ha="center" if x > 0.7 else "left",
                va="bottom", fontsize=7.5, weight="bold")
    ax.plot([0, 1.0], [top + 0.02, top + 0.02], color=INK2, lw=0.8)
    for i, (claim, test, *st) in enumerate(rows):
        y = top - (i + 1) * rh
        ax.text(0.004, y + rh / 2, claim, va="center", fontsize=6.9)
        ax.text(xs[1], y + rh / 2, test, va="center", fontsize=6.6, color=INK2)
        for x, w, s in zip(xs[2:], ws[2:], st):
            fc, tc, label = style[s]
            ec = MUTED if fc == SURF else fc
            box(ax, x + 0.006, y + 0.018, w - 0.012, rh - 0.036, label, fc=fc, ec=ec, lw=0.6,
                ls="-", size=6.6, color=tc, r=0.012)
        ax.plot([0, 1.0], [y, y], color=FAINT, lw=0.6)
    ax.text(0.004, top - len(rows) * rh - 0.045,
            "done: run and reported. partial: the original rule failed in one wording, the reading is confounded, or "
            "too few questions to read;\nfor the patch sweep the peers are read in the same layers as the bridge acts "
            "(4B), not clearly after. not read: measured, not reported.\nSources: method-map.html §5, plan.md §6, "
            "run summaries 1–3.",
            fontsize=6.6, color=INK2, va="top")
    save(fig, "fig01_claims_tests")


# ---- Figure 2: one prompt, tokens along, layers up (method-map.html §2) -------------------------------------------
def fig_prompt_layers():
    fig, ax = blank((7.0, 3.9))
    toks = [("Several AI\nagents…", 0.08), ("Question: What letter …\nColosseum … start with?", 0.2),
            ("Agent\n2:", 0.05), ("I", 0.03), ("Agent\n3:", 0.05), ("I", 0.03), ("Agent\n4:", 0.05), ("I", 0.03),
            ("You are Agent 1. Give your\nown answer. Answer with\none letter.", 0.2), ("", 0.035)]
    x0, ybase, ytop = 0.04, 0.17, 0.86       # layer 0 at ybase, layer 64 at ytop
    L = lambda l: ybase + (ytop - ybase) * l / 64  # noqa: E731
    xs = []
    x = x0
    for t, w in toks:
        xs.append((x, w))
        x += w + 0.006
    xend = x
    # token row
    for (t, w), (xx, _) in zip(toks, xs):
        peer = t == "I"
        box(ax, xx, 0.055, w, 0.095, t if t else "", fc=FAINT if not peer else "#fde3d7",
            ec=ORANGE if peer else RULE, lw=0.9 if peer else 0.6, size=6.0 if len(t) > 3 else 7, r=0.008)
    lastx, lastw = xs[-1]
    ax.text(lastx + lastw / 2, 0.105, "→", ha="center", va="center", fontsize=8)
    ax.text(lastx + lastw / 2, 0.035, "last", ha="center", va="center", fontsize=6.4, color=INK2)
    # layer axis
    ax.annotate("", xy=(0.018, ytop + 0.02), xytext=(0.018, ybase), arrowprops=dict(arrowstyle="-|>", color=INK2,
                                                                                     lw=0.8))
    for l in (0, 32, 54, 64):
        ax.text(0.012, L(l), str(l), ha="right", va="center", fontsize=6.5, color=INK2)
    ax.text(0.0, ytop + 0.05, "layer (27B: 64)", fontsize=6.8, color=INK2)
    # peer block: the peers' answer tokens, unreadable from later positions at every layer
    for i in (3, 5, 7):
        xx, w = xs[i]
        ax.add_patch(Rectangle((xx, ybase), w, ytop - ybase, fc="none", ec=ORANGE, lw=0.8, hatch="////"))
    # removal window: question's first word to the end of the prompt, one 6-layer window before the onset
    qx = xs[1][0]
    ax.add_patch(Rectangle((qx, L(42)), xend - qx - 0.006, L(54) - L(42) - 0.004, fc=BLUE, ec=BLUE_D, lw=0.8,
                           alpha=0.35))
    ax.text(qx + 0.005, L(48), "removal: the bridge's direction taken out, one 12-layer window (4B: 6)", fontsize=6.8,
            va="center", color=INK, bbox=dict(fc="#d4e4f7", ec="none", pad=0.6))
    # onset
    ax.plot([x0, xend], [L(54), L(54)], color=INK, lw=0.9, ls=(0, (4, 2)))
    ax.text(qx + 0.005, L(54) + 0.018, "answer onset (27B: layer 54); the window must end before it", fontsize=6.8,
            va="bottom", bbox=dict(fc=SURF, ec="none", pad=0.6))
    ax.text(qx + 0.005, L(24), "early layers: every window (27B: 12 layers, stride 6) that ends\nbefore the onset is tried; the best one counts",
            fontsize=6.6, color=INK2, va="center", bbox=dict(fc=SURF, ec="none", pad=0.6))
    # read-out
    ax.plot([lastx + lastw / 2], [ytop], marker="o", ms=6, color=INK)
    ax.text(lastx + lastw + 0.005, ytop - 0.035, "read P(right),\nP(peers') here", ha="left", va="top", fontsize=6.8)
    # legend for the hatch
    xb = xs[3][0]
    ax.text(xs[2][0], ytop + 0.035, "peer block (hatched): no later token reads the\npeers' answers, at every layer",
            fontsize=6.8, color=INK, ha="left")
    ax.text(0.04, 0.0, "Schematic, not to scale; layer numbers are the 27B's. The window drawn is one of those tried.\n"
            "The patch writes the hidden twin's state (peers say \"?\") into the hatched columns instead of blocking.",
            fontsize=6.4, color=INK2, va="top")
    _ = xb
    save(fig, "fig02_prompt_layers")


# ---- Figure 3: the gate funnel (funnel.csv, J++ lens, both halves) ----------------------------------------------
FAM = {"letters": ["C_LM", "C_LU", "C_LP"], "number lookups": ["C_MN", "C_SP", "C_EL", "C_PL", "C_IN"],
       "other categories": ["C_CT", "C_DS", "C_VC", "C_EO"]}
FAM_STYLE = {"letters": (BLUE_D, None), "number lookups": (BLUE_L, None), "other categories": (GREY, "....")}


def fig_funnel():
    stages = [("drafted", "drafted"), ("knows both hops", "knows it"), ("need ≥ 0.5, bridge first", "selected"),
              ("+ reach ≥ 0.5", "sel + reach")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharex=True)
    for ax, key in zip(axes, ["4b", "27b"]):
        name, rdir, _ = MODELS[key]
        f = pd.read_csv(rdir / "funnel.csv")
        f = f[f.config.str.endswith("_jpp") & (f.family != "all")]
        tot = {}
        for i, (lab, col) in enumerate(stages):
            left = 0
            for g, fams in FAM.items():
                v = int(f[f.family.isin(fams)][col].sum())
                fc, hatch = FAM_STYLE[g]
                ax.barh(i, v, left=left, height=0.62, color=fc, hatch=hatch, edgecolor=SURF, lw=1.0)
                if v >= 8:
                    ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=6.8,
                            color="#ffffff" if fc == BLUE_D else INK)
                left += v
            tot[lab] = left
            ax.text(left + 2, i, str(left), va="center", fontsize=7.2, weight="bold")
        ax.set_yticks(range(len(stages)), [s[0] for s in stages])
        ax.invert_yaxis()
        ax.set_xlim(0, 190)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("candidate questions")
        ax.set_title(f"{name}, J++ lens, {tot['drafted']} candidates")
    handles = [Rectangle((0, 0), 1, 1, fc=FAM_STYLE[g][0], hatch=FAM_STYLE[g][1], ec=SURF) for g in FAM]
    fig.legend(handles, [f"{g}" for g in FAM], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig03_gate_funnel")


# ---- Figure 4: the peer block and the patch (walkthrough.html §3–§4) ----------------------------------------------
def fig_block_patch():
    fig, ax = blank((7.0, 3.3))
    labels = ["Question: …", "Agent 2:", "I", "Agent 3:", "I", "Agent 4:", "I", "You are\nAgent 1. …", "last"]
    widths = [0.12, 0.07, 0.03, 0.07, 0.03, 0.07, 0.03, 0.14, 0.045]

    def row(x0, y, labs=labels, peer_fc="#fde3d7", peer_txt=None):
        xs, x = [], x0
        for i, (t, w) in enumerate(zip(labs, widths)):
            peer = t in ("I", "?")
            box(ax, x, y, w, 0.075, peer_txt if (peer and peer_txt) else t, fc=peer_fc if peer else FAINT,
                ec=ORANGE if peer else RULE, lw=0.9 if peer else 0.6, size=6.6, r=0.008)
            xs.append(x + w / 2)
            x += w + 0.005
        return xs

    # (a) full attention
    ax.text(0.0, 0.97, "a  Full attention: the mask cuts every look back at a peer's answer",
            fontsize=7.6, weight="bold", va="top")
    xs = row(0.02, 0.66)
    last = xs[-1]
    for i, xc in enumerate(xs[:-1]):
        cut = labels[i] == "I"
        h = 0.06 + 0.012 * (len(xs) - i)
        arrow(ax, (last, 0.735), (xc, 0.735), color=ORANGE if cut else INK2, lw=0.8, ls="--" if cut else "-",
              style="-|>", ms=6, connectionstyle=f"arc3,rad={0.22 + 0.02 * i}")
        if cut:
            ax.text(xc + 0.01, 0.80 + 0.005 * (8 - i), "×", color=ORANGE, fontsize=9, ha="center", va="center",
                    weight="bold")
    # (b) linear attention
    ax.text(0.0, 0.53, "b  Linear attention: the running memory passes the peer positions unchanged",
            fontsize=7.6, weight="bold", va="top")
    xs = row(0.02, 0.33)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a, 0.445), (b, 0.445), color=INK2, lw=0.8, ms=5)
    for i, xc in enumerate(xs):
        if labels[i] == "I":
            ax.text(xc, 0.32, "writes\nnothing", ha="center", va="top", fontsize=5.8, color=ORANGE)
    ax.text(0.02, 0.2, "at a peer's answer token: input zeroed, nothing written (beta = 0), no fade (decay = 0)",
            fontsize=6.6, color=INK2, va="top")
    # (c) patch
    ax.text(0.73, 0.97, "c  The patch, a second method", fontsize=7.6, weight="bold", va="top")
    plab = ["Agent 2:", "?", "→", "Agent 2:", "I"]
    box(ax, 0.745, 0.79, 0.09, 0.07, "Agent 2:", fc=FAINT, ec=RULE, size=6.6, r=0.008)
    box(ax, 0.84, 0.79, 0.035, 0.07, "?", fc="#dbe8fa", ec=BLUE, size=7, r=0.008)
    ax.text(0.89, 0.825, 'hidden twin,\nrun first', fontsize=6.2, va="center", color=INK2)
    box(ax, 0.745, 0.60, 0.09, 0.07, "Agent 2:", fc=FAINT, ec=RULE, size=6.6, r=0.008)
    box(ax, 0.84, 0.60, 0.035, 0.07, "I", fc="#fde3d7", ec=ORANGE, size=7, r=0.008)
    ax.text(0.89, 0.635, "agree prompt,\nscored", fontsize=6.2, va="center", color=INK2)
    arrow(ax, (0.8575, 0.785), (0.8575, 0.675), color=BLUE, lw=1.0, ms=7)
    ax.text(0.745, 0.55, "The residual at the peer positions\nis copied in from the twin, at every\n"
            "layer (or from layer L up: the\nsweep). The peers stay readable\nbut say nothing.", fontsize=6.4,
            va="top", color=INK)
    ax.text(0.745, 0.25, "Leak test, both models: the peers\nsaying I or F give the same output\n"
            "under the block (difference 0.0,\n7 of 7 pairs).", fontsize=6.4, va="top", color=INK2)
    _ = plab
    save(fig, "fig04_block_patch")


# ---- Figure 5: the 2x2 on one prompt (walkthrough.html §2, scope.md §11) ------------------------------------------
def fig_2x2_design():
    fig, ax = blank((7.0, 4.0))
    gx, gy, cw, ch = 0.12, 0.12, 0.17, 0.3
    ax.text(gx + cw / 2, gy + 2 * ch + 0.07, "peers visible", ha="center", fontsize=7.5, weight="bold")
    ax.text(gx + cw * 1.5 + 0.01, gy + 2 * ch + 0.07, "peers blocked", ha="center", fontsize=7.5, weight="bold")
    ax.text(gx - 0.012, gy + 1.5 * ch, "bridge\nintact", ha="right", va="center", fontsize=7.5, weight="bold")
    ax.text(gx - 0.012, gy + 0.5 * ch, "bridge\nremoved", ha="right", va="center", fontsize=7.5, weight="bold")
    cells = [("A", "no edit\nthe prompt as written", 0, 1, SURF), ("C", "peer block on\nfall from A: dC", 1, 1, FAINT),
             ("B", "bridge removed\nfall from A: dB", 0, 0, "#dbe8fa"), ("D", "both on\nfall from A: dD", 1, 0, "#c3d8f3")]
    for k, txt, cx, cy, fc in cells:
        x, y = gx + cx * (cw + 0.01), gy + cy * (ch + 0.01)
        box(ax, x, y, cw, ch, "", fc=fc, ec=INK2, lw=0.8, r=0.01)
        ax.text(x + 0.012, y + ch - 0.03, k, fontsize=13, weight="bold", va="top")
        ax.text(x + cw / 2, y + 0.11, txt, ha="center", va="center", fontsize=6.8, color=INK)
    ax.text(gx - 0.1, -0.02, "What each difference measures (log-odds of P(right); each cell is one forward pass on "
            "the identical text)", fontsize=7, weight="bold", va="top")
    diffs = [("A − B  (dB)", "the bridge's effect with the peers visible"),
             ("C − D", "the bridge's effect with the peers blocked: compare with A − B"),
             ("A − C  (dC)", "the peers' effect with the bridge intact"),
             ("A − D  (dD)", "both together; with dB it sorts the readings")]
    for k, (d, t) in enumerate(diffs):
        yy = -0.1 - k * 0.075
        ax.text(gx - 0.1, yy, d, fontsize=6.9, weight="bold", va="top")
        ax.text(gx + 0.03, yy, t, fontsize=6.9, va="top", color=INK2)
    ax.text(0.52, -0.1, "A cell collapses when it falls ≥ 2 log-odds below A.", fontsize=6.9, color=INK2, va="top")
    ax.set_ylim(-0.42, 1)
    # readings, in order
    rx = 0.52
    ax.text(rx, 0.86, "Readings, taken in this order (agree prompts)", fontsize=7.6, weight="bold")
    reads = [("1", "dC ≥ 2", "C well below A: the model leaned on the peers from the start", "leaned"),
             ("2", "dB < 2, dD ≥ 2", "only D collapses: the peers were the backup; the own route is there",
              "only D"),
             ("3", "dB ≥ 2, dD ≥ 2", "both collapse: the bridge is needed either way", "both"),
             ("4", "dB < 2, dD < 2", "neither: a second route or a weak tool (the hidden check decides)",
              "neither"),
             ("5", "dB ≥ 2, dD < 2", "only B collapses", "only B")]
    lut = {v[0]: v for v in READ.values()}
    for i, (n, rule, text, key) in enumerate(reads):
        y = 0.76 - i * 0.135
        _, fc, tc, hatch = lut[key]
        box(ax, rx, y - 0.04, 0.075, 0.08, key, fc=fc, ec=fc if fc != FAINT else MUTED, size=6.6, color=tc,
            hatch=hatch, r=0.01)
        ax.text(rx + 0.085, y + 0.018, f"{n}.  {rule}", fontsize=7, weight="bold", va="center")
        ax.text(rx + 0.085, y - 0.02, text, fontsize=6.6, va="center", color=INK2)
    save(fig, "fig05_2x2_design")


# ---- Figure 6: dose-response (dose.csv) ----------------------------------------------------------------------------
def fig_dose():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharex=True)
    s = [0.25, 0.5, 0.75, 1.0]
    for r, key in enumerate(["4b", "27b"]):
        name, rdir, n = MODELS[key]
        d = pd.read_csv(rdir / "dose.csv")
        for c, fr in enumerate(["neutral", "defer"]):
            ax = axes[r, c]
            for cond in ["alone", "hidden", "agree"]:
                row = d[(d.frame == fr) & (d.condition == cond)].iloc[0]
                y = [row[str(v)] for v in s]
                st = SERIES[cond]
                lab = {"alone": "alone", "hidden": 'hidden twin, 3 peers ("?")', "agree": "3 agreeing peers"}[cond]
                ax.plot(s, y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.6, ms=5, label=lab,
                        markeredgecolor=SURF, markeredgewidth=0.8)
                ax.text(1.03, y[-1], lab.split(",")[0], va="center", fontsize=6.6, color=INK2)
            ax.axhline(2, color=MUTED, lw=0.7, ls=":")
            ax.set_ylim(-0.3, 13.5)
            ax.set_xlim(0.2, 1.22)
            ax.set_xticks(s)
            ax.set_title(f"{name}, {WORDING[fr]} wording, n = {n} questions")
            if c == 0:
                ax.set_ylabel("median fall in log-odds of P(right)")
            if r == 1:
                ax.set_xlabel("removal strength (share of the bridge's part removed)")
    axes[0, 0].text(0.62, 2.3, "collapse line (2)", fontsize=6.4, color=MUTED)
    fig.tight_layout(h_pad=1.6, w_pad=2.0)
    save(fig, "fig06_dose_response")


# ---- Figure 7: finding 1, probability beside log-odds, with ceiling counts ---------------------------------------
# Probability-rule "still needed" counts: the run reports print them only as text tables.
PROB_NEEDED = {  # (model, wording): [alone, agree 1, agree 2, agree 3]
    ("4b", "neutral"): [19, 9, 3, 2],     # run 1 report_conformity_v2_qwen35_4b_jpp.txt, "Alone: labels",
    ("4b", "deferential"): [19, 0, 0, 0],  # "Peers agree: label by number of peers"; _defer_ file for group-aware
    ("27b", "neutral"): [17, 16, 10, 9],  # run 2 report_conformity_v2_qwen36_27b_jpp.txt, same tables
    ("27b", "deferential"): [17, 13, 3, 4],  # run 2 report_conformity_v2_defer_qwen36_27b_jpp.txt
}


def fig_finding1():
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 6.4), sharex=True)
    cols = [("4b", "neutral"), ("4b", "deferential"), ("27b", "neutral"), ("27b", "deferential")]
    x = [0, 1, 2, 3]
    for c, (key, fr) in enumerate(cols):
        name, rdir, n = MODELS[key]
        h = pd.read_csv(rdir / "hidden_check.csv")
        h = h[h.wording == fr]
        get = lambda cond, k, col: float(h[(h.condition == cond) & (h.peers == k)][col].iloc[0])  # noqa: E731
        alone = {col: get("alone", 0, col) for col in ["need_prob_median", "need_lo_median", "needed", "ceiling"]}
        for r, col in enumerate(["need_prob_median", "need_lo_median"]):
            ax = axes[r, c]
            for cond in ["hidden", "agree"]:
                y = [alone[col]] + [get(cond, k, col) for k in (1, 2, 3)]
                st = SERIES[cond]
                ax.plot(x, y, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.5, ms=5,
                        markeredgecolor=SURF, markeredgewidth=0.8, label=st["label"])
            ax.plot([0], [alone[col]], marker="o", color=INK, ms=5.5, ls="none", label="alone")
            if r == 0:
                ax.set_ylim(-0.05, 1.08)
                ceil = [int(get("agree", k, "ceiling")) for k in (1, 2, 3)]
                ax.set_title(f"{name}\n{WORDING[fr]} wording, n = {n}", fontsize=7.8, pad=13)
                ax.text(0.0, 1.0, "agree at ceiling: " + " · ".join(map(str, ceil)), transform=ax.transAxes,
                        fontsize=7, color=INK2, va="bottom")
            else:
                ax.set_ylim(-0.6, 13)
                ax.axhline(2, color=MUTED, lw=0.7, ls=":")
        ax = axes[2, c]
        lo = [int(alone["needed"])] + [int(get("agree", k, "needed")) for k in (1, 2, 3)]
        sc = pd.read_csv(ROOT / "results/conformity_stats_v2/summary.csv")
        sc = sc[(sc.model == f"{name} J++ v2") & (sc.wording == fr)].set_index("peers")
        pr = [PROB_NEEDED[(key, fr)][0]] + [int(sc.loc[k, "old_still_needed"]) for k in (1, 2, 3)]
        assert pr == PROB_NEEDED[(key, fr)], (key, fr, pr)  # report text tables and summary.csv agree
        ax.plot(x, pr, color=INK2, marker="o", mfc=SURF, ls="--", lw=1.3, ms=5, label="probability rule")
        ax.plot(x, lo, color=BLUE_D, marker="o", ls="-", lw=1.6, ms=5, label="log-odds rule",
                markeredgecolor=SURF)
        ax.set_ylim(-1, 21)
        ax.axhline(n, color=RULE, lw=0.7)
        ax.text(3.15, n, f"n = {n}", fontsize=6.8, color=MUTED, va="center")
        ax.set_xticks(x, ["alone", "1", "2", "3"])
        ax.set_xlabel("agreeing peers")
    axes[0, 0].set_ylabel("median need,\nprobability")
    axes[1, 0].set_ylabel("median need,\nlog-odds")
    axes[2, 0].set_ylabel("questions labelled\n\"still needed\"")
    axes[1, 0].text(3.0, 2.3, "needed ≥ 2", fontsize=6.8, color=MUTED, ha="right")
    for ax in axes[0, 1:].tolist() + axes[1, 1:].tolist() + axes[2, 1:].tolist():
        ax.tick_params(labelleft=False)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h3, l3 = axes[2, 0].get_legend_handles_labels()
    order = [l1.index("alone"), l1.index(SERIES["hidden"]["label"]), l1.index(SERIES["agree"]["label"])]
    fig.legend([h1[i] for i in order] + h3, [l1[i] for i in order] + [f"still needed, {t}" for t in l3],
               loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.045), fontsize=7.2)
    fig.tight_layout(h_pad=1.2, w_pad=0.8)
    save(fig, "fig07_finding1_prob_logodds")


# ---- Figure 8: disagreeing peers, state and pull (report_conformity_v2_*.txt, typed from the text tables) --------
DISAGREE = {  # state at 3 peers: holds / partial / full (of n); then per state pooled over 1–3 peers:
    # prompts, "bridge still pulls", median pushback (bridge dLO − noise)
    ("4b", "neutral"): dict(k3=(8, 2, 8), other3=2, rows={"holds": (35, 17, 3.50), "partial give-in": (3, 1, 3.00),
                                                          "full give-in": (15, 3, 1.87)}),
    ("4b", "defer"): dict(k3=(0, 0, 20), other3=0, rows={"holds": (2, 1, 2.75), "partial give-in": (3, 1, 1.00),
                                                         "full give-in": (50, 8, 1.12)}),
    ("27b", "neutral"): dict(k3=(12, 1, 6), other3=0, rows={"holds": (44, 40, 8.54),
                                                            "partial give-in": (2, 1, 2.87),
                                                            "full give-in": (11, 1, 1.87)}),
    ("27b", "defer"): dict(k3=(5, 2, 12), other3=0, rows={"holds": (27, 21, 8.84), "partial give-in": (5, 0, 1.00),
                                                          "full give-in": (25, 13, 3.11)}),
}


def fig_disagree():
    fig, ax = blank((7.0, 2.55))
    cols = ["Model, wording", "State at 3 peers\nholds · partly · fully", "Holds\npulls / prompts",
            "Holds\npushback", "Gives in fully\npulls / prompts", "Gives in fully\npushback"]
    xs = [0.0, 0.25, 0.47, 0.61, 0.73, 0.88]
    top = 0.86
    for x, cname in zip(xs, cols):
        ax.text(x + 0.004, top + 0.02, cname, fontsize=6.8, weight="bold", va="bottom")
    ax.plot([0, 1], [top, top], color=INK2, lw=0.8)
    for i, ((key, fr), d) in enumerate(DISAGREE.items()):
        name, _, n = MODELS[key]
        y = top - 0.13 - i * 0.16
        ax.text(0.004, y, f"{name}, {WORDING[fr]}\n(n = {n} questions)", fontsize=7.2, va="center")
        hold, part, full = d["k3"]
        # segmented bar of states at 3 peers
        left = 0.25
        for v, fc, hatch in [(hold, BLUE_D, None), (part, BLUE_L, None), (full, ORANGE, "////")]:
            w = 0.19 * v / n
            if v:
                ax.add_patch(Rectangle((left, y - 0.045), w, 0.09, fc=fc, ec=SURF, lw=1.0, hatch=hatch))
                if w > 0.02:
                    ax.text(left + w / 2, y, str(v), ha="center", va="center", fontsize=6.5,
                            color="#ffffff" if fc == BLUE_D else INK)
            left += w
        if d["other3"]:
            ax.text(left + 0.003, y, f"+{d['other3']}", fontsize=6.0, color=INK2, va="center")
        for x, (st, j) in zip(xs[2:], [("holds", "pp"), ("holds", "m"), ("full give-in", "pp"),
                                       ("full give-in", "m")]):
            pr, pulls, med = d["rows"][st]
            ax.text(x + 0.004, y, f"{pulls} / {pr}" if j == "pp" else f"{med:.1f}", fontsize=7.4, va="center")
        ax.plot([0, 1], [y - 0.08, y - 0.08], color=FAINT, lw=0.6)
    ax.text(0.0, 0.13, "Pushback (median, log-odds): how far removing the bridge moves log P(peers') − log P(right) "
            "toward the peers, minus the everyday-word noise.\nPulls: labelled \"bridge still pulls\". "
            "Prompts pooled over 1–3 peers. "
            "4B neutral: 2 of 20 questions at 3 peers end in another letter (+2). Probability labels, J++ lens. "
            "Development wrong letters are same-kind letters, so D's destination is not read.",
            fontsize=6.3, color=INK2, va="top")
    save(fig, "fig08_disagree")


# ---- Figure 9: the 2x2 readings, counts per model and wording (twobytwo_counts.csv) ------------------------------
def fig_2x2_counts():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    rows = [("neutral", "agree", 1), ("neutral", "agree", 2), ("neutral", "agree", 3), ("neutral", "hidden", 3),
            ("defer", "agree", 1), ("defer", "agree", 2), ("defer", "agree", 3), ("defer", "hidden", 3)]
    ylabels = [f"{WORDING[f]}, {'agree' if c == 'agree' else 'hidden'} {k}" for f, c, k in rows]
    for ax, key in zip(axes, ["4b", "27b"]):
        name, rdir, n = MODELS[key]
        t = pd.read_csv(rdir / "twobytwo_counts.csv")
        for i, (fr, cond, k) in enumerate(rows):
            sub = t[(t.frame == fr) & (t.condition == cond) & (t.peers == k)]
            left = 0
            for rd in READ_ORDER:
                v = int(sub[sub.reading == rd].n.sum())
                lab, fc, tc, hatch = READ[rd]
                if v:
                    ax.barh(i + (0.4 if i >= 4 else 0), v, left=left, height=0.7, color=fc, hatch=hatch,
                            edgecolor=SURF, lw=1.0)
                    if v >= 2:
                        ax.text(left + v / 2, i + (0.4 if i >= 4 else 0), str(v), ha="center", va="center",
                                fontsize=6.8, color=tc)
                left += v
            assert left == n, (key, fr, cond, k, left)
        ax.set_yticks([i + (0.4 if i >= 4 else 0) for i in range(len(rows))], ylabels)
        if key == "4b":
            ax.invert_yaxis()  # shared y: invert once
        ax.set_xlim(0, n)
        ax.set_xticks([0, 5, 10, 15, n])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("questions")
        ax.set_title(f"{name}, n = {n} questions")
    handles = [Rectangle((0, 0), 1, 1, fc=READ[r][1], hatch=READ[r][3], ec=MUTED if READ[r][1] == FAINT else SURF)
               for r in READ_ORDER]
    labels = ["leaned on the peers (C well below A)", "only D collapses (peers the backup)",
              "both collapse (bridge needed either way)", "neither collapses", "only B collapses"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig09_2x2_counts")


# ---- Figure 10: per-question 2x2 readings on the 27B (twobytwo.csv, run 2 §5) -----------------------------------
def fig_2x2_per_question():
    t = pd.read_csv(R2 / "twobytwo.csv")
    t = t[t.calibration == False]  # noqa: E712
    cols = [("neutral", "agree", 1), ("neutral", "agree", 3), ("defer", "agree", 1), ("defer", "agree", 3),
            ("neutral", "disagree", 3), ("defer", "disagree", 3), ("neutral", "hidden", 3), ("defer", "hidden", 3)]
    qs = sorted(t.question.unique())
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.set_xlim(0, len(cols)), ax.set_ylim(len(qs), 0)
    for j, (fr, cond, k) in enumerate(cols):
        for i, q in enumerate(qs):
            rd = t[(t.frame == fr) & (t.condition == cond) & (t.peers == k) & (t.question == q)].reading.iloc[0]
            lab, fc, tc, hatch = READ[rd]
            ax.add_patch(Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, fc=fc, hatch=hatch, ec=SURF, lw=0.8))
            ax.text(j + 0.5, i + 0.5, lab, ha="center", va="center", fontsize=6.2, color=tc)
    ax.set_xticks([j + 0.5 for j in range(len(cols))],
                  [f"{WORDING[f]}\n{c} {k}" for f, c, k in cols], fontsize=6.8)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(qs))], [q.replace("V_", "") for q in qs], fontsize=6.8)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Qwen3.6-27B, J++ lens, 19 development questions", pad=34)
    ax.text(0, len(qs) + 0.9, "LM landmark → country, LP person → birth city, LU university → city. "
            "Disagree prompts use same-kind wrong letters, so D's destination is not read here.",
            fontsize=6.3, color=INK2, va="top")
    save(fig, "fig10_2x2_per_question_27b")


# ---- Figure 11: block validation (validation.csv, leak_test.json) -------------------------------------------------
def fig_block_validation():
    rows = [("copy-only: P(peers') back toward the no-peer level", "Copy-only: made-up subject, 3 wrong peers;\n"
             "P(peers') closes ≥ half the gap to no peers"),
            ("give-ins under disagree: back to the right letter", "Give-ins under disagree:\n"
             "the right letter beats the peers'"),
            ("hidden prompts: almost no change", "Hidden prompts: P(right) moves ≤ 1\nlog-odds; row median ≤ 0.5"),
            ("told to copy: P(peers') drops", "Told to copy:\nP(peers') falls ≥ 2 log-odds")]
    cols = [("4b", "neutral"), ("4b", "defer"), ("27b", "neutral"), ("27b", "defer")]
    fig, ax = blank((7.0, 3.0))
    x0, cw, top, rh = 0.36, 0.16, 0.80, 0.145
    for j, (key, fr) in enumerate(cols):
        ax.text(x0 + j * cw + cw / 2, top + 0.03, f"{MODELS[key][0]}\n{WORDING[fr]}", ha="center", va="bottom",
                fontsize=7, weight="bold")
    ax.plot([0, 1], [top + 0.015, top + 0.015], color=INK2, lw=0.8)
    # leak test row
    y = top - rh / 2
    ax.text(0.004, y, "Leak test: peers I against F under the\nblock give identical outputs", fontsize=6.9,
            va="center")
    for j, (key, fr) in enumerate(cols):
        if fr != "neutral":
            continue
        lt = json.loads((MODELS[key][1] / "leak_test.json").read_text())
        pairs = lt["pairs"]
        npass = sum(p["passed"] for p in pairs)
        mx = max(p["blocked_diff"] for p in pairs)
        ax.text(x0 + j * cw + cw, y, f"{npass}/{len(pairs)} pairs identical; largest blocked\ndifference {mx:.1f} (neutral prompts)",
                ha="center", va="center", fontsize=6.6)
    ax.plot([0, 1], [top - rh, top - rh], color=FAINT, lw=0.6)
    for i, (rowkey, lab) in enumerate(rows):
        y = top - (i + 1.5) * rh
        ax.text(0.004, y, lab, fontsize=6.9, va="center")
        for j, (key, fr) in enumerate(cols):
            v = pd.read_csv(MODELS[key][1] / "validation.csv")
            r = v[(v.frame == fr) & (v.row == rowkey) & (v.cell == "C")].iloc[0]
            share = r.passing / r.eligible
            bx = x0 + j * cw + 0.012
            bw = cw - 0.024
            ax.add_patch(Rectangle((bx, y - 0.025), bw, 0.022, fc=FAINT, ec="none"))
            ax.add_patch(Rectangle((bx, y - 0.025), bw * share, 0.022, fc=BLUE_D if r.passed else ORANGE, ec="none"))
            ax.plot([bx + 0.75 * bw] * 2, [y - 0.035, y + 0.007], color=INK, lw=0.8)
            ax.text(bx, y + 0.02, f"{int(r.passing)}/{int(r.eligible)}  {'pass' if r.passed else 'FAIL'}",
                    fontsize=6.8, va="bottom")
            if rowkey.startswith("hidden"):
                ax.text(bx, y - 0.04, f"row median {r['median']:.2f}", fontsize=6.2, va="top", color=INK2)
        ax.plot([0, 1], [y - rh / 2, y - rh / 2], color=FAINT, lw=0.6)
    ax.text(0.004, top - 5 * rh - 0.03, "Bars: share of eligible prompts that pass, under the block at every layer "
            "(cell C). Tick: the pass mark, three quarters (scope.md §11).", fontsize=6.3, color=INK2, va="top")
    save(fig, "fig11_block_validation")


# ---- Figures 12–13: run-3 placeholders ---------------------------------------------------------------------------
def fig_placeholder(name, title, body):
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True), s.set_linestyle("--"), s.set_color(MUTED)
    ax.grid(False)
    ax.set_title(title)
    ax.text(0.5, 0.5, body, ha="center", va="center", fontsize=7.5, color=INK2, transform=ax.transAxes)
    save(fig, name)

# ---- Figure 12: replication on the confirmation set (run 3 replication.csv) ---------------------------------------
R3 = ROOT / "results/v2_run3_qwen36_27b"


def fig_replication():
    rep = pd.read_csv(R3 / "replication.csv")
    rows = [  # group, label, wording, share
        ("2×2, three agreeing peers", "neutral: both collapse", "neutral", "2x2 agree 3: both collapse"),
        ("2×2, three agreeing peers", "group-aware: only D collapses", "deferential", "2x2 agree 3: only D collapses"),
        ("2×2, three agreeing peers", "group-aware: both collapse", "deferential", "2x2 agree 3: both collapse"),
        ("2×2, three agreeing peers", "group-aware: leaned on the peers", "deferential", "2x2 agree 3: C well below A"),
        ("Still needed, log-odds", "alone", "neutral", "LO alone: needed"),
        ("Still needed, log-odds", "neutral, agree 3", "neutral", "LO agree 3: needed"),
        ("Still needed, log-odds", "group-aware, agree 3", "deferential", "LO agree 3: needed"),
        ("Weakens against the hidden twin (3 peers)", "neutral", "neutral", "weakens at 3 peers (vs '?')"),
        ("Weakens against the hidden twin (3 peers)", "group-aware", "deferential", "weakens at 3 peers (vs '?')"),
        ("Hidden check: B collapses, \"?\" 1–3 (prompts)", "neutral", "neutral", "2x2 hidden 1-3: B collapses"),
        ("Hidden check: B collapses, \"?\" 1–3 (prompts)", "group-aware", "deferential",
         "2x2 hidden 1-3: B collapses"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    y, ticks, labels, last = 0, [], [], None
    for grp, lab, fr, share in rows:
        if grp != last:
            y += 0.55 if last else 0
            ax.text(-0.02, y - 0.05, grp, fontsize=7.4, weight="bold", ha="right", va="bottom",
                    transform=ax.get_yaxis_transform())
            y += 0.55
            last = grp
        r = rep[(rep.wording == fr) & (rep.share == share)].iloc[0]
        ax.add_patch(Rectangle((r.dev_lo, y - 0.22), max(r.dev_hi - r.dev_lo, 0.004), 0.44, fc=FAINT, ec=RULE, lw=0.6))
        ax.plot([r.dev, r.dev], [y - 0.22, y + 0.22], color=INK2, lw=1.4)
        ax.plot([r.conf_lo, r.conf_hi], [y, y], color=BLUE_D, lw=1.1)
        ax.plot([r.conf], [y], marker="o", ms=5.5, color=BLUE_D if r.replicates else ORANGE,
                markeredgecolor=SURF, markeredgewidth=0.8)
        ax.text(1.02, y, f"{int(r.dev_count)}/{int(r.dev_n)} → {int(r.conf_count)}/{int(r.conf_n)}",
                fontsize=6.8, va="center", transform=ax.get_yaxis_transform())
        ax.text(1.27, y, "yes" if r.replicates else "no", fontsize=6.8, va="center", weight="bold",
                transform=ax.get_yaxis_transform())
        ticks.append(y)
        labels.append(lab)
        y += 1
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set_ylim(y - 0.3, -0.4)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of questions (hidden check: of prompts)")
    ax.grid(axis="y", visible=False)
    ax.text(1.02, -0.3, "run 2 → run 3", fontsize=6.8, weight="bold", transform=ax.get_yaxis_transform())
    ax.text(1.27, -0.3, "inside?", fontsize=6.8, weight="bold", transform=ax.get_yaxis_transform())
    ax.set_title("Qwen3.6-27B: run 2 (19 development questions) against run 3 (20 new questions)", fontsize=8,
                 pad=14)
    handles = [Rectangle((0, 0), 1, 1, fc=FAINT, ec=RULE),
               plt.Line2D([], [], color=BLUE_D, marker="o", lw=1.1, markeredgecolor=SURF),
               plt.Line2D([], [], color=ORANGE, marker="o", lw=0, markeredgecolor=SURF)]
    fig.legend(handles, ["run 2 value and 95% interval (bar)", "run 3 value and 95% interval, inside run 2's interval",
                         "run 3 value outside run 2's interval"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=6.8)
    fig.subplots_adjust(left=0.33, right=0.76, top=0.92, bottom=0.14)
    save(fig, "fig12_confirmation_replication")


# ---- Figure 13: swap toward another bridge (run 3 swap_rates.csv) -------------------------------------------------
def fig_swap():
    sw = pd.read_csv(R3 / "swap_rates.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    conds = [("alone", "alone"), ("agree", "3 agreeing peers"), ("disagree", "3 disagreeing peers")]
    for ax, fr in zip(axes, ["neutral", "defer"]):
        d = sw[sw.frame == fr]
        for i, (cond, lab) in enumerate(conds):
            g = d[d.condition == cond]
            vf, vn = int(g.visible_follow.sum()), int(g.visible_n.sum())
            ev, en = int(g.visible_everyday_through.fillna(0).sum()), int(g.visible_everyday_n.sum())
            ax.plot([0], [i], marker="x", ms=6, color=MUTED, mew=1.2)
            ax.text(0.015, i + 0.26, f"everyday word {ev}/{en}", fontsize=6.0, color=MUTED, va="center")
            if cond == "alone":
                ax.plot([vf / vn], [i], marker="o", ms=7, color=INK, markeredgecolor=SURF)
                ax.text(vf / vn, i - 0.3, f"{vf}/{vn}", fontsize=6.8, ha="center", va="center")
                continue
            bf, bn = int(g.blocked_follow.sum()), int(g.blocked_n.sum())
            ax.plot([vf / vn, bf / bn], [i, i], color=RULE, lw=2.2, zorder=1)
            ax.plot([vf / vn], [i], marker="^", ms=7, color=ORANGE, markeredgecolor=SURF, zorder=3)
            ax.plot([bf / bn], [i], marker="s", ms=6.5, color=BLUE_D, markeredgecolor=SURF, zorder=3)
            ax.text(vf / vn, i - 0.3, f"{vf}/{vn}", fontsize=6.8, ha="center", va="center")
            ax.text(bf / bn, i + 0.3, f"{bf}/{bn}", fontsize=6.8, ha="center", va="center", color=BLUE_D)
        ax.set_yticks(range(3), [c[1] for c in conds])
        ax.set_ylim(2.6, -0.6)
        ax.set_xlim(-0.03, 1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("questions where the answer follows the swap")
        ax.set_title(f"Qwen3.6-27B, {WORDING[fr]} wording, n = 20"
                     + ("\n19 for disagree: one prompt has no onset" if fr == "defer" else "\n "), fontsize=7.8)
    handles = [plt.Line2D([], [], color=INK, marker="o", lw=0, ms=6),
               plt.Line2D([], [], color=ORANGE, marker="^", lw=0, ms=6),
               plt.Line2D([], [], color=BLUE_D, marker="s", lw=0, ms=6),
               plt.Line2D([], [], color=MUTED, marker="x", lw=0, ms=6, mew=1.2)]
    fig.legend(handles, ["alone", "peers visible", "peers blocked", "control: swap onto an everyday word (letters only)"],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig13_swap")


if __name__ == "__main__":
    fig_claims()
    fig_prompt_layers()
    fig_funnel()
    fig_block_patch()
    fig_2x2_design()
    fig_dose()
    fig_finding1()
    fig_disagree()
    fig_2x2_counts()
    fig_2x2_per_question()
    fig_block_validation()
    fig_replication()
    fig_swap()
