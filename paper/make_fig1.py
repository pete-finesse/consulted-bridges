"""Figure 1 of the rewritten paper: one group prompt and the four runs on it.

Reads the Eiffel Tower rows (three agreeing peers) from the run 1 and run 2 2x2 tables, so every number in the
figure comes from the result files. Run from the repository root:

    .venv/bin/python paper/make_fig1.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "fig1_example.svg"

BLUE, ORANGE, INK, GREY, RULE = "#1f5fae", "#c2410c", "#1b1f24", "#5b6470", "#c9ced6"
SANS = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"

CASES = [  # (run folder, frame, title line 1, title line 2, reading, gloss lines)
    ("v2_run2_qwen36_27b", "neutral", "Qwen3.6-27B", "neutral instruction", ["needs its own", "reasoning"],
     ["Without the bridge", "the answer is lost,", "peers visible or", "blocked."]),
    ("v2_run1_qwen35_4b", "neutral", "Qwen3.5-4B", "neutral instruction", ["either is", "enough"],
     ["The answer survives", "either loss and is", "lost only under", "both."]),
    ("v2_run1_qwen35_4b", "defer", "Qwen3.5-4B", "group-aware instruction", ["needs the", "group"],
     ["Blocking the peers", "alone loses", "the answer."]),
]
EXPECT = ["both collapse", "only D collapses", "C well below A"]


def row(run, frame):
    d = pd.read_csv(ROOT / "results" / run / "twobytwo.csv")
    s = d[(d.question == "V_LM_eiffel") & (d.frame == frame) & (d.condition == "agree") & (d.peers == 3)]
    assert len(s) == 1, (run, frame, len(s))
    return s.iloc[0]


def fmt(p):
    return "<0.01" if p < 0.005 else f"{p:.2f}"


def text(x, y, s, size=12, fill=INK, weight="normal", anchor="start", family=SANS):
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{s}</text>')


def main():
    o = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 356" width="960" height="356">',
         '<rect width="960" height="356" fill="white"/>']

    # ---- panel a: the two routes
    o.append(text(0, 16, "a", 14, weight="bold"))
    o.append(text(16, 16, "Two ways to the same reply", 13, weight="bold"))
    o.append('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
             f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{INK}"/></marker></defs>')

    def box(x, y, w, h, fill, stroke):
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>'

    def arrow(d):
        return f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="1.3" marker-end="url(#ah)"/>'

    # own route
    o.append(text(0, 52, "Its own reasoning", 12.5, BLUE, "bold"))
    o.append(box(0.5, 62, 112, 38, "#f6f7f9", RULE))
    o.append(text(56.5, 86, "Eiffel Tower", 13, INK, anchor="middle"))
    o.append(text(56.5, 116, "in the question", 11, GREY, anchor="middle"))
    o.append(arrow("M113 81 H146"))
    o.append(box(148, 62, 92, 38, "#dbe7f6", BLUE))
    o.append(text(194, 86, "France", 13.5, BLUE, "bold", "middle"))
    o.append(text(194, 116, "the bridge: never written,", 11, GREY, anchor="middle"))
    o.append(text(194, 130, "a pattern in the model's internal state", 11, GREY, anchor="middle"))
    o.append(text(194, 150, "pattern removed in runs B and D", 11, BLUE, "bold", "middle"))
    o.append(arrow("M240 81 H336 V140"))
    # peer route
    o.append(text(0, 212, "Going along with the group", 12.5, ORANGE, "bold"))
    o.append(box(0.5, 222, 240, 38, "#fdf0e9", ORANGE))
    o.append(f'<text x="120.5" y="246" font-family="{SANS}" font-size="13" fill="{INK}" text-anchor="middle" xml:space="preserve">'
             f'Agent 2: <tspan fill="{ORANGE}" font-weight="bold">F</tspan>'
             f'   Agent 3: <tspan fill="{ORANGE}" font-weight="bold">F</tspan>'
             f'   Agent 4: <tspan fill="{ORANGE}" font-weight="bold">F</tspan></text>')
    o.append(text(120.5, 276, "the peers' answers, in the prompt", 11, GREY, anchor="middle"))
    o.append(text(120.5, 296, "blocked in runs C and D", 11, ORANGE, "bold", "middle"))
    o.append(arrow("M241 241 H336 V192"))
    # the reply
    o.append(box(302, 142, 68, 48, "white", INK))
    o.append(text(336, 174, "F", 22, INK, "bold", "middle"))
    o.append(text(378, 162, "the", 11, GREY))
    o.append(text(378, 176, "reply", 11, GREY))

    # ---- panel b: the four runs, three cases
    bx = 446
    o.append(text(bx, 16, "b", 14, weight="bold"))
    o.append(text(bx + 16, 16, "Four runs of one prompt: probability of the correct letter", 13, weight="bold"))
    gx0, gw, gap, cw, ch = bx + 98, 124, 18, 62, 50
    top = 104
    o.append(text(gx0 - 8, top + 30, "bridge intact", 11.5, INK, anchor="end"))
    o.append(text(gx0 - 8, top + ch + 4 + 30, "bridge removed", 11.5, INK, anchor="end"))
    for k, (run, frame, t1, t2, reading, gloss) in enumerate(CASES):
        r = row(run, frame)
        assert r.reading.startswith(EXPECT[k]), r.reading
        gx = gx0 + k * (gw + gap)
        cx = gx + gw / 2
        o.append(text(cx, 44, t1, 12.5, INK, "bold", "middle"))
        o.append(text(cx, 60, t2, 11.5, GREY, anchor="middle"))
        for j, lab in enumerate(["visible", "blocked"]):
            o.append(text(gx + cw * j + cw / 2, 82, "peers", 11, GREY, anchor="middle"))
            o.append(text(gx + cw * j + cw / 2, 95, lab, 11, GREY, anchor="middle"))
        cells = {"A": (0, 0, r.P_right_A, 0.0), "C": (1, 0, r.P_right_C, r.dC),
                 "B": (0, 1, r.P_right_B, r.dB), "D": (1, 1, r.P_right_D, r.dD)}
        for name, (col, rw, p, fall) in cells.items():
            x, yy = gx + col * cw, top + rw * (ch + 4)
            down = p < 0.5
            if down:
                o.append(f'<rect x="{x + 2}" y="{yy + 1}" width="{cw - 4}" height="{ch - 2}" rx="4" fill="white" '
                         f'stroke="{ORANGE}" stroke-width="1.5" stroke-dasharray="4 3"/>')
            else:
                o.append(f'<rect x="{x + 2}" y="{yy + 1}" width="{cw - 4}" height="{ch - 2}" rx="4" fill="{BLUE}"/>')
            fg = INK if down else "white"
            o.append(text(x + 8, yy + 15, name, 10.5, fg, "bold"))
            o.append(text(x + cw / 2, yy + 35, fmt(p), 15, fg, "bold", "middle"))
        for i, ln in enumerate(reading):
            o.append(text(cx, 232 + 15 * i, ln, 12.5, INK, "bold", "middle"))
        for i, ln in enumerate(gloss):
            o.append(text(cx, 270 + 14 * i, ln, 11, GREY, anchor="middle"))

    # legend
    ly = 344
    o.append(f'<rect x="{bx}" y="{ly - 11}" width="22" height="14" rx="3" fill="{BLUE}"/>')
    o.append(text(bx + 28, ly, "answer survives (probability at least 0.5)", 11, GREY))
    o.append(f'<rect x="{bx + 252}" y="{ly - 11}" width="22" height="14" rx="3" fill="white" stroke="{ORANGE}" '
             f'stroke-width="1.5" stroke-dasharray="4 3"/>')
    o.append(text(bx + 280, ly, "answer lost", 11, GREY))
    o.append("</svg>")
    OUT.write_text("\n".join(o), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
