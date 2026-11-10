"""Table 1 of the rewritten paper, generated from the 2x2 tables so every cell comes from the result files.

    .venv/bin/python paper/make_table1.py     (rewrites the Table 1 block inside paper.md)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = Path(__file__).resolve().parent / "paper.md"
RUNS = {("27B", "dev"): "v2_run2_qwen36_27b", ("27B", "new"): "v2_run3_qwen36_27b", ("4B", "dev"): "v2_run1_qwen35_4b"}
ROWS = [("27B", "neutral", "dev"), ("27B", "neutral", "new"), ("27B", "group-aware", "dev"),
        ("27B", "group-aware", "new"), ("4B", "neutral", "dev"), ("4B", "group-aware", "dev")]


def sel(model, word, split, cond, k=3):
    d = pd.read_csv(ROOT / "results" / RUNS[(model, split)] / "twobytwo.csv")
    if "calibration" in d:
        d = d[d.calibration == False]  # noqa: E712
    return d[(d.frame == ("neutral" if word == "neutral" else "defer")) & (d.condition == cond) & (d.peers == k)]


def outcome(r):
    a, b, c, d = r.P_right_A >= .5, r.P_right_B >= .5, r.P_right_C >= .5, r.P_right_D >= .5
    if a and not d and c and not b:
        return "reason"
    if a and not d and b and c:
        return "either"
    if a and not d and b and not c:
        return "group"
    return "other"


def p(x):
    return "&lt;0.01" if x < 0.005 else f"{x:.2f}"


def table():
    rows = []
    for model, word, split in ROWS:
        ag, tw = sel(model, word, split, "agree"), sel(model, word, split, "hidden")
        oc = ag.apply(outcome, axis=1).value_counts()
        cells = [p(tw.P_right_B.median()), p(ag.P_right_A.median()), p(ag.P_right_B.median()),
                 p(ag.P_right_C.median()), p(ag.P_right_D.median()), int((ag.P_right_B >= .5).sum()),
                 oc.get("reason", 0), oc.get("either", 0), oc.get("group", 0), oc.get("other", 0)]
        rows.append(f"<tr><td>{model}, {word}, {split} ({len(ag)})</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return """<table class="data tight">
<caption><b>Table 1.</b> Three peers. Left: median probability of the correct answer in the four runs with agreeing peers (A unedited, B bridge removed, C peers blocked, D both), and with the bridge removed in the hidden twin. Right: number of questions whose answer survives run B, and number with each outcome. "Dev" is the development set and "new" the confirmation set; the number of questions is in brackets.</caption>
<thead>
<tr><th rowspan="2">Model, instruction</th><th colspan="5">Median probability of the correct answer</th><th colspan="5">Questions</th></tr>
<tr><th>twin B</th><th>A</th><th>B</th><th>C</th><th>D</th><th>survive B</th><th>need own reasoning</th><th>either is enough</th><th>need the group</th><th>other</th></tr>
</thead>
<tbody>
""" + "\n".join(rows) + "\n</tbody>\n</table>"


if __name__ == "__main__":
    t = PAPER.read_text(encoding="utf-8")
    a = t.index('<table class="data')
    assert "<b>Table 1.</b>" in t[a:a + 200]
    b = t.index("</table>", a) + len("</table>")
    PAPER.write_text(t[:a] + table() + t[b:], encoding="utf-8")
    print("Table 1 rewritten from the result files")
