"""Build the paper: paper.md -> paper.html -> causal-test-of-silent-dissent.pdf (headless Chrome).

The page loads the Old Standard TT font from Google Fonts, so the build needs a network connection.

Run from the repository root:

    .venv/bin/python paper/build.py
"""
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: letter; margin: 1in;
        @bottom-center { content: counter(page); font-family: 'Old Standard TT', 'Times New Roman', serif; font-size: 10pt; } }
@page :first { @bottom-center { content: none; }
        @bottom-left { content: "\\00B9  Extended from work begun at the Swarm Alignment Hackathon, November 2026";
                       font-family: 'Old Standard TT', 'Times New Roman', serif; font-size: 9.5pt; vertical-align: top;
                       padding-top: 5pt; white-space: nowrap;
                       background: linear-gradient(#000, #000) no-repeat left top / 2in 0.6pt; } }
html { font-family: 'Old Standard TT', 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.3; color: #000; }
body { margin: 0; overflow-wrap: break-word; }
div.titleblock { border-top: 3pt solid #000; border-bottom: 0.75pt solid #000; padding: 24pt 0 30pt; margin: 18pt 0 26pt; }
h1 { font-size: 20pt; line-height: 1.2; text-align: center; margin: 0; text-transform: uppercase; letter-spacing: 0.05em; }
h1 sup { font-size: 11pt; letter-spacing: 0; }
div.authors { display: flex; justify-content: center; gap: 0.9in; font-size: 11pt; margin: 0 0 10pt; }
p.with { text-align: center; font-size: 11pt; margin: 0 0 20pt; line-height: 1.4; }
h2 { font-size: 14pt; color: #434343; margin: 13pt 0 4pt; break-after: avoid; }
h2.abstract-h { text-align: center; color: #000; font-size: 14pt; margin: 0 0 12pt; }
h3 { font-size: 12pt; color: #666666; margin: 9pt 0 3pt; break-after: avoid; }
p { margin: 0 0 5pt; text-align: justify; hyphens: auto; orphans: 2; widows: 2; }
ul, ol { margin: 0 0 7pt; padding-left: 22pt; }
li { margin: 0 0 3pt; text-align: left; hyphens: auto; }
div.abstract { margin: 0 0.8in 8pt; }
div.abstract p { text-align: justify; }
pre { background: #f6f7f9; border: 0.6pt solid #c9ced6; border-radius: 4pt; padding: 6pt 9pt; margin: 4pt 0 8pt;
      white-space: pre-wrap; break-inside: avoid; }
pre code { font-family: Menlo, Consolas, monospace; font-size: 8.3pt; line-height: 1.4; }
code { font-family: Menlo, Consolas, monospace; font-size: 0.82em; overflow-wrap: anywhere; hyphens: none; }
figure { margin: 8pt 0 9pt; break-inside: avoid; text-align: center; }
figure img { width: 100%; }
figure.narrow img { width: 66%; }
figure.mid img { width: 80%; }
figcaption, caption { font-size: 0.9em; line-height: 1.28; text-align: left; margin-top: 4pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 9pt; font-size: 0.9em; line-height: 1.25;
        break-inside: avoid; }
caption { caption-side: top; margin: 0 0 4pt; font-size: 1em; }
th, td { padding: 2.6pt 5pt; vertical-align: top; text-align: left; }
thead th { border-bottom: 0.8pt solid #222; font-weight: bold; vertical-align: bottom; }
thead tr:first-child th { border-top: 0.8pt solid #222; }
tbody tr:last-child td { border-bottom: 0.8pt solid #222; }
tbody td { border-bottom: 0.3pt solid #d5d9df; }
table.data th, table.data td { text-align: center; }
table.data th:first-child, table.data td:first-child { text-align: left; white-space: nowrap; }
table.data thead tr:first-child th[colspan] { border-bottom: 0.4pt solid #888; }
table.tight { font-size: 0.86em; }
table.tight th, table.tight td { padding: 2.4pt 3pt; }
ol.refs { padding-left: 22pt; }
ol.refs li { font-size: 0.92em; text-align: left; margin-bottom: 4pt; overflow-wrap: anywhere; hyphens: none; }
div.pagebreak { break-after: page; }
strong, b { font-weight: bold; }
"""


FIGS = ["fig09_2x2_counts", "fig13_swap", "fig11_block_validation", "fig03_gate_funnel",
        "fig07_finding1_prob_logodds", "fig06_dose_response", "fig10_2x2_per_question_27b", "fig08_disagree",
        "fig12_confirmation_replication", "fig02_prompt_layers", "fig04_block_patch"]
CLEAN = {" (scope.md §11).": "."}


def copy_figures():
    out = HERE / "figs"
    out.mkdir(exist_ok=True)
    for name in FIGS:
        svg = (HERE / "figures" / f"{name}.svg").read_text(encoding="utf-8")
        for old, new in CLEAN.items():
            svg = svg.replace(old, new)
        assert "scope.md" not in svg and ".html" not in svg, name
        (out / f"{name}.svg").write_text(svg, encoding="utf-8")


def main():
    copy_figures()
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render((HERE / "paper.md").read_text(encoding="utf-8"))
    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>A Causal Test of Silent Dissent under Conformity</title>'
            f'<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Old+Standard+TT:ital,wght@0,400;0,700;1,400&display=block">'
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out_html = HERE / "paper.html"
    out_html.write_text(html, encoding="utf-8")
    out_pdf = HERE / "causal-test-of-silent-dissent.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True)
    print("wrote", out_pdf)


if __name__ == "__main__":
    main()
