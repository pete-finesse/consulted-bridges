# Paper

`causal-test-of-silent-dissent.pdf` is the current version. It was revised after the Swarm Alignment Hackathon;
the version submitted on 9 November 2026 is in `submitted/`. Appendix D of the paper, and the repository README,
say what changed and why.

## Rebuild

From the repository root:

```bash
.venv/bin/python paper/build.py
```

The build needs `markdown-it-py`, Google Chrome, and a network connection for the Old Standard TT font.

## Files

| File | What it is |
|---|---|
| `paper.md` | The text, with Table 1 written in by `make_table1.py` |
| `build.py` | Page style and the Markdown to HTML to PDF build |
| `make_table1.py` | Table 1, from the 2x2 tables of runs 1 to 3 |
| `make_fig1.py`, `make_fig2.py` | Figures 1 and 2, from the same tables |
| `figures/make_figures.py` | The appendix figures, from the run folders under `results/` |
| `submitted/` | The paper as submitted to the hackathon |

Every number in the tables and figures is read from `results/v2_run1_qwen35_4b`, `results/v2_run2_qwen36_27b` and
`results/v2_run3_qwen36_27b`.
