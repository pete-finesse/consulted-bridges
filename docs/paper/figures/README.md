# Paper figures

*9 Nov 2026, Claude (figures and critic, `docs/paper/redteam-protocol.md`). Every figure is drawn by
`make_figures.py` from the files named below; rerun with `.venv/bin/python docs/paper/figures/make_figures.py` from
the repository root. One SVG (text kept as text) and one PNG (220 dpi) per figure. Codex owns the caption prose in
the paper; the captions here give the content: what the reader should see, then the source.*

Colour is never the only channel: line style and marker separate alone / hidden twin / agree; the "leaned on the
peers" reading is hatched; heatmap and bar cells carry their label or count as text.

**Priority for the 8-page body** (from the session that set the deadline): Fig. 5, Fig. 7, Fig. 3, Fig. 11. The
rest can go to an appendix.

| File | Outline § | Caption content | Plotted numbers come from |
|---|---|---|---|
| `fig01_claims_tests` | 1.3 | Each claim rests on a named test; most are run on both models, the swap and the replication wait on run 3. | `docs/conformity/method-map.html` §5 "Each claim, the tests under it" (states), `docs/conformity/plan.md` §6. No numbers. |
| `fig02_prompt_layers` | 3.2 | One agree-3 prompt, tokens along and layers up: removal in one early 6-layer window from the question's first word to the end, before the answer onset (27B: layer 54 of 64); the peer block covers the peers' answer tokens at every layer; P(right) is read at the last token. | Schematic. Layer 54 of 64: `results/v2_run2_qwen36_27b/summary.md` Setup. Layout after `method-map.html` §2. |
| `fig03_gate_funnel` | 3.3, 4 | The gate keeps most letter questions and few number lookups: 75 of 171 candidates pass with reach on the 4B, 64 of 167 on the 27B. | `results/v2_run1_qwen35_4b/funnel.csv` and `results/v2_run2_qwen36_27b/funnel.csv`, rows `config` = `*_jpp`, both splits, `family` ≠ `all`; columns `drafted`, `knows it`, `selected`, `sel + reach`. Families pooled: letters = C_LM, C_LU, C_LP; number lookups = C_MN, C_SP, C_EL, C_PL, C_IN; other categories = C_CT, C_DS, C_VC, C_EO. Totals match summaries §1 (J++, J++ + reach). |
| `fig04_block_patch` | 3.7 | The peer block cuts every look back at a peer's answer on full-attention layers and stops the running memory taking anything from it on linear-attention layers; the patch instead writes the hidden twin's state into the same positions. | Schematic after `docs/conformity/walkthrough.html` §3–§4. Leak-test line: `leak_test.json` in both run folders (7 of 7 pairs, blocked difference 0.0). |
| `fig05_2x2_design` | 3.8 | Four runs on the identical prompt (A no edit, B bridge removed, C peers blocked, D both) and the five readings, taken in order, with the 2 log-odds collapse line. | `docs/conformity/scope.md` §11 "The 2x2" (rules and order); cell layout per `walkthrough.html` §2. No data. |
| `fig06_dose_response` | 5.2 | Removal works in the group layout: the fall rises with removal strength and saturates by 0.5–0.75; agreeing peers sit below the hidden twin at every strength, on both models and wordings. | `results/v2_run1_qwen35_4b/dose.csv`, `results/v2_run2_qwen36_27b/dose.csv` (rows `frame` × `condition` alone / hidden / agree; columns 0.25–1.0). Hidden and agree are 3 peers (summaries §3 "Dose-response"). |
| `fig07_finding1_prob_logodds` | 5.4 | In probability the bridge's need falls toward zero with agreeing peers; in log-odds it weakens but stays above the "needed" line on the 27B neutral wording (14 of 19 still needed at 3 peers) and stops on the 4B and on the 27B group-aware wording; the ceiling counts show where probability cannot move. | Rows 1–2 and the log-odds counts: `hidden_check.csv` in both run folders (`need_prob_median`, `need_lo_median`, `needed`, `ceiling`; `condition` alone / agree / hidden; `wording` neutral / deferential). Probability-rule counts (row 3, dashed): the run reports' text tables "Alone: labels" and "Peers agree: label by number of peers" in `report_conformity_v2_qwen35_4b_jpp.txt`, `..._defer_qwen35_4b_jpp.txt`, `report_conformity_v2_qwen36_27b_jpp.txt`, `..._defer_qwen36_27b_jpp.txt` (typed into `PROB_NEEDED` in the script). Caveat for the caption: the 4B group-aware hidden twin is a weak baseline (P(right) ≈ 0.4; run 1 summary §3). |
| `fig08_disagree` | 5.5 | When the model holds against wrong peers, the bridge still pulls toward the right letter (27B: 40 of 44 neutral prompts, median pushback 8.5 log-odds); when it gives in fully, the pull is weaker. | Text tables "Peers disagree: state by number of peers", "label by state", "median pushback … by state" in the four `report_conformity_v2_*_jpp.txt` files above (typed into `DISAGREE`). Probability labels, J++. Caption must state the wrong-letter confound (run 2 summary §5). |
| `fig09_2x2_counts` | 5.6 | The 2×2 readings per model and wording: at 3 agreeing peers the 27B neutral is "both collapse" in 14 of 19 (own route kept next to the peers), the 4B neutral "only D" in 15 of 20 (peers the backup), the 4B group-aware "leaned" in 18 of 20; hidden-twin prompts collapse under removal. | `twobytwo_counts.csv` in both run folders (`frame` × `condition` × `peers` × `reading`, column `n`); the script checks each row sums to n. Matches summaries §5. |
| `fig10_2x2_per_question_27b` | 5.6 | Per question on the 27B: the person questions and Fuji are the ones that lean on the peers; the group-aware wording splits the rest between "only D" and "both". | `results/v2_run2_qwen36_27b/twobytwo.csv` column `reading`, `calibration` = False, conditions agree 1 / 3, disagree 3, hidden 3 per wording. Matches summary §5 "Per question". |
| `fig11_block_validation` | 5.7 | The block passes the leak test exactly and all four known-case rows on both models and wordings. | `validation.csv` in both run folders (`cell` = C; `passing`, `eligible`, `passed`); leak test from `leak_test.json` (`pairs[].passed`, `blocked_diff`). Pass mark ¾: `scope.md` §11. |
| `fig12_confirmation_placeholder` | 6 | PLACEHOLDER (titled empty panel): each development reading on the 20 frozen questions against run 2's 95% interval. | Waits on `results/v2_run3_qwen36_27b/summary.md`, `replication.csv`. |
| `fig13_swap_placeholder` | 3.10, 6 | PLACEHOLDER: does the letter follow a swap of the bridge, peers visible and blocked. | Waits on `results/v2_run3_qwen36_27b/summary.md`, `swap_rates.csv`. |

## Notes for the text

- All numbers are the development set (runs 1–2), J++ lens: 20 questions on the 4B, 19 on the 27B (calibration
  questions excluded).
- "Group-aware" is the wording the files call "defer" / "deferential".
- Not drawn, though the outline lists them as SETTLED tables: the hidden-check table (run summaries §3), the patch
  correlations (§6), the variants (§7). They read fine as text or tables.
