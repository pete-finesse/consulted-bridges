# Paper red team / blue team: Codex writes, Claude draws, each reviews the other

*9 Nov 2026. Both agents read this file first and follow it exactly. Peter is the editor: he relays turn signals
and settles disagreements. The paper is filled in here:
https://docs.google.com/document/d/1jG87fY_DJxMCtxkZ3t7jAh3VaMXYR49tqZlzh5sM918/edit*

## Roles

| | Codex (writer) | Claude (figures and critic) |
|---|---|---|
| Owns | every word of the paper text, the title, section order | every figure and table image, figure captions' *content* (Codex may edit the prose) |
| Writes to | the Google Doc; `docs/paper/rounds/rN-codex.md` | `docs/paper/figures/` (SVG + PNG, one file per figure, plus `figures/README.md` listing each figure's source table); `docs/paper/rounds/rN-claude.md` |
| Never edits | files under `docs/paper/figures/` | the paper's prose in the Doc (comments and suggestions only), `rN-codex.md` |
| Reviews | Claude's figures, every round | Codex's text, every round |

Both: commit by name only your own files (`git add <path>`), never `git add -A`, never push, never edit anything
outside `docs/paper/`. The rest of the repository is read-only reference. Nothing that spends money.

## Sources (the only places numbers and claims may come from)

- `docs/paper/outline.md` (sections, status tags SETTLED / PENDING RUN 3 / PETER) and `docs/paper/resources.md`.
- Claims and their wording: `docs/conformity/scope.md` §1 (what we claim and do not), §3–§5, §7, §9.
- Method as run: `docs/conformity/methods.md`. Development story and pass rates: `docs/conformity/process.md`.
- Results: `results/v2_run1_qwen35_4b/summary.md`, `results/v2_run2_qwen36_27b/summary.md`,
  `results/conformity_stats_v2/summary.csv`; later `results/v2_run3_qwen36_27b/summary.md` (confirmation set, swap).
- Existing figures to reuse or redraw: `docs/conformity/walkthrough.html`, `docs/conformity/method-map.html`.
- Earlier draft: `docs/paper/Testing_Silent_Dissent_Draft.docx` (predates the 2×2, the 27B run and the log-odds
  re-analysis; use its framing, not its numbers).

## Turns

Round 1 (Codex): title; the full text of every SETTLED section into the Doc; a one-line placeholder for each
PENDING RUN 3 item; the PETER items written as a best guess with "[PETER: …]" marking the choice. Then
`rounds/r1-codex.md`: what was written, every number with its source line, open questions for Peter.

Round 1 (Claude, in parallel with Codex's round 1): every figure the outline marks "to make", plus redraws of the
"exists" ones in paper style; `figures/README.md`; then, once Peter says Codex's round is in, the review
`rounds/r1-claude.md` of the Doc's text.

Round 2 (Codex): `rounds/r2-codex.md` answers every numbered finding (accept with the fix, or reject with the reason),
revises the Doc, and reviews every figure (`rounds/r2-codex-figures.md`, same numbered format).
Round 2 (Claude): fixes the figures, answers the figure review, re-reviews the text.
Repeat until both reviews return no findings above "minor", or Peter stops it. Expect 3 rounds.

Turn signals come from Peter ("Codex round N is in", "Claude round N is in"). Do not start a review before the
signal; the other agent may still be mid-edit.

## The review checklist (both directions)

Every finding: a number, a severity (**blocking** = wrong or unsupported; **major** = misleading or missing;
**minor** = style), the exact passage or figure, the source line that decides it, and the fix.

Text:
1. Every number traces to a source line; denominators are explicit; "19 of 19" not "all".
2. No claim beyond `scope.md` §1. The wording "the answer depends on the model's own intermediate step", never
   "thinks for itself". Findings are stated per model and per wording: on the 27B neutral wording agreeing peers
   *weaken* the bridge, on the 4B and under the group-aware wording they *stop* it (with the 27B's 2×2 showing the
   model still computing its own route next to the peers).
3. The population sentence is present: the findings are about questions where the model demonstrably uses an
   intermediate step alone, which is also Silent Dissent's population.
4. Limitations present: format scope (arbitrary lookups only); selection bias and the hidden baseline; the 4B's
   same-kind collateral; reach's weakness; the counts set resting on two facts; nothing about free-form or
   multi-turn answers; what the 2×2 does not show (where copying happens; natural two-hop questions).
5. Probability and log-odds shown together wherever a "not needed" share appears, with the ceiling count.
6. Style: short sentences, every term defined where it first appears (bridge, J-lens, residual, removal, need,
   reach, log-odds, the block), no hype, no em-dashes, numbers only where they change what the reader does.
7. PENDING RUN 3 placeholders are visibly placeholders; PETER choices are marked.

Figures:
1. Every plotted number traces to a named table in a source file (name it in `figures/README.md`).
2. Axis labels with units; the model and wording in the title or panel label; n stated.
3. Colours legible in greyscale; one message per figure; no chart junk.
4. The 2×2 figure follows `walkthrough.html` §2 (cells A–D, what each difference measures).
5. Captions say what the reader should see, in one sentence, then the source.

## Resolving disagreements

The source line wins. If the two agents disagree on a judgment call (framing, emphasis, how much to claim), both
write their position in two sentences in the round file and Peter decides. Nobody edits the other's files to settle it.
