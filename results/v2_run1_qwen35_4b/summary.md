# GPU run 1: Qwen3.5-4B, candidate screen and the group-layout tools

*9 Nov 2026 UTC. Plan: `docs/conformity/plan.md` Phase C (steps 4–7). Pass marks fixed before the run in
`docs/conformity/scope.md` §11 (commit c290a5f). Runner `scripts/pod/v2_run.sh`; tables from
`consulted_bridges.conformity_funnel`, `conformity_stats`, `conformity_v2_report`.*

## Setup

- **Question.** A two-hop question (Colosseum → Italy → first letter I) has a hidden middle step, the **bridge**
  (Italy). **Removal** takes the bridge's J++ lens direction out of the residual stream, from the question's first
  word to the end of the prompt, in one 6-layer window before the answer forms. If P(right) collapses, the model
  uses the bridge.
- **Group prompts.** Peers answer first (Agent 2, 3, 4), then the model. The peers either agree (right letter),
  disagree (one wrong letter), or are hidden: they write "?" or "(answer submitted)". **Neutral** wording: "Give
  your own answer". **Group-aware** wording: "Taking the other agents' answers into account".
- **Peer block (C).** At every layer, no later position can read the peers' answer tokens: an attention mask on the
  full-attention layers, and zeroed input, update and decay at those tokens on the linear-attention layers.
- **The 2×2.** On each prompt: A no edit, B bridge removed, C peers blocked, D both. Scored in log-odds of P(right).
  A cell **collapses** when it falls ≥ 2 log-odds below A.
- **Questions.** 24 development letter questions (`conformity.QUESTIONS["qwen35_4b"]`). The 4 calibration questions
  are left out of every count except the validation rows, leaving 20. V_LP_beyonce is in the list but is not
  J++-selected. Screen: the 171 v2 candidates under both lenses.

## Ran

| Step | What | Outcome |
|---|---|---|
| a | Leak test, block at every layer, peers I vs F (and "?") on 4 questions | **Pass, exact**: blocked logit difference 0.0 on all 7 pairs (unblocked up to 7.75). Default kernels; `USE_HUB_KERNELS=0` was not needed |
| b | Trimmed solo scorecard on 171 candidates × 2 lenses | 342 cards, none failed |
| c+d | Conformity, J++, every B1 variant plus own-part removal, both wordings | 1,064 prompts. 9 lost to CUDA out-of-memory (3 question-wordings) and re-run whole, then folded in (`scripts/v2_merge_redo.py`) |
| e+f | Block, 2×2, dose-response, patch; both wordings | 592 prompt rows; patch on all 24 agree-3 prompts per wording |

Pods (CA-MTL-3, shared volume): one A100 for the leak test and screen, then 2 H200 and 1 more A100 for the group
steps, 12 question-shards chained. GPU work 07:58–08:40 UTC. Pods kept up afterwards on Peter's word.

## 1. Funnel (gate screen, both halves; the confirmation half as counts only)

Selected = need ≥ 0.5 and the bridge comes first (the existing rule). "+reach" = also reach ≥ 0.5.

| Family | Drafted | Knows it | Selected J-lens | Selected J++ | J++ + reach |
|---|---|---|---|---|---|
| Letter, landmark → country (C_LM) | 40 | 38 | 36 | 32 | 31 |
| Letter, university → city (C_LU) | 28 | 26 | 23 | 21 | 20 |
| Letter, person → birth city (C_LP) | 30 | 15 | 10 | 8 | 8 |
| Month → number (C_MN) | 14 | 13 | 7 | 9 | 7 |
| Players per side (C_SP) | 5 | 5 | 3 | 3 | 3 |
| Continent (C_CT) | 12 | 12 | 4 | 5 | 4 |
| Element → atomic number (C_EL) | 6 | 3 | 1 | 1 | 0 |
| Planet → position (C_PL) | 5 | 2 | 1 | 1 | 1 |
| Instrument → strings (C_IN) | 2 | 2 | 1 | 1 | 1 |
| Driving side (C_DS) | 11 | 8 | 0 | 0 | 0 |
| Starts with a vowel, yes/no (C_VC) | 14 | 7 | 0 | 0 | 0 |
| Even/odd letter count (C_EO, built to fail) | 4 | 2 | 0 | 0 | 0 |
| **All** | **171** | **133** | **86** | **81** | **75** |

By half: development 57 / 54 of 114 selected (J-lens / J++), confirmation 29 / 27 of 57. Of the development
questions, 48 are selected by both lenses, 9 by the J-lens only and 6 by J++ only. Single-token and onset checks
remove almost nothing.

**Reading.** Letters survive in two relations (landmark, university) at ≥ 75%. Person → city loses half at
knows-it. Of the number formats only months reach a usable size (7–9). Driving side and the vowel class fail
outright on the 4B, so by the stop rule (scope.md §7, below one in five) they are out for the removal arm on this
model. So are element and planet (1 each). The lenses pass about the same number; J++ passes fewer landmark
questions (32 against 36) and more month questions (9 against 7).

## 2. Old labels against log-odds, and the replication

The base conditions reproduce the earlier J++ run (round 14) on the same questions. Agree prompts, 20 questions:

| Wording | Peers | Old label still needed | LO still needed | LO not needed | LO inconclusive | at ceiling | median need_LO |
|---|---|---|---|---|---|---|---|
| neutral | 1 | 9 | 4 | 0 | 13 | 0 | 1.72 |
| neutral | 2 | 3 | 2 | 4 | 8 | 0 | 0.99 |
| neutral | 3 | 2 | 0 | 7 | 7 | 0 | 0.83 |
| group-aware | 1 | 0 | 0 | 6 | 11 | 0 | 0.58 |
| group-aware | 2 | 0 | 0 | 18 | 1 | 7 | 0.12 |
| group-aware | 3 | 0 | 0 | 18 | 1 | 15 | −0.06 |

Alone: 11 of 20 needed in log-odds (9 inconclusive, from the 4B's same-kind control), median need_LO 7.2.
Previous J++ run, agree 3 neutral: 0 of 19 needed, 7 not needed. Group-aware: 0 needed, 17 not needed.

## 3. Does removal still work in the group layout? (step 5)

**By the pre-registered rule: no, in both wordings.** In hidden prompts the bridge is "needed" in 20–30% of
questions (neutral) and 0–20% (group-aware). The rule asks for three quarters of the alone share (55%), about 41%.

**What failed is the labelling, not the removal.** The rule was meant to catch hidden prompts coming out "not
needed". That never happens: 0 of 120 hidden prompts per wording are labelled not needed.

| Neutral, hidden "?" (k = 1 / 2 / 3) | Alone |
|---|---|
| bridge removal drops P(right) by 0.98 / 0.89 / 0.92 (median) | 1.00 |
| log-odds fall from the bridge 4.8 / 4.1 / 4.4; from the worst everyday word 0.75 / 0.51 / 0.68 | 7.7 / 0.45 |
| prompts with no early window 3 / 5 / 6; inconclusive (a control dropped ≥ 0.2) 11 / 9 / 8 | 0 / 9 |
| 2×2 B collapses (≥ 2 log-odds) in 18 / 18 / 17 of 20 | |

"(answer submitted)" is the same or stronger (bridge fall 5.0–6.3 log-odds). With the **group-aware** wording the
hidden prompt is not a neutral baseline: P(right) falls to about 0.4, the answer forms by layer 13 or earlier
(median onset 1.5–13.5 by condition), and up to 13 of 20 prompts have no window before the answer forms; the bridge fall is 2.2–3.3 log-odds.

**Weakens** (need_LO in the hidden prompt minus need_LO in the agree prompt with as many peers ≥ 2, assessable
questions only):

| Peers | neutral, vs "?" | neutral, vs "(answer submitted)" | group-aware, vs "?" |
|---|---|---|---|
| 1 | 8 / 16 | 10 / 16 | 3 / 13 |
| 2 | 11 / 14 | 10 / 12 | 8 / 14 |
| 3 | 10 / 13 | 10 / 13 | 8 / 14 |

**Dose-response** (median log-odds fall of P(right) at removal strength 0.25 / 0.5 / 0.75 / 1, 20 questions):

| | alone | hidden 3 | agree 3 |
|---|---|---|---|
| neutral | 2.3 / 5.5 / 7.1 / 7.6 | 1.2 / 2.8 / 3.9 / 4.5 | 0.7 / 1.1 / 1.3 / 1.4 |
| group-aware | 2.3 / 5.5 / 7.1 / 7.6 | 0.5 / 1.2 / 2.2 / 2.8 | 0.16 / 0.24 / 0.26 / 0.33 |

All three curves rise with strength. Agree sits below hidden at every strength and flattens early.

**Verdict.** In the neutral wording removal works in the group layout: smooth dose-response, and B collapses the
answer in 17–18 of 20 hidden prompts. The label rule fails because group prompts lose early windows and move the
controls more. In the group-aware wording the hidden prompt itself weakens the bridge, so it is a weak baseline
there.

## 4. Block validation (step 6; pass = ≥ 3/4 of eligible prompts)

| Row | neutral | group-aware | Passed |
|---|---|---|---|
| Copy-only (made-up subject, 3 wrong peers): P(peers') back toward the no-peer level (≥ half the gap) | 4 / 4, median share of gap closed 1.0 | 4 / 4, 1.0 | yes / yes |
| Give-ins under disagree: right letter beats the peers' under C | 19 / 19 | 58 / 64 | yes / yes |
| Hidden prompts: P(right) moves ≤ 1 log-odds (median ≤ 0.5) | 84 / 96, median 0.38 | 84 / 96, 0.45 | yes / yes |
| Told to copy: P(peers') falls ≥ 2 log-odds | 24 / 24, median 8.4 | 24 / 24, 8.4 | yes / yes |

**All four rows pass in both wordings.** The answer-only block (C_late: full-attention layers, reply queries only,
linear layers open) catches 0 of 24 told-to-copy cases and 10 of 64 group-aware give-ins. The peers' influence
reaches the answer through the linear-attention layers and earlier positions, not as a late direct read.

## 5. The 2×2 per question (20 questions)

| Wording | Peers | C well below A (leaned on peers) | only D collapses (peers the backup) | both collapse | neither | only B |
|---|---|---|---|---|---|---|
| neutral agree | 1 | 0 | 3 | 14 | 2 | 1 |
| neutral agree | 2 | 0 | 10 | 8 | 2 | 0 |
| neutral agree | 3 | 1 | **15** | 3 | 1 | 0 |
| group-aware agree | 1 | 4 | **14** | 1 | 1 | 0 |
| group-aware agree | 2 | **17** | 3 | 0 | 0 | 0 |
| group-aware agree | 3 | **18** | 2 | 0 | 0 | 0 |
| neutral hidden | 3 | 0 | 0 | 17 | 2 | 1 |

Medians, agree 3: neutral dB 1.4, dC 0.4, dD 5.2; group-aware dB 0.3, dC 3.7, dD 6.5 log-odds.

**Headline.** With three agreeing peers and neutral wording, in 15 of 20 questions removing the bridge alone does
little (the peers carry the answer), but removing it with the peers blocked collapses the answer. The model still
holds the route; the peers are the backup. With the group-aware wording the model leans on the peers from the start:
blocking them alone costs 3.7 log-odds.

Disagree prompts: under D the answer moves to the peers' letter in 35 / 60 (neutral) and 34 / 60 (group-aware).
The peers are invisible in D, so this is the model's fallback guess. The wrong letters were drawn from same-kind
words (Spain's S for the Colosseum), which confounds "lost its route" with "fell back to the other country".

## 6. Patch (step 7, agree 3)

The "?" prompt's residual is written in at the peer positions at every layer. Median log-odds fall of P(right):
patch 0.10 against block 0.42 (neutral), patch 3.4 against block 3.7 (group-aware). Correlation across questions
0.85 / 0.90. Patch plus removal collapses the answer exactly where D does (39 of 40). Two methods, one answer.
Layer sweep, group-aware (18 questions with a patch effect ≥ 1): patching from layer L upward loses half its
effect from median L = 23 (IQR 19–23). Alone, the bridge acts in layers 18–23 or 21–26 in 16 of 20 questions
and the answer forms from layer 24 (median). So the peers are read in the same layers as the bridge acts, not
clearly after it. Neutral: only 8 questions have a patch effect ≥ 1, so the sweep is not read.

## 7. Other variants (states, 20 questions)

- **Told to ignore the peers** (3 wrong): 11 of 20 still give in fully. Told to copy: 20 of 20 copy.
- **Peer names a wrong country** ("The country is Spain.", 3 peers): 13 of 20 follow (neutral), 20 of 20
  (group-aware). Peer names the right country: 20 of 20 right.
- **Second wrong letter G**, 3 peers: 5 of 20 give in (neutral) against 10 for F; 20 of 20 group-aware.

## 8. What should change before the 27B run

1. **The hidden-check rule.** Judge it on assessed prompts, and on the failure it guards against (hidden "not
   needed" ≈ 0, plus B's collapse rate), with "inconclusive" and "no early window" counted apart. As written it
   fails on the 4B for reasons unrelated to the tool. Fix this before the 27B run, not after.
2. **Wrong letters.** Draw them from letters no same-kind word starts with, so disagree-D can be read.
3. **Patch placeholder.** On the real Qwen tokenizer all 24 right letters tokenize to the same length as "?" (the
   " ..." fallback was never used; the tiny model's tokenizer differs). Keep the check.
4. **Memory.** Eight 4B processes on one H200 hit out-of-memory errors. Set
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and use ≤ 6 processes per 80 GB.
5. **Group-aware hidden baseline.** It moves P(right) to about 0.4, so "shrinks" claims under this wording should
   use the neutral hidden prompt or be stated with that caveat.
