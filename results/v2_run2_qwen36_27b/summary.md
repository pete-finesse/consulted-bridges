# GPU run 2: Qwen3.6-27B, candidate screen and the group-layout tools

*9 Nov 2026 UTC. The same protocol as run 1 (`results/v2_run1_qwen35_4b/summary.md`) on the main model. Plan:
`docs/conformity/plan.md` Phase C (steps 4–7). Pass marks fixed before run 1 in `docs/conformity/scope.md` §11.
Runner `scripts/pod/v2_run.sh` (step a) and the job-queue worker `scripts/pod/v2_jobs.sh` (the rest); tables from
`consulted_bridges.conformity_funnel --active`, `conformity_stats`, `conformity_v2_report`.*

## Setup

- **Question.** A two-hop question (Colosseum → Italy → first letter I) has a hidden middle step, the **bridge**
  (Italy). **Removal** takes the bridge's J++ lens direction out of the residual stream, from the question's first
  word to the end of the prompt, in one 6-layer window before the answer forms (the answer forms at layer 54 of 64
  on this model). If P(right) collapses, the model uses the bridge.
- **Group prompts.** Peers answer first, then the model. The peers agree (right letter), disagree (one wrong
  letter), or are hidden: they write "?" or "(answer submitted)". **Neutral** wording: "Give your own answer".
  **Group-aware** wording ("defer" in the files): "Taking the other agents' answers into account".
- **Peer block (C).** At every layer, no later position can read the peers' answer tokens: an attention mask on the
  full-attention layers, and zeroed input, update and decay at those tokens on the linear-attention layers.
- **The 2×2.** On each prompt: A no edit, B bridge removed, C peers blocked, D both. Scored in log-odds of P(right).
  A cell **collapses** when it falls ≥ 2 log-odds below A.
- **Questions.** 25 development letter questions (`conformity.QUESTIONS["qwen36_27b"]`). The 6 calibration
  questions are left out of every count except the validation rows, leaving 19. Two of the 19 (V_LP_hendrix,
  V_LU_pompeu) were selected with the J-lens and are not J++-selected. Screen: the 171 v2 candidates under both
  lenses, reported on the 167 active ones (the 4 DROPPED at the fact pass left out).

## Ran

| Step | What | Outcome |
|---|---|---|
| a | Leak test, block at every layer, peers I vs F (and "?") on 4 questions | **Pass, exact**: blocked logit difference 0.0 on all 7 pairs (unblocked 2.9–8.7). Default kernels; `USE_HUB_KERNELS=0` not needed. Repeated on the off-volume H200 install: the same |
| b | Trimmed solo scorecard on 171 candidates × 2 lenses | 342 cards in 24 shards, none failed |
| c+d | Conformity, J++, every B1 variant plus own-part removal, both wordings | 1,112 prompts in 26 shards; no prompt failed |
| e+f | Block, 2×2, dose-response, patch; both wordings | every shard; patch on all 25 agree-3 prompts per wording |

Pods: CA-MTL-3 had no H200 and only three A100s free, so the run used 3 A100 80 GB on the shared volume
(leak test, then chain shards) and 11 H200s in other regions without the volume (model and both lenses downloaded,
about 2 min each; two 27B processes per H200), joined through a job queue. GPU work 08:37–09:17 UTC. One process
hit CUDA out-of-memory while loading (a third process put on one H200 by mistake). It wrote nothing and was re-run.
The 11 H200s are deleted; the 3 A100s are kept for the confirmation run.

**Code snapshot.** The pods ran the code as of 08:42 UTC, before the candidate fact pass (commit 2180ed4). Ids
are unchanged, so the screen stands. The nine replaced same-kind controls were screened with the old words. The
trimmed screen's selection (need, comes first) does not use them.

## 1. Funnel (gate screen, both halves; the confirmation half as counts only)

Selected = need ≥ 0.5 and the bridge comes first (the existing rule). "+reach" = also reach ≥ 0.5.

| Family | Drafted | Knows it | Selected J-lens | Selected J++ | J++ + reach |
|---|---|---|---|---|---|
| Letter, landmark → country (C_LM) | 38 | 38 | 22 | 29 | 27 |
| Letter, university → city (C_LU) | 27 | 27 | 13 | 18 | 15 |
| Letter, person → birth city (C_LP) | 29 | 24 | 10 | 11 | 10 |
| Month → number (C_MN) | 14 | 14 | 1 | 1 | 1 |
| Players per side (C_SP) | 5 | 5 | 0 | 4 | 4 |
| Continent (C_CT) | 12 | 12 | 0 | 0 | 0 |
| Element → atomic number (C_EL) | 6 | 6 | 2 | 3 | 3 |
| Planet → position (C_PL) | 5 | 5 | 1 | 3 | 2 |
| Instrument → strings (C_IN) | 2 | 2 | 1 | 1 | 1 |
| Driving side (C_DS) | 11 | 11 | 0 | 1 | 1 |
| Starts with a vowel, yes/no (C_VC) | 14 | 7 | 0 | 0 | 0 |
| Even/odd letter count (C_EO, built to fail) | 4 | 2 | 0 | 0 | 0 |
| **All** | **167** | **153** | **50** | **71** | **64** |

By half: development 34 / 47 of 112 selected (J-lens / J++), confirmation 16 / 24 of 55. Of the development
questions, 31 are selected by both lenses, 3 by the J-lens only and 16 by J++ only.

**Reading.** The 27B knows almost every fact (153 of 167, against 133 of 171 on the 4B). It loses candidates at
the next step instead. Of the 117 J-lens and 96 J++ rejections, 71 and 43 are "untestable": nothing dropped, and
the edit does not remove the bridge from the prompt, so the null cannot be read. The rest are spread out (16 / 17)
or bystanders (14 / 15). So on the 27B the screen is limited by **reach**, and J++ reaches more (71 selected
against 50). Letters survive in landmarks (29 of 38 with J++) and universities (18 of 27). Persons do worse
(11 of 29). Of the number formats only players per side (4 of 5, J++ only) and element and planet (3 each) pass at
all. Months (1 of 14) and continent (0 of 12) fail here, although months passed on the 4B.

## 2. Old labels against log-odds, and the replication

The base conditions reproduce the earlier J++ run on the 27B (round 14, 21 questions then, 19 now). Agree
prompts, 19 questions (round 14 in brackets):

| Wording | Peers | LO still needed | LO not needed | LO inconclusive | at ceiling | median need_LO |
|---|---|---|---|---|---|---|
| neutral | 1 | 14 (16) | 0 (0) | 3 | 2 | 5.4 |
| neutral | 2 | 14 (16) | 1 (1) | 2 | 4 | 3.8 |
| neutral | 3 | 14 (15) | 2 (1) | 2 | 5 | 3.2 |
| group-aware | 1 | 14 (16) | 0 (1) | 3 | 2 | 4.5 |
| group-aware | 2 | 7 (7) | 5 (6) | 0 | 11 | 1.4 |
| group-aware | 3 | 8 (7) | 5 (5) | 0 | 10 | 1.2 |

Alone: 15 of 19 needed (4 inconclusive), median need_LO 11.9. Disagree, three peers: the model gives in fully in
6 of 19 (neutral) and 12 of 19 (group-aware); the round-14 run had P(right) medians 0.75 and 0.04 there, this run
0.79 and 0.03.

## 3. Does removal still work in the group layout? (step 5)

**By the pre-registered rule: yes in the neutral wording, narrowly no in the group-aware one.** The rule asks, in
every hidden condition, for a "needed" share of at least three quarters of the alone share (15 of 19, so 11.25 of
19).

| Hidden condition, peers | neutral needed | group-aware needed | group-aware inconclusive |
|---|---|---|---|
| "?" 1 / 2 / 3 | 14 / 15 / 16 | **11 / 10 / 11** | 7 / 7 / 8 |
| "(answer submitted)" 1 / 2 / 3 | 14 / 15 / 14 | **11** / 14 / 12 | 7 / 5 / 7 |

Group-aware misses by one question in four of six cells.

**Against the failure the rule guards against** (the parent plan's reading after run 1, analysis only): no hidden
prompt is labelled "not needed" (0 of 114 per wording), and none lacks an early window (0 of 114). The whole
shortfall is "inconclusive": an everyday control word moved P(right) by ≥ 0.2. In the 2×2, B collapses the answer
(≥ 2 log-odds) in 19 / 19 / 19 hidden "?" prompts (neutral, 1–3 peers) and 18 / 17 / 19 (group-aware). Median need_LO
in the hidden prompts: neutral 9.3–11.1, group-aware 6.7–7.9, alone 11.9.

**The group-aware hidden baseline is not weak on the 27B.** On the 4B, P(right) fell to about 0.4 in the
group-aware "?" prompt and the answer formed by layer 13. On the 27B P(right) stays at 0.94–0.98 (median), the
answer forms at layer 54 as alone, and every prompt has an early window. The bridge's effect is smaller there
(need_LO about 7.5 against 10–11 neutral), and the controls move more.

**Weakens** (need_LO in the hidden prompt minus need_LO in the agree prompt with as many peers ≥ 2; 19 assessable):

| Peers | neutral, vs "?" | neutral, vs "(answer submitted)" | group-aware, vs "?" | group-aware, vs "(answer submitted)" |
|---|---|---|---|---|
| 1 | 7 | 6 | 11 | 11 |
| 2 | 17 | 17 | 15 | 16 |
| 3 | 17 | 17 | 19 | 19 |

**Dose-response** (median log-odds fall of P(right) at removal strength 0.25 / 0.5 / 0.75 / 1, 19 questions):

| | alone | hidden 3 | agree 3 |
|---|---|---|---|
| neutral | 5.7 / 11.6 / 12.4 / 12.6 | 6.0 / 10.7 / 11.1 / 11.2 | 2.3 / 4.2 / 4.2 / 4.2 |
| group-aware | 5.7 / 11.6 / 12.4 / 12.6 | 3.7 / 8.1 / 9.0 / 8.4 | 0.9 / 1.3 / 1.5 / 1.6 |

All curves rise with strength and saturate by 0.5–0.75. Agree sits below hidden at every strength.

**Verdict.** Removal works in the group layout on the 27B. The neutral wording passes the rule as written, and
the group-aware wording fails it only through inconclusive labels: B collapses the answer in 17–19 of 19 hidden
prompts, with a healthy baseline. With agreeing peers the bridge's effect drops by ≥ 2 log-odds in 17–19 of 19
questions at 2–3 peers, but in the neutral wording it mostly stays above the "needed" line (14 of 19).

## 4. Block validation (step 6; pass = ≥ 3/4 of eligible prompts)

| Row | neutral | group-aware | Passed |
|---|---|---|---|
| Copy-only (made-up subject, 3 wrong peers): P(peers') back toward the no-peer level (≥ half the gap) | 6 / 6, median share of gap closed 1.03 | 6 / 6, 1.08 | yes / yes |
| Give-ins under disagree: right letter beats the peers' under C | 19 / 19, median +3.2 log-odds | 36 / 38, +5.5 | yes / yes |
| Hidden prompts: P(right) moves ≤ 1 log-odds (median ≤ 0.5) | 96 / 100, median 0.28 | 90 / 100, 0.38 | yes / yes |
| Told to copy: P(peers') falls ≥ 2 log-odds | 25 / 25, median 10.0 | 25 / 25, 10.0 | yes / yes |

**All four rows pass in both wordings.** The answer-only block (C_late: full-attention layers, reply queries only,
linear layers open) catches 4 of 25 told-to-copy cases, as on the 4B. But it returns 15 of 19 (neutral) and 30 of
38 (group-aware) give-ins to the right letter; the 4B's returned 10 of 64 group-aware give-ins. On the 27B a plain disagreement
reaches the answer largely through a late direct read of the peers. An instruction to copy does not.

## 5. The 2×2 (19 questions)

| Wording | Peers | C well below A (leaned on peers) | only D collapses (peers the backup) | both collapse | neither | only B |
|---|---|---|---|---|---|---|
| neutral agree | 1 | 0 | 0 | 17 | 2 | 0 |
| neutral agree | 2 | 4 | 2 | 13 | 0 | 0 |
| neutral agree | 3 | 3 | 2 | **14** | 0 | 0 |
| group-aware agree | 1 | 4 | 1 | 14 | 0 | 0 |
| group-aware agree | 2 | 6 | 6 | 6 | 1 | 0 |
| group-aware agree | 3 | 4 | **7** | **7** | 1 | 0 |
| neutral disagree | 3 | 0 | 0 | 15 | 2 | 2 |
| group-aware disagree | 3 | 0 | 0 | 6 | 6 | 7 |
| neutral hidden | 3 | 0 | 0 | 19 | 0 | 0 |
| group-aware hidden | 3 | 0 | 0 | 19 | 0 | 0 |

Medians, agree 3: neutral dB 4.2, dC 0.4, dD 10.2; group-aware dB 1.6, dC 0.6, dD 7.9 log-odds.

**Per question** (agree and disagree, 1 and 3 peers; "both" = both collapse, "only D" = peers the backup,
"leaned" = C well below A):

| Question | neutral agree 1 | neutral agree 3 | group-aware agree 1 | group-aware agree 3 | neutral disagree 3 | group-aware disagree 3 | neutral hidden 3 | group-aware hidden 3 |
|---|---|---|---|---|---|---|---|---|
| LM_angkor | both | both | both | only D | both | only B | both | both |
| LM_eiffel | neither | both | both | both | both | both | both | both |
| LM_fuji | neither | leaned | leaned | leaned | both | both | both | both |
| LM_kremlin | both | both | both | both | both | both | both | both |
| LM_machu | both | both | both | both | neither | only B | both | both |
| LM_petronas | both | both | both | only D | both | neither | both | both |
| LM_pisa | both | both | both | both | both | both | both | both |
| LM_pyramids | both | both | both | both | both | both | both | both |
| LM_sagrada | both | both | both | neither | both | neither | both | both |
| LM_sydney | both | both | both | both | both | neither | both | both |
| LP_ali | both | both | both | leaned | both | only B | both | both |
| LP_beyonce | both | leaned | leaned | leaned | both | both | both | both |
| LP_hemsworth | both | leaned | leaned | leaned | both | only B | both | both |
| LP_hendrix | both | only D | both | only D | only B | only B | both | both |
| LP_obama_m | both | only D | leaned | only D | only B | neither | both | both |
| LU_alberta | both | both | both | only D | both | only B | both | both |
| LU_berklee | both | both | both | both | both | neither | both | both |
| LU_fudan | both | both | both | only D | both | only B | both | both |
| LU_pompeu | both | both | only D | only D | neither | neither | both | both |

**Reading.** On the 27B with the neutral wording, three agreeing peers do not switch the bridge off. Removing it
collapses the answer with the peers visible in 14 of 19 questions; on the 4B that happened in 3 of 20, and only
D collapsed in 15. The model keeps using its own route next to the peers. With the group-aware wording the
questions split: in 7 the peers are the backup (only D collapses), in 7 the bridge is needed either way, and in 4
the model leans on the peers from the start (all four are person or Fuji questions, where blocking the peers alone
drops P(right) by ≥ 2). On the 4B, 18 of 20 leaned at 3 peers. The person → city questions lean on the peers most
in both wordings.

**Disagree, D's destination.** With the peers blocked and the bridge removed, P(peers') > P(right) in 13 of 19
disagree prompts per peer count (neutral) and 12–15 (group-aware). This is confounded. The wrong letter is the
first same-kind word's letter (`conformity.wrong_letter`; Spain's S for the Colosseum). So removing the bridge can
send the model to that alternative's letter without any copying. The letters were kept as they are, so this run
compares with the earlier 27B runs. D's drop can be read; its destination cannot.

## 6. Patch (step 7, agree 3)

Patching the hidden "?" prompt's residual into the peer positions at every layer (P_all) tracks the block: the
correlation of P_all with C is 0.99 (neutral) and 0.95 (group-aware), with the median fall from A 0.2 and 0.3 log-odds (C: 0.4 and 0.6).
Patch plus removal (PB_all) collapses where D does in 18 of 19 and 17 of 19 questions. The layer sweep has an
effect worth locating (≥ 1 log-odds either way) in 6 questions (neutral) and 8 (group-aware), mostly person
questions. There, patching stops working when it starts at layer 35–59 for the person questions and Fuji, and at
23–27 for Eiffel, the Pyramids and Petronas (where the patch raises P(right)).

## 7. Variants

| Variant (3 peers unless noted) | neutral | group-aware | 4B, neutral / group-aware |
|---|---|---|---|
| Told to copy: full give-in | 19 / 19 | 19 / 19 | 20 / 20 |
| Told to ignore the peers: still gives in fully (partly) | 3 (2) / 19 | 3 (2) / 19 | 11 / 20 |
| A peer names the right country: right | 19 / 19 | 19 / 19 | 20 / 20 |
| A peer names a wrong country: full give-in, 1 / 3 peers | 2 / 2 of 19 | 2 / 4 of 19 | 4 / 13 and 7 / 20 of 20 |
| Disagree with a group-aware sentence in the neutral wording (disagree_g), 3 peers | 4 (2) / 19 | 12 / 19 | 5 / 20 |

The 27B follows instructions about the peers much as the 4B does, but resists a wrong country named by a peer
(2–4 of 19 against 13–20 of 20).

## 8. What should change before the confirmation set is run

1. **Hidden check: write the run-1 reading into scope.md §11 before the confirmation run.** Judge the tool on
   assessed prompts, the share of hidden prompts labelled "not needed" (≈ 0), and B's collapse rate, with
   inconclusive and no-early-window counted apart. On this run the label rule misses by one question in four of
   six group-aware cells, all through controls that moved. Without the fix the confirmation verdict would turn on
   one question.
2. **Pick the confirmation questions with J++ and require reach.** On the 27B reach is the binding gate (71 J-lens
   untestable against 43 J++), and J++ selects 24 of 55 confirmation candidates (20 with reach) against the
   J-lens's 16. Drop the families that select nothing on the 27B: continent, driving side, vowel class, even/odd,
   months (1 of 14).
3. **Re-screen the nine candidates whose same-kind controls were replaced** (fact pass, commit 2180ed4), and use
   the current split (167 active). This run screened them with the old controls.
4. **Wrong letters for disagree in the confirmation set:** draw them from letters no same-kind word starts with, so
   D's destination can be read. The development runs kept the old rule for comparability; the confirmation set has
   no earlier run to match. The per-format wrong-peer rule for the number and category families
   (`conformity.wrong_letter` returns a letter there) is needed before any of those enter a group run.
5. **Memory:** at most two 27B processes per H200 and one per 80 GB card, with `expandable_segments`. The only
   out-of-memory error came from a third process on one H200.
