# GPU run 3: Qwen3.6-27B, the frozen confirmation set

*9 Nov 2026 UTC. The confirmation run of `docs/conformity/plan.md` Phase C (§7 order item 5, step 7b), on 20 new
questions, frozen in `frozen.md` and committed with the code (6e5cfa3) before any run job. Readings fixed in
`docs/conformity/scope.md` §11 ("Run 3 readings") before the run. Tables from `consulted_bridges.conformity_v3_report`
(`report.txt` and the CSVs in this folder). Nothing was tuned after results.*

## Setup

- **Question.** A two-hop question (Moai statues → Chile → first letter C) has a hidden middle step, the **bridge**
  (Chile). **Removal** takes the bridge's J++ lens direction out of the residual stream, from the question's first
  word to the end of the prompt, in one 6-layer window before the answer forms (layer 54 of 64). If P(right)
  collapses, the model uses the bridge. Scored in log-odds (LO) of P(right).
- **Group prompts.** Peers answer first, then the model: **agree** (the right letter), **disagree** (one wrong
  letter), **hidden** (peers write "?" or "(answer submitted)"), **told to copy**. **Neutral** wording: "Give your own
  answer". **Group-aware**: "Taking the other agents' answers into account".
- **The 2×2.** A no edit, B bridge removed, C peers blocked (no later position can read the peers' answers, every
  layer), D both. A cell **collapses** when it falls ≥ 2 LO below A. Readings: *both collapse* (bridge needed either
  way), *only D* (peers were the backup), *leaned* (C alone ≥ 2 below A: the model leaned on the peers from the start),
  *neither*, *only B*.
- **Swap** (new, plan.md step 7b). Neuronpedia's move: the bridge's lens component is moved onto a same-kind word
  (Chile → Peru) on the question sentence's tokens, layers 0 to onset − 1. *Follows* = the greedy answer becomes that
  word's answer (P). Control: the same move onto an everyday word (table → T) must not produce its letter.
- **What changed from run 2** (all before the run): disagreeing peers give a **"free" wrong letter**, one no same-kind
  or swap word starts with (run 2 used the first same-kind word's letter, Spain's S, so D's destination could not be
  read); number questions get a nearby digit no same-kind word gives. **Reach** edits only the positions the naming
  prompt shares with the original prompt. The 2×2 records P(same-kind answer) too.

## Frozen set and funnel

The confirmation half of the candidates (55 active), through the run-2 J++ gate screen on the 27B:

| Family | Drafted | Knows it | Need ≥ 0.5 | Selected | + reach (frozen) |
|---|---|---|---|---|---|
| Letter, landmark → country | 14 | 14 | 10 | 10 | 8 |
| Letter, university → city | 9 | 9 | 7 | 7 | 6 |
| Letter, person → birth city | 10 | 7 | 3 | 3 | 2 |
| Element → atomic number | 2 | 2 | 2 | 2 | 2 |
| Planet → position | 2 | 2 | 1 | 1 | 1 |
| Players per side | 1 | 1 | 1 | 1 | 1 |
| Months, strings, continent, driving side, vowel, even/odd | 17 | 16 | 0 | 0 | 0 |
| **All** | **55** | **51** | **24** | **24** | **20** |

C_LU_case (controls replaced at the fact pass) was re-screened with its new controls before the freeze: need 0.95,
reach 0.60, kept. 16 letter and 4 number questions.

## Ran

| Step | What | Outcome |
|---|---|---|
| Leak test | block at every layer, peers' letter swapped, on each install | **Pass, exact** on all 15 installs (blocked difference 0) |
| Conformity | 20 questions × 2 wordings: alone, agree 1–3, disagree 1–3, hidden "?" and "(answer submitted)" 1–3, told to copy; own-part removal | 560 prompts |
| Block, 2×2, dose, patch | as run 2 | all questions; patch skipped on the 4 number questions (no same-length placeholder for a digit) |
| Swap | alone, agree 3, disagree 3; visible and blocked; everyday control | all questions; one prompt without an onset (Tikal, group-aware disagree 3) |

**Pods.** 12 H200s off the volume (model and lenses downloaded, ~2–9 min of setup), 2 H200s and the 3 A100s on
the volume, one job per question and wording (40 jobs) through a job queue. GPU work 09:46–10:17 UTC. The two volume
H200s hung reading the network volume (16 min, no prompt); their four jobs were moved to running pods and the two
deleted. Eight prompts (C_PL_saturn neutral, C_SP_gretzky group-aware: disagree 1–3 and told to copy) hit CUDA
out-of-memory, two 27B processes on one H200 with the number questions' long answer lists; those two jobs were re-run
whole, unchanged code, one process per H200, 10:28–10:40. Every pod was deleted when its jobs finished.

## 1. The hidden check (§11 run-3 reading): passes in both wordings

| Wording | Hidden prompt | Prompts | No early window | Inconclusive | Assessed (clean) | "Not needed" | B collapses (2×2) |
|---|---|---|---|---|---|---|---|
| neutral | "?" 1 / 2 / 3 | 20 each | 0 | 6 / 5 / 5 | 14 / 15 / 15 | **0** | **20 / 20 / 20** of 20 |
| neutral | "(answer submitted)" 1 / 2 / 3 | 20 each | 0 | 4 / 5 / 5 | 16 / 15 / 15 | **0** | 20 of 20 (k = 3) |
| group-aware | "?" 1 / 2 / 3 | 20 each | 0 | 6 / 7 / 6 | 14 / 13 / 14 | **0** | **20 / 20 / 20** of 20 |
| group-aware | "(answer submitted)" 1 / 2 / 3 | 20 each | 0 | 7 / 7 / 8 | 13 / 13 / 12 | **0** | 20 of 20 (k = 3) |

Pass marks: "not needed" ≤ 1 in 10 of the assessed prompts, B collapses in ≥ 3/4. No hidden prompt is "not needed";
removal collapses the answer in every hidden prompt the 2×2 runs. The old rule (needed share ≥ 3/4 of alone's 17/20)
passes in 11 of 12 cells and misses group-aware "(answer submitted)" 3 by one question (12 against 12.75), all through
inconclusive labels, as in run 2. Baselines are healthy: P(right) median 0.99 (neutral) and 0.97 (group-aware) in
hidden prompts, the onset at layer 54 in every neutral hidden prompt and 86% of group-aware ones.

## 2. Old labels against log-odds (alone and agree)

| Wording | Peers | Old: needed / not needed | LO: needed | LO: not needed | LO: mixed | Inconclusive | At ceiling | Median need_LO | Median reach (shared / q→end) |
|---|---|---|---|---|---|---|---|---|---|
| — | alone | 20 / 0 | 17 | 0 | 0 | 3 | 0 | 8.9 | 0.15 / 0.99 |
| neutral | 1 | 14 / 0 | 15 | 0 | 1 | 4 | 2 | 5.9 | 0.13 / 0.95 |
| neutral | 2 | 9 / 1 | 12 | 0 | 5 | 3 | 7 | 4.8 | 0.24 / 0.90 |
| neutral | 3 | 9 / 1 | 14 | 0 | 5 | 1 | 6 | 4.6 | 0.29 / 0.90 |
| group-aware | 1 | 8 / 1 | 12 | 0 | 6 | 2 | 5 | 4.0 | 0.16 / 0.93 |
| group-aware | 2 | 2 / 6 | 7 | 3 | 10 | 0 | 12 | 1.2 | 0.20 / 0.89 |
| group-aware | 3 | 2 / 7 | 5 | 3 | 12 | 0 | 12 | 1.1 | 0.27 / 0.89 |

As on the development set, the probability labels understate need at the ceiling (P(right) ≥ 0.9 after removal):
in log-odds the neutral wording still needs the bridge in 14 of 20 at three peers. **Reach on shared positions is
much weaker**: median 0.13–0.29 in agree prompts against 0.89–0.95 with the old q→end edit; it passes in 47% of all
prompts against 87%. So many low-need agree prompts are now "mixed" rather than "not needed": the null cannot be read
there. The 3 + 3 group-aware "not needed" are Moai, Goethe, UNSW (reach 0.67–0.95), each with need_LO < 1.

**Weakens** (need_LO in the hidden prompt minus agree with as many peers ≥ 2), 20 questions: neutral 12 / 17 / 15
at 1 / 2 / 3 peers against "?", group-aware 12 / 17 / 16.

## 3. Block validation (pass = ≥ 3/4 of eligible prompts)

| Row | neutral | group-aware |
|---|---|---|
| Copy-only (made-up subject) | no cases (calibration questions only) | no cases |
| Give-ins under disagree: right beats the peers' letter under C | 1 / 1 (the only give-in), +5.2 LO — pass (one case) | 20 / 20, median +5.2 LO — pass |
| Hidden prompts: P(right) moves ≤ 1 LO (median ≤ 0.5) | 72 / 80, median 0.28 — pass | 71 / 80, median 0.32 — pass |
| Told to copy: P(peers') falls ≥ 2 LO | 20 / 20, median 12.6 — pass | 20 / 20, median 12.6 — pass |

## 4. The 2×2 (20 questions) — the headline replicates

| Wording | Peers | both collapse | only D (peers the backup) | leaned on the peers | neither | only B |
|---|---|---|---|---|---|---|
| neutral agree | 1 | 20 | 0 | 0 | 0 | 0 |
| neutral agree | 2 | 13 | 4 | 2 | 1 | 0 |
| neutral agree | 3 | **17** | 1 | 1 | 1 | 0 |
| group-aware agree | 1 | 16 | 3 | 0 | 1 | 0 |
| group-aware agree | 2 | 7 | 8 | 4 | 1 | 0 |
| group-aware agree | 3 | 6 | **10** | 2 | 2 | 0 |
| neutral hidden | 3 | 20 | 0 | 0 | 0 | 0 |
| group-aware hidden | 3 | 18 | 0 | 1 | 0 | 1 |
| neutral disagree | 3 | 17 | 0 | 0 | 1 | 2 |
| group-aware disagree | 3 | 5 | 0 | 1 | 4 | 10 |

Medians at agree 3: neutral dB 4.8, dC 0.3, dD 9.7; group-aware dB 1.4, dC 0.8, dD 7.0 LO (run 2: 4.2 / 0.4 / 10.2
and 1.6 / 0.6 / 7.9).

**Per question** (agree 1 and 3, disagree 3, hidden 3):

| question | neutral agree 1 | neutral agree 3 | group-aware agree 1 | group-aware agree 3 | neutral disagree 3 | group-aware disagree 3 | neutral hidden 3 | group-aware hidden 3 |
|---|---|---|---|---|---|---|---|---|
| C_EL_helium | both | both | both | both | both | both | both | both |
| C_EL_oxygen | both | both | both | both | both | only B | both | both |
| C_LM_alhambra | both | both | only D | leaned | both | only B | both | leaned |
| C_LM_bluelagoon | both | both | only D | only D | only B | only B | both | both |
| C_LM_chichen | both | only D | both | only D | neither | only B | both | both |
| C_LM_galapagos | both | both | both | only D | both | neither | both | both |
| C_LM_moai | both | both | both | only D | both | both | both | both |
| C_LM_tikal | both | both | both | only D | both | both | both | both |
| C_LM_timbuktu | both | neither | neither | neither | only B | only B | both | only B |
| C_LM_vasa | both | both | both | both | both | neither | both | both |
| C_LP_dante | both | both | both | only D | both | neither | both | both |
| C_LP_kanye | both | both | both | only D | both | both | both | both |
| C_LU_bohr | both | both | both | only D | both | only B | both | both |
| C_LU_case | both | both | only D | neither | both | both | both | both |
| C_LU_emory | both | both | both | both | both | neither | both | both |
| C_LU_goethe | both | both | both | only D | both | only B | both | both |
| C_LU_mcgill | both | both | both | both | both | only B | both | both |
| C_LU_unsw | both | both | both | only D | both | only B | both | both |
| C_PL_saturn | both | both | both | both | both | only B | both | both |
| C_SP_gretzky | both | leaned | both | leaned | both | leaned | both | both |

**Reading.** The development result holds on new questions. With the neutral wording and three agreeing peers the
27B keeps using its own route: removing the bridge collapses the answer with the peers visible in 17 of 20 (run 2:
14 of 19). With the group-aware wording the peers become the backup in half the questions (only D in 10 of 20; run
2: 7 of 19) and the bridge is needed either way in 6. Leaning on the peers from the start is rarer than in
development (2 against 4 at three peers): the development leaners were person questions and Fuji, and the frozen set
has two person questions. Timbuktu (Mali) is the one question where neither cell collapses in both wordings.

**Disagree: where D goes, now readable.** With the free wrong letters the destination can be read: under D (bridge removed, peers blocked) the right letter, the peers' letter and the first same-kind word's letter are scored. At three disagreeing peers D collapses in 17 of 20 (neutral) and 6 of 20 (group-aware). P(peers') > P(right) under D in **3 of 20** (neutral) and **2 of 20** (group-aware), against 13 of 19 in run 2, where the peers' letter *was* the same-kind letter. Among the three letters, D's top choice is still the right letter in 12 (neutral) and 14 (group-aware), the same-kind letter in 5 and 4, the peers' letter in 3 and 2. So "lost its route" shows as a drop that mostly stays on the model's own letter or drifts to the same-kind word's, not as a move to the peers: the run-2 destination finding was the confound.

**All block validation rows with cases pass.** Disagreement persuades less with the free letters: full give-ins at three peers 0 of 20 (neutral; run 2: 6 of 19) and 8 of 20 (group-aware; run 2: 12 of 19); told to copy, 20 of 20 in both wordings.

## 5. Swap toward another bridge (plan.md step 7b)

| Wording | Prompt | Visible: follows | Visible: says the peers' letter | Blocked: follows | Everyday word gets through (visible, blocked) |
|---|---|---|---|---|---|
| neutral | alone | **18 / 20** | — | — | 0 / 16 |
| neutral | agree 3 | 15 / 20 | 4 (= the right letter) | **19 / 20** | 0 / 16, 0 / 16 |
| neutral | disagree 3 | **18 / 20** | 1 | 19 / 20 | 0 / 16, 0 / 16 |
| group-aware | alone | **18 / 20** | — | — | 0 / 16 |
| group-aware | agree 3 | 6 / 20 | 12 (= the right letter) | **19 / 20** | 0 / 16, 0 / 16 |
| group-aware | disagree 3 | 7 / 19 | 11 | **16 / 19** | 0 / 15, 0 / 15 |

Letters 16 questions, numbers 4 (the number questions have no everyday-letter control; they follow in 3 of 4 alone and
4 of 4 under the block in every condition). One group-aware disagree-3 prompt (Tikal) has no onset and is left out.

**Reading.** The control holds everywhere (the everyday word never produces its letter), so every cell can be read (§11). Alone, moving the bridge's component onto a same-kind word makes the model give that word's letter in 18 of 20 questions: the letter is computed from the bridge. Under the peer block the swap follows in 16–19 of 20 in every group prompt, about as often as alone. With the peers visible it depends on the wording: neutral, the swap still wins against three agreeing peers in 15 of 20 and against three disagreeing peers in 18 of 20; group-aware, it wins in only 6 and 7, and the model says the peers' letter instead (12 and 11). This is the positive counterpart of the 2×2: in the neutral wording the model's own bridge produces the answer even next to the peers; in the group-aware wording the visible peers override it, and the bridge's route is still there underneath (the blocked swap follows).

## 6. Dose-response and patch

| Median LO fall at strength 0.25 / 0.5 / 0.75 / 1 | alone | hidden 3 | agree 3 |
|---|---|---|---|
| neutral | 3.8 / 7.8 / 8.9 / 9.4 | 3.7 / 8.5 / 9.4 / 9.4 | 1.3 / 4.1 / 4.6 / 4.8 |
| group-aware | 3.8 / 7.8 / 8.9 / 9.4 | 2.8 / 5.5 / 7.0 / 7.5 | 0.5 / 0.9 / 1.3 / 1.4 |

Every curve rises with strength and saturates by 0.75; agree sits below hidden at every strength in both wordings,
as in run 2. Patching the hidden "?" prompt's residual into the peer positions at every layer (16 letter questions; no digit placeholder for the 4 number questions) tracks the block in the neutral wording (correlation of P_all with C 0.93; run 2: 0.99) and less well in the group-aware one (0.55; run 2: 0.95), where both effects are small (median fall from A 0.35 and 0.46 LO). Patch plus removal collapses wherever D does (16 of 16 in both wordings). The layer sweep has an effect worth locating in 5 (neutral) and 4 (group-aware) questions; patching stops working when it starts at layers 13–43 and 10–51.

## 7. Replication of the development findings (§11 rule)

Each share's confirmation value is checked against run 2's 95% bootstrap interval (resampling questions, 2,000
draws). **61 of 72 shares replicate.** Of the 11 that do not, 8 are against a development interval of width zero (0 of 19 or 19 of 19 in run 2) and miss it by one to four questions; the two intervals overlap in 7 of those 8. The three substantive misses: full give-in at three disagreeing peers in both wordings (expected: the wrong-letter rule changed, see §4) and leaning on the peers at one agreeing peer, group-aware (0 of 20 against 4 of 19; the development leaners were person questions). Selected rows:

| Share | Run 2 (19 development) | Run 3 (20 confirmation) | Replicates |
|---|---|---|---|
| neutral, agree 3: both collapse (**headline**) | 14/19 [0.53, 0.95] | 17/20 [0.70, 1.00] | yes |
| group-aware, agree 3: only D (**headline**) | 7/19 [0.16, 0.58] | 10/20 [0.30, 0.70] | yes |
| group-aware, agree 3: both collapse | 7/19 [0.16, 0.58] | 6/20 [0.10, 0.50] | yes |
| neutral, agree 3: LO needed | 14/19 [0.53, 0.89] | 14/20 [0.50, 0.90] | yes |
| group-aware, agree 3: LO needed | 8/19 [0.21, 0.63] | 5/20 [0.10, 0.45] | yes |
| alone: LO needed | 15/19 [0.58, 0.95] | 17/20 [0.70, 1.00] | yes |
| hidden: "not needed" (assessed) | 0 / 90 and 0 / 73 | 0 / 90 and 0 / 79 | yes |
| hidden 1–3: B collapses | 57/57 and 54/57 | 60/60 and 60/60 | yes |
| neutral, weakens at 3 peers | 17/19 [0.74, 1.00] | 15/20 [0.55, 0.90] | yes |
| group-aware, weakens at 3 peers | 19/19 [1.00, 1.00] | 16/20 [0.60, 0.95] | **no** |
| group-aware, agree 1: leaned on the peers | 4/19 [0.05, 0.42] | 0/20 | **no** |
| neutral, disagree 3: full give-in | 6/19 [0.11, 0.53] | 0/20 | **no** (wrong-letter rule changed) |
| group-aware, disagree 3: full give-in | 12/19 [0.42, 0.84] | 8/20 [0.20, 0.60] | **no**, just (0.40 against 0.42; wrong-letter rule changed) |

## 8. What this changes in the write-up

1. **The 2×2 headline is confirmed on held-out questions** in both wordings, with the frozen rules; it can be
   stated as a confirmed finding, with run 3's counts and intervals beside run 2's.
2. **The swap gives the positive counterpart of the removal null.** Moving the bridge onto a same-kind word moves the answer to that word's letter in 18 of 20 alone and 16–19 of 20 with the peers blocked; with the peers visible it still wins in 15–18 of 20 in the neutral wording but only 6–7 of 20 in the group-aware one. The everyday control never gets through. This can be stated as "the answer is produced from the model's own bridge", next to the removal result, replacing "the swap mostly fails" (scope.md §8, which was the exchange formula).
3. **Run 2's "D moves to the peers' letter" was mostly the same-kind confound.** With free wrong letters, D lands on
   the peers' letter in only 3 of 20 (neutral) and 2 of 20 (group-aware) questions at three peers, against 13 of 19 in run 2. Drop the run-2 sentence or present it as superseded.
4. **Plain disagreement persuades less with the free letters**: in the neutral wording no question gives in fully at
   three disagreeing peers (run 2: 6 of 19); group-aware GIVEIN3. Part of run 2's give-ins rode on a plausible
   (same-kind) letter. Say so where give-in rates are quoted.
5. **Reach on shared positions is weak** (median 0.13–0.29 in agree prompts): most low-need agree prompts are now
   unreadable nulls ("mixed"), not "not needed". Quote "not needed" counts with this caveat, or lean on the hidden
   prompt and the block, which do not depend on reach.
6. **Numbers are too few to read apart** (4 questions in 3 families); they behave like letters (both collapse
   at neutral agree 3 in 3 of 4; Gretzky leans on the peers). The number format stays below the §7 stop rule.
7. Two process notes for the methods: one 27B process per H200 for number questions (two OOM'd), and network-volume
   pods in CA-MTL-3 can hang on first read; off-volume download pods were reliable.
