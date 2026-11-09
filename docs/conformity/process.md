# Research process: the rounds

*How the work actually went, round by round: the rule we fixed first, what we drafted or ran, how many passed, what
we learned, and what changed for the next round. For the paper's development story and for not repeating a round
by accident. Add one entry per round, at the time, with the numbers. Companion files: `plan.md` (next steps),
`scope.md` (claims and defences), `methods.md` (the method as run, paper-ready), `EXPERIMENT_LOG.md` at the repo root
(one line per thing tried, with thread links). Started 8 November 2026; earlier rounds reconstructed from the
scorecard docs and the experiment log.*

**The loop every round follows.** Fix the rule before looking → draft or pick questions → screen them on the model
→ count what passes and why the rest fail → learn → change the rule, the question format, or the model → repeat.
The two things that keep the loop honest: the rule is written down before the run, and the fail counts are kept.

---

## Round 1: one question, by hand (before 6 Nov 2026, Qwen3.5-4B)

- **Set:** can removing a hidden word's J-lens direction change a two-hop answer at all?
- **Ran:** "How many legs do the animals that spin webs have?" Remove or swap "spider".
- **Found:** yes: the answer "8" moves. Also found the first confound: on Mozart → Salzburg → Austria, removing the
  bridge works partly through what Salzburg shares with Austria.
- **Changed:** the bridge and the answer must be separable; a scorecard is needed, not one example.

## Round 2: scorecard v1 on the paper's own questions (before 6 Nov, Qwen3.5-4B)

- **Set:** v1 criteria (need, specificity, isolation) with verdict rules, 6-layer windows, edits from the question's
  first word on.
- **Ran:** 6 hand sentences, then the Silent Dissent paper's own question set after a known-fact screen: the model
  knew 1,057 of 4,250 birth-city and university facts.
- **Found:** most of the paper's questions are shortcuts: where the shortcut gives a wrong answer (Sydney Opera House,
  Keanu Reeves) the model gives it. "Bystander" (removal does nothing) could also mean the lens misses the word.
- **Changed:** an outside review led to v2.

## Round 3: scorecard v2 with gates (before 6 Nov, Qwen3.5-4B)

- **Set, before the run:** four gates (knows both hops and the chain; edited words one token; the answer has an onset
  layer; the edit reaches the bridge in this prompt), then the tests; full questions shown; collateral damage
  report-only.
- **Ran:** 28 questions.
- **Found:** two consulted bridges (spider legs; York University → Toronto → T). Gate 4 (reach) failed most: for
  country bridges and several cities one J-lens direction is not how the model holds the word before the answer
  forms. Bystanders became trustworthy.
- **Changed:** question *format* became the lever: ask for a property of the bridge that the answer cannot echo.

## Round 4: scorecard v3, the letter format, and a quality number (before 6 Nov, Qwen3.5-4B)

- **Set:** families built so the answer is a property of the bridge only (first letter of the hidden word; leg-count
  riddles without a class word; parts of an object); quality = need × specificity × isolation.
- **Ran:** 18 new questions plus round 3's on the same scale.
- **Found:** letters fix isolation completely (1.00 on every testable letter question against 0.01 for the same
  Toronto bridge when the answer is the country). The model's knowledge is the limit: 4 of 9 letter questions wrong.
  Specificity was low because same-kind words overlap in lens space; the bridge's *own part* (minus all same-kind
  controls at once) still breaks the answer in 5 of 5, so "strict specificity" replaced it. A prefill bug had zeroed
  every reach check; 11 questions re-run.
- **Changed:** letters became the main format; strict specificity; onset from the answer's own part after Peter's
  objection that overlap pulls the onset early (v4, v4.1).

## Round 5: 44 creative questions from four designers (before 6 Nov, Qwen3.5-4B)

- **Set:** four sub-agents each drafted a category (sequences, language, science, culture) with deliberately bad
  questions as checks; quality with strict specificity.
- **Ran:** 44 questions, 8 of them bad on purpose.
- **Found:** 23 of 44 fail gate 1: the model knows both hops but cannot chain letters at word ends, syllables,
  reversal, alphabet order, planet or month order, Pokémon. Last letter is untestable. All 8 bad questions were
  caught, but by the gates, not the quality factors. Hockey players-per-side reads like spider.
- **Changed:** the ideas sets were retired as a source of questions (kept as `screened_out`/`retired`). Lesson for
  drafting: the second hop must be something the model can do on a word it never writes.

## Round 6: Qwen3.6-27B and the question bank (before 7 Nov)

- **Set:** the same scorecard on the 64-layer model on a rented GPU; a bank table with one verdict per question.
- **Ran:** 62 core questions across 8 families on both models.
- **Found:** the letter questions carry the bank on both models; Colosseum, Gates, Piaf score 0.96–0.98 on the 27B.
- **Changed:** the bank became the development set; a method critique by a separate reviewer followed.

## Round 7: close variants and trend questions (6 Nov)

- **Set:** two drafted sets with predictions written first: 38 close variants of the questions that worked (landmark →
  country, person → birth city, university → city letters; players per team; days in a month; legs), and 23 questions
  picked to test trends about which bridges work.
- **Ran:** both sets on both models.
- **Found:** close variants: 17 of 38 pass on the 4B, 15 of 38 on the 27B, 10 on both. Trend set: few are testable.
  This is the first measured **pass rate for drafted letter questions: roughly two in five to one in two per model**,
  higher within the landmark family.
- **Changed:** drafting now copies the shape of questions that pass. Swap test added and re-run with the move
  formula (first attempt 10 of 240 follow; with the move, editing question words only, about 7 in 10 follow).

## Round 8: the group pilot and the give-up experiments (6–7 Nov, Qwen3.6-27B)

- **Set:** the 6 best letter questions asked alone, with 3 agreeing peers, with 3 disagreeing peers, the pilot's
  wording; same scorecard, same thresholds.
- **Found:** with the right answer from peers, 5 of 6 stop needing the bridge; with a wrong answer, 5 of 6 give in;
  only Colosseum keeps its bridge throughout. Give-up follow-up with 1–3 wrong peers: when the model holds, the
  bridge holds it; when it gives in partly, the bridge still pulls (Mozart 0.47 → 0.00, spider 0.36 → 0.00); when
  it gives in fully, the leftover pull is near the same-kind controls.
- **Changed:** the narrow proposal ("Bridge Under Conformity"): one claim, trimmed checks, two group conditions, a
  copy-only control, a neutral wording in place of the pilot's.

## Round 9: the conformity run (7 Nov, both Qwen models)

- **Set, before the run:** selection rule = solo need ≥ 0.5 and the bridge acts first; labels and thresholds fixed
  on the 6 pilot questions, which are then reported separately; neutral wording; 1–3 peers.
- **Ran:** 25 of 37 letter questions pass selection on the 27B, 24 of 37 on the 4B. 706 prompts. The neutral
  wording gave few give-ins, so the pilot's wording was added as a second condition during the run.
- **Found:** agreeing peers make the bridge stop mattering while the answer stays right and the bridge stays readable
  (present but not needed); holding against wrong peers is the bridge holding; giving in still shows the bridge
  pulling in about half the 27B cases.
- **Changed:** a second answer format was wanted so the result is not only about spelling.

## Round 10: count questions, two drafting rounds (8 Nov)

- **Set, before drafting:** three screens at drafting time (the bridge's name does not contain the number; the answer
  is not the default for the kind; the description does not carry the number); answers 1–9 because Qwen splits
  digits; shapes copied from the bank's few passing counts.
- **Ran:** 29 drafted + 3 anchors on both models. 27B: 4 new consulted (3 of them person → birth country → count).
  4B: 1. Most failures: the model gets the two-hop question wrong while getting each hop right (0 World Cups for
  Spain, Germany, Italy), or the edit never reaches the bridge (bands, Mars, Fuji). Round two: 24 more
  person → birth country → count on the 27B → 16 consulted in all, but 14 rest on two facts (Austria 9, Australia 6).
- **Found:** **pass rate for drafted count questions: about one in three on the 27B, near zero on the 4B**, and
  copying one passing shape produces many questions on the same fact. Conformity was run on the 16.
- **Changed:** the unit of analysis became the question, or the fact where questions share one; distinct facts
  became a drafting rule.

## Round 11: other models (8 Nov)

- **Set:** the same selection rule applied to each model's own solo screen of all 37 letter questions.
- **Ran:** Gemma-4-E4B-it with the published base-model lens: 8 of 37 selected (13 fail the onset gate at P ≈ 1);
  with the paper's chat-model refit lens: 1 of 37. Llama-3.1-8B-Instruct: 15 of 37 (19 untestable, 3 spread out).
  Conformity on each selection, both wordings. Seen and later diagnosed: under the group-aware wording Llama's bridge
  ranks ≥ 100 in every group prompt because the wording fills the last token.
- **Found:** the lens matters as much as the model; no calibration question passed on Gemma, so it has no
  copy-only control.
- **Changed:** intervals by question and threshold sensitivity added to the stats; the J++ lens tried on both Qwen
  models (results pending as of 8 Nov).

## Round 12: the plan review and the candidate draft (8 Nov)

- **Set:** three plans combined into one; a check of the saved agree prompts before any new run.
- **Found in the saved data:** on the 27B, 22 of 42 (neutral) and 40 of 61 (group-aware) "not needed" agree prompts
  still have P(right) > 0.9 after removal, with the log-odds falling by about 2 and 1 respectively: half the "not
  needed" verdicts are a ceiling effect. On the 4B, up to 8 of 40 have nonspecific damage. The bridge is present in
  100% of alone and agree prompts, so presence cannot sort prompts.
- **Set for the next rounds:** the four-check gate; the hidden-answer prompt as the baseline for "shrinks"; log-odds
  beside probability; development/confirmation split; the stop rule (≥ 20 gated per format, ≥ 2 formats, on the 27B,
  at most two more drafting rounds).
- **Drafted:** 172 candidates in 12 families with the drafting rules written first. The Qwen token check then dropped
  the whole currency family (every currency name is two tokens) and 28 two-token bridges, and recast Gemini's
  vowel/consonant format as yes/no. Split 112 development / 60 confirmation by seed; digest committed.
- **Forecast, to check against the screen:** about 65 of 99 letters pass on each Qwen model; about 10 of 32 number
  questions on the 27B and almost none on the 4B; the category and yes/no families unknown.

## Round 13: the log-odds re-analysis, A1 (8 Nov, all saved runs, no GPU)

- **Set, before computing (scope.md §7):** need in log-odds = largest early bridge fall − largest early everyday-word
  fall; needed if ≥ 2, not needed if < 1 with reach; ceiling flag at P(right) after removal ≥ 0.9; inconclusive if
  an everyday-word or same-kind drop ≥ 0.2; clean subset = reach, present, same-kind quiet, onset within one step of
  alone. Output to `results/conformity_stats_v2/` (`summary.csv`, `prompts_lo.csv`); the old tables untouched.
- **Found, Qwen3.6-27B neutral:** the headline moves. With three agreeing peers the bridge was "still needed" in 5 of
  19 by probability; in log-odds it is 10 of 19 (interval 0.32–0.74), and 9 of 15 on the clean subset. 7 of the 19
  sit at the ceiling. Need in log-odds falls from 8.4 alone to 5.9, 3.0 and 2.2 with 1, 2, 3 peers: agreeing peers
  **weaken** the bridge's effect on the neutral wording; they do not switch it off.
- **Found, Qwen3.6-27B group-aware:** the old reading holds. 2 of 19 still needed at three peers in log-odds, 10 not
  needed, need median 0.55, 14 of 19 at the ceiling: here removal really does nothing.
- **Found, Qwen3.5-4B:** the inconclusive rule flags 12 of 20 *alone* prompts, because same-kind words drop 0.38 on
  median even alone on this model. That is a statement about the probe's specificity on the 4B, not about peers.
  Ignoring that rule, need in log-odds still falls from 6.3 alone to 2.1 / 1.5 / 1.2 at 1 / 2 / 3 peers (needed: 13,
  6, 1 of 20), and under the group-aware wording to about 0 (not needed 18–19 of 20): on the 4B the old reading
  survives in log-odds. No 4B prompt is at the ceiling.
- **Found, Llama-3.1-8B neutral:** 9 of 90 agree prompts have no early window: the answer forms in the first layer
  window once peers are present (also 4 of 120 on the 4B, 10 of 114 with the J++ lens, 11 and 24 of the disagree
  prompts). Those prompts are unassessable by removal and are themselves evidence for a copy route that is fast.
  Llama also has 8–13 of 15 agree prompts inconclusive while its alone prompts are quiet: the controls start moving
  when peers appear, which needs a look before Llama's group numbers are quoted.
- **Changed:** the presentation's finding 1 is reworded per model and wording: "weakens" on the 27B neutral,
  "stops" on the 27B group-aware and on the 4B. Report log-odds beside probability everywhere. The 4B's same-kind
  overlap goes into the funnel as a probe limit. "No early window" becomes a reported category.

## Round 14: the J++ lens against the J-lens (8 Nov, Peter's runs, both Qwen models)

- **Set:** the same trimmed solo screen, selection rule and conformity run, with the J++ lens (`configs/*_jpp.yaml`)
  in place of the J-lens; `scripts/jpp_compare.py` and `results/jpp_compare/`.
- **Found, selection:** letters 27 of 37 (J++) against 25 (J-lens) on the 27B, 23 against 24 on the 4B: about the
  same. **Counts: 26 of 56 consulted against 16.** The ten new passes include Christ the Redeemer → Brazil → 5,
  Senna → Brazil → 5, Le Corbusier → Switzerland → 4, Mars → 2 moons; several of them still have reach near 0 under
  either lens (Gandhi 0.015, Mars 0.003, Niemeyer 0.046), so the gate as written would hold them back.
- **Found, probe quality (from the A1 tables):** alone, the 4B's same-kind drop falls from 0.38 to 0.10 on median and
  the share of inconclusive alone prompts from 60% to 37%; the 27B's alone need in log-odds rises from 8.4 to 11.5
  with the controls unchanged near 0. J++ removes more of the bridge with no more collateral.
- **Found, the group result under J++ (27B neutral):** still needed at 1 / 2 / 3 peers 18 / 16 / 15 of 21 in
  log-odds (J-lens: 14 / 12 / 10 of 19); group-aware 16 / 7 / 7 of 21. The "weakens, does not stop" reading gets
  stronger with the better lens; the group-aware "stops" softens to "weakens" too.
- **Implication:** the lens is a tool choice that belongs before the candidate screen. Proposed: J++ as the primary
  lens for the screen and the Phase C runs on the Qwen models, J-lens kept as the replication on the development
  set; Llama and Gemma need a J++ fit first (cost to be checked). Decision pending.

## Round 15: Phase B built and tested on the tiny model (8 Nov, two forked agents in parallel)

- **Set:** the Phase B list in `plan.md`; every piece gets an exact test on the 4-layer tiny Qwen3.5 (two
  linear-attention and two full-attention layers, so the hybrid block can be tested for leaks).
- **Built:** `peerblock.py` with `Block` (attention mask on full-attention layers; mixer input zeroed and beta and
  log-decay set to 0 at the peer positions on linear-attention layers), `Patch` (the peer positions' residual from
  the "?" prompt written into the letter prompt), and `two_by_two`; `interventions.applied` takes Edit, Block and
  Patch in one list; ablate takes a strength for the dose-response sweep; `conformity.prompts` takes `extra=`
  variants: hidden "?", hidden "(answer submitted)", a second wrong letter G, told-to-copy, told-to-ignore, peer
  supplies the bridge (right and wrong); the own-part removal behind `--own-part`; the peer regex matches a whole
  answer line.
- **Found (tests, 38 passing):** the leak test is exact: with the block at every layer the last-position logits are
  invariant to the peers' letter; a block on only the full-attention layers leaks, and so does one on only the linear
  layers, so both paths are needed. The identity patch is a no-op; the "?"-into-"I" patch changes the output; a patch
  at the last layer only is a no-op, as it must be. The original eight conditions per question are byte-identical.
  " ?", " I" and " F" are each two tokens on Qwen (space, then the character), with equal lengths across variants, so
  positions line up for patching.
- **To watch on the real models:** run the leak test as the first step of each GPU run. transformers 5.22.1 may
  replace the GatedDeltaNet forward with a hub kernel when the `kernels` package is present, which would bypass the
  beta/g zeroing; if the leak test fails, set `USE_HUB_KERNELS=0` before loading. In bf16 compare the two prompts'
  logits after casting to float32, or expect bit-identical outputs since every non-peer input is identical.
- **Decided along the way:** `from_query` applies to full-attention layers only. Hidden prompts record P(peers') as
  NaN. `disagree_g` for Colosseum is S, a plausible alternative bridge's letter, because the existing wrong letter has
  the same property; the candidates module enforces the rule at drafting time instead.

## Round 16: GPU run 1, Qwen3.5-4B (9 Nov UTC, 4 pods, `results/v2_run1_qwen35_4b/summary.md`)

- **Set, before the run:** pass marks for the hidden check, block validation, 2x2 and patch fixed in `scope.md` §11
  and committed first (c290a5f). Selection rule unchanged (need >= 0.5, comes first). The confirmation half of the
  candidates screened like the rest, reported as counts only. Group steps on the 24 development questions with J++.
- **Ran:** leak test; trimmed solo screen of the 171 candidates under both lenses (342 cards); conformity with every
  B1 variant and the own-part removal, both wordings (1,064 prompts; 9 lost to CUDA out-of-memory errors, three
  questions re-run whole); block, 2x2, dose-response and patch on both wordings. One A100, then 2 H200 and another
  A100 for the group steps; GPU work 07:58–08:40 UTC.
- **Found, tools:** the leak test is exact on the real model (blocked logit difference 0.0, unblocked up to 7.75).
  All four block validation rows pass in both wordings: copy-only back to the no-peer level (4/4), give-ins back to
  the right letter (19/19 neutral, 58/64 group-aware), hidden prompts barely move (84/96 within 1 log-odds), told to
  copy falls by a median 8.4 log-odds (24/24). The answer-only block (full-attention layers, reply queries) catches
  0/24 told-to-copy: the peers reach the answer through the linear-attention layers. Patching "?" into the agree
  prompt agrees with the block (correlation 0.85–0.90; patch plus removal collapses exactly where D does, 39/40).
- **Found, the hidden check:** fails the pre-registered rule in both wordings (hidden "needed" 20–30% neutral, 0–20%
  group-aware, against 55% alone; the rule asked for about 41%). But no hidden prompt comes out "not needed" (0 of
  120 per wording); the shortfall is no-early-window and control-moved labels. In the neutral wording removal still
  drops P(right) by about 0.9 (alone 1.0), and B collapses the answer in 17–18 of 20 hidden prompts. The group-aware
  "?" prompt is not neutral: P(right) about 0.4, answer formed by layer 13 or earlier.
- **Found, the 2x2:** with three agreeing peers and neutral wording, only D collapses in 15 of 20 questions (the
  bridge is not needed while the peers are visible, and is needed again once they are blocked): the peers are the
  backup, the route is still there. Group-aware, 2–3 agreeing peers: blocking the peers alone drops P(right) by
  >= 2 log-odds in 17–18 of 20, so the model leans on the peers from the start. Agree versus hidden, three peers:
  the bridge's effect weakens by >= 2 log-odds in 10 of 13 assessable questions (neutral).
- **Found, the screen:** selected J-lens 86 of 171, J++ 81 (development 57 / 54 of 114). Letters: landmark 36 / 32
  of 40, university 23 / 21 of 28, person 10 / 8 of 30 (half fail knows-it). Months 7 / 9 of 14, players per side
  3 of 5, continent 4 / 5 of 12; element, planet, strings one each; driving side, starts-with-a-vowel and even/odd 0.
- **Found, variants:** told to ignore the peers, 11 of 20 still give in fully; a peer naming a wrong country is
  followed in 13 of 20 (neutral) and 20 of 20 (group-aware).
- **Changed:** for the 27B run, (1) the hidden check judged on assessed prompts and on hidden "not needed" plus B's
  collapse rate, with inconclusive and no-early-window counted apart (fix before the run); (2) wrong letters drawn
  from letters no same-kind word starts with, so the disagree D cell can be read (35 of 60 move to the peers' letter
  with the peers invisible, the alternative country's letter); (3) at most 6 processes per 80 GB and
  `expandable_segments` against out-of-memory errors; (4) driving side, vowel class, element and planet leave the
  removal arm on the 4B by the stop rule.

## Round 15b: fact pass on the candidates (9 Nov, separate session)

- **Set:** check every candidate's first hop, second hop, aliases, and whether the subject carries the bridge; check
  the same-kind controls against "same kind, different answer" and the run-time wrong peer (`same_kind[0]`) against
  "not a plausible alternative bridge's answer". `docs/conformity/candidate-fact-check.md`.
- **Found:** no bridge or answer is a wrong fact. 30 of 171 had an issue: 9 same-kind controls shared the bridge's
  letter (a dead control, not a biased one), 16 ambiguities (mostly the wrong peer being a plausible alternative,
  e.g. Kenya for Kilimanjaro, Cambridge for Hawking), 5 cosmetic.
- **Changed:** 9 controls replaced, 8 reordered, 4 questions dropped (Persepolis, National Taiwan University, Michael
  Jordan, Suomenlinna). Ids stay defined so the running screens keep theirs; `ACTIVE` and `split()` exclude the
  drops; the split now groups a bridge across families (Washington, Atlanta). 167 active, 112 development / 55
  confirmation, digest f35aa82d…. Two follow-ups for before any candidate enters a group run: the wrong-peer rule
  returns a *letter* for the number and category families (needs a per-format rule in `conformity.wrong_letter`),
  and the 4B screen's same-kind results for the 9 fixed controls were measured with the old controls.

---

## Pass rates so far (for forecasting the next draft)

| Family | Model | Drafted | Passed | Rate | Main failure |
|---|---|---|---|---|---|
| Letters, all three relations | Qwen3.6-27B | 37 | 25 | 0.68 | spread out / untestable |
| Letters, all three relations | Qwen3.5-4B | 37 | 24 | 0.65 | knows-it, spread out |
| Letters | Llama-3.1-8B | 37 | 15 | 0.41 | untestable (19), spread out (3) |
| Letters | Gemma-4-E4B base lens | 37 | 8 | 0.22 | onset gate at P ≈ 1 (13) |
| Letters | Gemma-4-E4B refit lens | 37 | 1 | 0.03 | untestable (24), spread out (12) |
| Close variants (mixed formats) | both Qwen | 38 | 17 / 15 | 0.45 / 0.39 | knows-it |
| Counts | Qwen3.6-27B | ~56 | 16 | ~0.29 | two-hop wrong; edit misses the bridge |
| Counts | Qwen3.5-4B | ~32 | 1 | 0.03 | knows-it |
| Creative ideas (4 sets) | Qwen3.5-4B | 44 | 0 usable | 0 | gate 1 (23 of 44) |
| Trend set | both Qwen | 23 | few | low | untestable |
| v2 candidates, letters (landmark / university / person) | Qwen3.5-4B, J-lens / J++ | 40 / 28 / 30 | 36 / 23 / 10 and 32 / 21 / 8 | 0.9 / 0.8 / 0.3 | knows-it (person) |
| v2 candidates, months | Qwen3.5-4B, J-lens / J++ | 14 | 7 / 9 | 0.5 / 0.6 | need below 0.5 |
| v2 candidates, other numbers (element, planet, strings, sports) | Qwen3.5-4B | 18 | 6 / 6 | 0.33 | knows-it |
| v2 candidates, continent / driving side / vowel class / even-odd | Qwen3.5-4B | 12 / 11 / 14 / 4 | 4–5 / 0 / 0 / 0 | low | need (overlap), knows-it |

## Lessons that now shape drafting

1. The second hop must be something the model can do on a word it never writes: first letter yes; last letter,
   syllables, letter counts no.
2. The bridge and the answer must be separable in the lens: arbitrary lookups, not capitals or languages.
3. Copying one passing shape makes many questions on one fact; distinct second-hop facts are a drafting rule.
4. Country bridges often fail reach on the 4B; cities pass more; the 27B is kinder to both.
5. Single-token check before anything else: it removed a whole family and a quarter of the letter drafts at no cost.
6. Pass rates differ by model more than by family; forecast per model.
7. The lens is part of the probe: J++ passes ten more count questions and halves the 4B's collateral; fix the lens before a screen, not after.

## Template for the next entry

    ## Round N: <name> (<date>, <model>)
    - **Set, before the run:** <rule, thresholds, split>
    - **Ran:** <what, how many, cost>
    - **Found:** <pass counts by family and why the rest failed; the result>
    - **Changed:** <what the next round does differently>
