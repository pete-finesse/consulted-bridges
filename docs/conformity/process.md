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

## Lessons that now shape drafting

1. The second hop must be something the model can do on a word it never writes: first letter yes; last letter,
   syllables, letter counts no.
2. The bridge and the answer must be separable in the lens: arbitrary lookups, not capitals or languages.
3. Copying one passing shape makes many questions on one fact; distinct second-hop facts are a drafting rule.
4. Country bridges often fail reach on the 4B; cities pass more; the 27B is kinder to both.
5. Single-token check before anything else: it removed a whole family and a quarter of the letter drafts at no cost.
6. Pass rates differ by model more than by family; forecast per model.

## Template for the next entry

    ## Round N: <name> (<date>, <model>)
    - **Set, before the run:** <rule, thresholds, split>
    - **Ran:** <what, how many, cost>
    - **Found:** <pass counts by family and why the rest failed; the result>
    - **Changed:** <what the next round does differently>
