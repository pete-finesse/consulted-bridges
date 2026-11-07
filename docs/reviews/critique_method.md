# Critique of the bridge scorecard and its follow-up experiments

*Independent review. I read `docs/scorecard/v3_v4.md` (from "Round 3" on), `NEXT_EXPERIMENTS.md`, `docs/reviews/review_scorecard.md`, `SUMMARY.md`, `DESIGN.md`, and the code (`src/consulted_bridges/scorecard.py`, `giveup_remove.py`, `giveup_common.py`, `bank_build.py`, `onset27b.py`, `src/consulted_bridges/interventions.py`). I rebuilt the question bank and recomputed numbers from the saved CSVs. I used no GPU and changed no project file. My throwaway scripts are in the session scratchpad (`a1.py`–`a5.py`).*

## Terms used below

- **Residual stream:** the long list of numbers the model carries at each word position from one layer to the next. Every edit here changes it.
- **J-lens direction of a word:** a fixed direction in that list, one per layer. How far the residual points along it is the J-lens's "reading" of the word at that layer.
- **Remove (ablate):** delete the residual's component along a word's direction, at every position from the question's first word to the end of the prompt, in a block of layers.
- **Window:** that block of layers. The 4B model uses 6 layers, stepping by 3. The 27B model uses 12 layers, stepping by 6.
- **Drop:** how much the right answer's probability falls after an edit. 1 means it's gone.
- **Bridge:** the hidden middle step (Italy in "the letter the name of the Colosseum's country starts with").
- **Same-kind control:** another word of the bridge's kind (Spain, France for Italy).
- **Everyday words / noise floor:** words unrelated to the question (table, river…). The largest drop any of them causes is the "noise floor".
- **Answer onset:** the first window from which removing the answer's own direction kills the answer. Before it, the answer is supposedly not yet formed.
- **Swap:** replace the bridge's component with another word's (Italy → Spain). If the model really reads the country from that direction, it should then answer "S".
- **Cosine:** how much two directions overlap. 1 means identical, 0 means unrelated.
- **Log-odds shift (experiment 1):** the change in log P(peers' letter) − log P(right letter). Positive means the edit pushed the agent toward the peers.

## 1. Verdict

1. The goal (find questions where the model provably uses the bridge) is a reasonable first step toward a causal test of "silent dissent". But as run, it no longer tests the paper. The paper's own city → country facts all fail, and the survivors are letter-of-a-hidden-word questions, where using the bridge is close to guaranteed by design.
2. The core effect is large but narrow in meaning. Removing one J-lens direction kills the answer (drop 0.95–0.99, noise ≤ 0.05) in about 6 questions per model. That shows the direction is *necessary*. It does not show that the direction *carries the bridge*: in the 27B model, only 2 of 37 swaps on the five letter questions make the model give the swapped word's letter, and 5 "consulted" cards fail the reach gate.
3. Specificity and isolation mostly measure how the directions overlap, not what the model does. For letter questions both are about 1 almost automatically, so quality ≈ need.
4. The criteria were changed about 8 times after seeing data, and every change made it easier to pass. Three headline cases pass only under the latest rules: Piaf in the 27B, and hockey and spider legs in the 4B.
5. Experiment 1's solid result is a null. When the agent fully gives in, removing the bridge does no more than removing another city (3 questions). Reading the partial give-ins as "silent dissent, shown causally" rests on 2 coin-flip prompts and just restates the single-question result.
6. Before building on this:
   - freeze the rules;
   - add random-direction controls of matched size, a dose-response check, the logit lens and activation patching;
   - measure the paper's own presence metric in the same prompts;
   - run on a fresh, pre-registered question set.

## 2. Issues, ranked by how much they could change a conclusion

### C1 (critical). The scorecard ends up testing a different task from the paper's

**What.** The paper claims that agents who give in still *represent* the bridge. Its evidence is presence: the bridge is in the J-lens's top 100 words. Its facts are city → country ("Mozart's birth city is in the country of…").
- **The paper's facts don't survive the scorecard.** Mozart's country, York's country, Gates's country and Mishima's country are either "untestable" (the edit can't reach the bridge) or "answer echo" (isolation 0.00–0.18; bank `results.csv`).
- **What survives is a different family:** the first *letter* of a hidden word, plus spider legs and hockey in the 4B.
- **In a letter question the answer is a spelling of the bridge.** To say "I", the model has to have "Italy" as a word, because spelling is a property of the word itself. Since a direct "Colosseum → I" shortcut is implausible, using the bridge is close to guaranteed whenever the model answers right. The scorecard confirming it is a *positive control* passing, not a discovery.
- **`docs/proposals/similar_questions.md` says as much.** It picks new letter questions *because* "country letters don't overlap country directions (isolation ≈ 1)".

**Why it matters.**
- **Nothing here bears on the paper's numbers.** Its 0.85, 0.22, 0.24 and 0.03 are for city → country facts. In those, the bridge and the answer are tied by meaning (Salzburg is Austrian), and that tie is exactly what the paper's critics (us) say inflates presence.
- **Copying is much easier in the letter family.** A single letter in the prompt is easier to copy than a country name, so the group results may not carry over either.
- **Presence is never measured.** The paper's claim is about presence, but no presence readout exists yet in any group prompt (experiment 1c is not done). So the work doesn't yet touch the paper's own measure.

**What it would take.**
- **To refute the paper's reading:** in the paper's format and fact type (or a close variant), show agents who give in where the bridge is present by the paper's metric (top-100 J-lens rank in its layer band), but has no bridge-specific causal effect beyond matched controls. Then show the same presence in conditions with no dissent at all: peers agree, or the agent never answered. SUMMARY already shows the second part. The first needs 1c in the same prompts.
- **To support it:** in agents who clearly give in (say P(right) < 0.05), removing the bridge must shift the balance toward the peers more than same-kind and random controls do, and strengthening it must bring back the right answer more than controls do. That has to hold across a few dozen questions, not 3.
- **Practical route:** keep letters as the *validation* family for the method. Then deliberately build a bridge family closer to the paper, where the answer is a property of the bridge but not its spelling (e.g. hockey → Six, Thanksgiving → November → 30), and test the city → country facts with a lens-free edit (patching, below).

### C2 (critical). Removing a J-lens direction shows that direction is necessary, not that it carries the bridge

**Evidence.**
- **Swaps almost never plant the swapped word.**

  | Model | Question | Swaps that gave the swapped word's letter or count |
  |---|---|---|
  | 27B | Colosseum | 0 of 6 (all still "I") |
  | 27B | Mozart | 0 of 8 |
  | 27B | York | 0 of 8 (all still "T") |
  | 27B | Gates | 1 of 8 |
  | 27B | Piaf | 1 of 7 |
  | 27B | Spider legs | 0 of 7 (all still "Eight") |
  | 4B | Colosseum | 3 of 6 |
  | 4B | Gates | 1 of 8 |
  | 4B | Mozart | 1 of 8 |
  | 4B | York | 0 of 8 |
  | 4B | Hockey | 0 of 6 |

  Source: `swaps.csv` in each results folder. These are "reported, not in the verdict", and none of the write-ups discuss them.
- **The reach gate fails on 5 "consulted" cards.** Reach is the drop in P(bridge name) when the model is asked to name the bridge first. It's 0.10 for 4B spider legs (a flagship case, quality 0.95). In the 27B it's 0.14 for the spider letter, 0.00 for Thanksgiving days, 0.24 for gold and 0.00 for the Thanksgiving month (`cards.csv`, column `reach`).
- **Adding along the direction breaks the model.** Steering and amplifying (experiment 2) break the 27B even at small strengths. Amplifying *everyday* words already pushes the agent toward its peers.

**Reading.** Deleting the direction kills the answer. Writing another word into it doesn't change the answer, and adding to it breaks the model. That's the classic pattern of a direction the downstream computation is *sensitive* to, but which is not the bridge's *content*. Two other explanations remain open:
- the removed direction is something the answer needs that travels with the bridge (e.g. "the model is about to name a country");
- removing a large component makes the residual unusual for the model, and later layers fail on it (off-distribution damage).

**What argues against generic damage.**
- Everyday words do ≤ 0.05 on the top cards.
- Unrelated questions are mostly untouched (≤ 0.07).

But:
- **Two consulted cards do damage unrelated questions:** 27B Mozart (0.21; its need is only 0.56) and Piaf (0.14).
- **Collateral was dropped from the verdict in round 3.** Yet the verdict text still says "no collateral" without checking it (`bridge_card4.py:423`; the docstring at `:44` still lists it as required).
- **The everyday words are not real size controls.** They are absent concepts, so their component in the residual is small. The bridge's strength isn't saved, so the "matched on strength" step can't be audited. The selected everyday words have strength 0.07–0.11 for *every* question (`controls.csv`), which suggests they aren't tracking the bridge.

**What would resolve it.**
1. **Random directions of matched size:** remove ≥ 20 random directions, each scaled so the removed component has the same size as the bridge's, in the same windows. Need should be measured against *their* 95th percentile.
2. **Dose-response:** remove 25/50/75/100% of the bridge's component. A graded fall supports a real effect. A cliff only at 100% suggests damage.
3. **Activation patching:** this test doesn't use a lens. Run "…the Colosseum…" and "…the Sagrada Família…" (Spain), and copy the residual at the final positions, layers 30–53, from the Spain prompt into the Italy prompt. If the answer becomes "S", the bridge's information sits there and is enough to set the answer. That is the sufficiency test the swaps were meant to be.
4. **The logit lens as a baseline:** the same removal with plain output-word directions. If it works as well, the J-lens adds nothing to the edit.

### C3 (major). Specificity and isolation mostly measure how the directions overlap

**Isolation.** Isolation is removing "the bridge minus the answer" (the part of the bridge's direction not shared with the answer's), divided by need.
- **The geometry.** Suppose the residual's bridge content lies along the bridge direction, and the bridge and answer directions have cosine c. Then removing "bridge minus answer" deletes a share 1 − c² of the bridge's reading.
- **What that means for letters.** For Italy and "I", c is around chance (~0.2). So the edit deletes about 96% of what removing Italy deletes: it's nearly the same edit. Isolation ≈ 1 is guaranteed by the geometry, whatever the model computes.
- **Where it can be low.** Only when the directions overlap a lot (Toronto/Canada, Salzburg/Austria). There, as the v1 review said (M4), "low isolation" can't tell an echo apart from a bridge consulted *through* its country attribute.
- **So two claims don't follow.** "The letter questions fix isolation completely… the shortcut fix for the paper's own facts" (V3, line 62) doesn't follow. Neither does the label "Answer echo" in `bank_build.py:52`.

**Strict specificity.** Strict specificity is removing the bridge minus all its same-kind controls, divided by need. In `spec_check2`:

| Question | Remove the controls together | Remove the bridge minus all controls |
|---|---|---|
| Colosseum | 0.97 | 0.99 |
| Mozart | 0.71 | 0.95 |
| Gates | 0.72 | 0.89 |
| York | 0.89 | 0.99 |
| Spider legs | 0.08 | 0.99 |

- **For place bridges, both edits kill the answer.** With 0.66 of Italy's direction outside the controls, each edit removes very roughly half of Italy's reading.
- **So the test can't say which half matters.** The fair reading is "you can't remove a large part of this direction without breaking the answer". It is not "the model uses something specific to Italy" (V3, line 122).
- **Swaps point the same way.** In the 27B, putting Spain into Italy's place leaves "I". That is evidence *against* the model reading an Italy-specific component from this direction.
- **Spider is the only clean contrast** (0.08 vs 0.99).

**Quality.** With isolation and specificity about 1 for letter questions, quality ≈ need, which is the drop minus the noise floor. The ranking among the top cards then mostly reflects how noisy each prompt is. Mozart 4B scores 0.66 because its noise floor is 0.32, not because its bridge is weaker.

**What would resolve it.**
- Report cosines next to isolation and specificity, and compare each against the value the geometry alone predicts.
- Judge specificity by swaps or patching (does Spain's information make the model say "S"?), not by projections.
- Score a same-kind control with the *same* answer (another country starting with "I": Iceland, India, Iran, Ireland) as the real category test. If removing Italy hurts and removing India doesn't, that's specificity. If swapping Italy for India keeps "I" while Spain gives "S", that's strong evidence.

### C4 (major). Garden of forking paths: the rules moved after the data, always in the bridge's favour

"Garden of forking paths" means that choosing analysis rules after seeing the data lets the results drift toward what you hoped for, even with no bad intent. Changes I could find:

| # | Change | When / trigger | Direction | Case it decided |
|---|---|---|---|---|
| 1 | Reach gate moved into the same prompt, and **low reach now only blocks negative verdicts** | Round 3, "Spider showed why" | Looser | 4B spider legs (reach 0.10) keeps "consulted" |
| 2 | Collateral damage made "reported only" | Round 3 | Looser | 27B Mozart (0.21) and Piaf (0.14) stay consulted |
| 3 | Specificity redefined twice: 1 − control/bridge, then bridge minus worst control, then bridge minus all controls | Round 3, after letters scored 0.13–0.26 | Looser | Letter family 0.13 → 1.00 |
| 4 | Answer onset from the answer's own part (minus the bridge) | After Mozart's country | Later onset, so looser | Mozart, Gates, CN Tower and the letters re-scored |
| 5 | Onset = first of two consecutive windows | Round 5, after 27B spider | Later onset | — |
| 6 | Raw answer tokens ("S" without a space) added | Round 5, after 15 questions failed gate 3 | More testable | 6 became consulted (a reasonable bug fix) |
| 7 | Onset = sustained to the last window ("v4.1") | **After 27B Piaf** | Later onset, though the doc calls it "stricter" | **Piaf 27B (0.96)** |
| 8 | A second verdict rule in `bank_build.py` (isolation < 0.5 → echo; specificity < 0.5 → category), next to the card's own rule, which still uses the old specificity | Nov 6 | Mixed | 4B Colosseum: "category effect" in `cards.csv`, "consulted" in the bank |

**Recomputed from the saved window profiles** (`a4.py`, `a5.py`), testing "needed early beats noise and comes first" under each onset rule:
- **Piaf 27B** passes only under the sustained rule. Under "first window that kills" or "two consecutive windows", using its own-part profile (0.53 and 0.64 at layers 24–41), it fails (need ≈ 0).
- **Hockey 4B** fails under the "first window that kills" rule: removing the answer's own part already kills "Six" by 0.78 in layers 15–20, and need before that is 0.12. It passes under the other two rules.
- **Colosseum, Gates, Mozart and York** (4B and 27B) pass under all six definitions (whole or own part × three onset rules). Those are the robust results.
- **The thresholds** (0.5, 0.3, a 3-layer lead) are still the v1 values. Those were set on spider and Mozart, the two cards they then classified (v1 review M7), and were never re-derived.

**Fair reading.** Each change had a plausible reason, and the strongest letter results don't depend on them. But one-way drift is the signature of tuning. And the doc presents the outputs as fixed criteria ("same criteria and thresholds as before"). See section 3 for what to freeze.

### C5 (major). The answer onset is late almost by definition, so "comes first" and "early windows" constrain little

- **Where the onset lands.** The onset is found by removing the answer's *output-word* direction ("I", "S", "Eight"), and that direction mostly matters only where the word is about to be written. Onsets in the 27B: 19 of 51 at layer 54 (the last window), 43 of 51 at layer 42 or later (`onset27b_qwen36_27b.csv`). In the 4B, 28 of 56 are at layer 21 or later.
- **Consequence for the tests.** "Early windows" is nearly every window, and "comes first" reduces to "the bridge matters somewhere before the final window".
- **In the 27B, 5 of the 12 passes have their whole bridge effect in layers 42–53.** That's the window right before the onset. For example, removing Italy over layers 16–47 drops "I" by only 0.03, while layers 42–53 drop it by 0.99. The "lead" of 12 layers is just the window grid.
- **What the tests can't tell apart:** "the model works out Italy as a middle step and then uses it" versus "the model forms the answer in some other form and only turns it into a word at the end". An answer can be formed long before its output-word direction becomes necessary.
- **The goal's "separate from the question's words" has no test.** The edit runs from the question's first word onward, so it also hits the subject word ("Colosseum"), which may carry Italy as an attribute. v1's test T (after the question vs on it) was dropped and never replaced.

**What would resolve it.**
- Find the onset with a lens-free method: patch the final positions from a prompt with a different answer, layer by layer, and see from which layer the answer follows.
- Report the edit restricted to (a) the subject word's positions only and (b) the positions after the question only.

### C6 (major). Experiment 1: the conclusions go further than the data

**Data:** `results/giveup_remove_qwen36_27b/early_summary.csv`, plus my recomputation from `windows.csv`.

1. **"When the agent holds, its bridge is what holds it."** True, but expected. It's the single-question result in a new prompt.
2. **"Gives in partly → silent dissent, shown causally."**
   - **Only 2 prompts:** Mozart with 3 peers (P(S) = P(B) = 0.47, a tie broken by greedy decoding) and spider with 3 peers (P(right) 0.36 vs P(peers) 0.59).
   - **These agents are torn, not giving in.** If the bridge is where P(right) comes from (the single-question result), removing it must lower P(right) whatever the peers say. That restates the single-question result. It doesn't show something *pushing back against the agent's answer*.
   - **The category is post hoc.** "Partly" was defined after seeing the probabilities.
   - **Spider's reach gate fails alone** (0.14). And Mozart's same-kind control (Vienna) gives a shift of 5.0, against 8.7 for the bridge.
3. **"Gives in fully: small leftover effect, close to same-kind controls."** This is the result that matters for the paper, and it is a **null**:

   | Prompt | Bridge | Same-kind | Everyday words |
   |---|---|---|---|
   | Gates, 3 peers | 1.50 | 1.50 | 0.75 |
   | Gates, 2 peers | 2.62 | 2.25 | — |
   | York, 2 peers | 2.37 | 2.25 | — |
   | York, 3 peers | 2.75 | 2.50 | — |
   | Piaf, 2 peers | 1.81 | 0.87 | — |
   | Piaf, 3 peers | 1.81 | 1.00 | — |

   *Values are log-odds shifts.*
   - **In probabilities, the changes are tiny.** P(right) goes from about 0.002–0.03 to about 0.001–0.003. The log-odds scale magnifies changes in the far tail.
   - **Some of the size is rounding.** Every shift is a multiple of 0.125 because the model runs in bf16 (a 16-bit number format), so shifts under ~0.5 are a few rounding steps.
   - **Epsilon isn't the problem.** Changing the small constant (epsilon) from 1e-9 to 1e-4 barely moves the results.
4. **The borrowed onset (layer 54).** It comes from the Colosseum question asked alone and is applied to all 18 group prompts.
   - In the full give-ins, removing the *right* letter in the same windows shifts the balance by up to 1.56. That's the same range as the bridge (1.5–2.75), so the answer check doesn't separate them there (1e says so).
   - More importantly, the copied answer is the *peers'* letter. Nobody removed the peers' letter's direction, which is the obvious check on the "copy route".
5. **The analysis can't be rerun from code.** `early_summary.csv` was made by no saved script. The scripted `summary.csv` picks the window over *all* windows, and that is layers 54–63 (the answer-writing window) for 12 of 18 prompts. The "early" restriction was added afterward.
6. **Sample.**
   - 6 questions, all letters, 1–3 peers, one wording.
   - The prompt tells the agent to defer ("Taking the other agents' answers into account").
   - The 18 prompts aren't independent: the real sample is 6 questions, of which 3 give in fully.
7. **"Present but unused" vs "not computed" can't be told apart yet.** No presence readout was taken (1c).

**What I'd conclude:** in this letter setup, there's no evidence that the bridge does bridge-specific work once the agent gives in fully. That's 3 questions, measured in the far tail, with no presence readout. This is weak evidence *against* the paper's interpretation, not *for* "silent dissent".

### C7 (moderate). Group pilot: "the model copies" is one of several readings

- When peers agree, need falls to 0.01–0.15 in 5 of 6 questions. But P(right) is 0.95–0.99 there (a ceiling), and two routes that can each produce the answer would also make the bridge unnecessary. `NEXT_EXPERIMENTS.md` 1b already says this. The V3 write-up (line 297) still says "the model copies the peers' letter".
- Spider's "agree" reach is 0.26. Mozart's single-question baseline has need 0.56 with collateral 0.21, so "stops using the bridge" there starts from a weak baseline.

### C8 (moderate). Small, selected, single-wording sample; no test of whether the method can say "no"

- **Few positives.** About 6–7 positive questions per model (bank: 7 "consulted" in the 4B, 6 in the 27B), each in one wording.
- **The letter family isn't consistent across models:**
  - York is 0.91 in the 4B and 0.40 in the 27B (a category effect);
  - Federer → Basel → B is a bystander in the 4B and spread out in the 27B;
  - Maldini is spread out in the 27B.
- **A selection effect, like the paper's.** The bank keeps the ~7–10% of questions where the edit happens to work. Group results on the bank describe that subset, just as the paper's 0.85 described 18 cities.
- **The method's ability to say "no" is untested.** 15 of the 16 deliberately bad questions were stopped by a gate before scoring. The only one scored (banana) isn't clearly bad.
- **No uncertainty.** There are no error bars and no paraphrase variance. Every number is one greedy run of one prompt.

### C9 (minor). Code and record-keeping

- **`bridge_card4.py:423`:** the "consulted" reason says "no collateral", but collateral isn't checked. The docstring (lines 43–49) still lists collateral and old specificity as verdict criteria.
- **The card's verdict still uses the old specificity** (`beats_kind`, line 422), while quality uses the strict one. V3 (line 237) says the labels "should be switched"; that was only patched in `bank_build.py`.
- **Quality is reported for untestable cards** (27B Mishima letter 0.47, which fails gate 4).
- **The bridge's strength isn't saved,** so the matching of everyday words can't be audited.
- **The bank mixes runs.** It combines v3 runs (`big_*`, with the whole-direction onset) and v4 re-scores in the 27B. Piaf's stored numbers are from v3.
- **The experiment-1 code isn't on v4.** `giveup_common.py` imports `Card` from `bridge_card3`, not `bridge_card4`.
- **Repeated maxima.** "Best of 8–10 windows" is used everywhere. For need against noise, it's conservative (the noise floor is a max over 12 words × windows). But reach, own part and the specificity numerators are each a max over windows, which biases them upward. That matters only near thresholds (Mozart 27B need 0.565; spider 27B 0.512).

### What holds up well

- **The size of the core effect is not a statistical accident.** On the top cards, removing the bridge gives a drop of 0.95–0.99, against 0.00–0.05 for everyday words and ≤ 0.07 for unrelated questions. Taking the best of 10 windows can't produce that.
- **The bookkeeping caught real problems** (the overlap artefact in the onset, the raw-token answers), and the write-ups record changes honestly.
- **Spider vs place bridges** in strict specificity (0.08 vs 0.71–0.97) is a genuine, interpretable contrast.

## 3. Status of the v1 review's points

| v1 point | Status |
|---|---|
| B1 broken token pieces | **Fixed** (gate 2) |
| B2 picks not known | **Fixed** (gate 1 checks both hops) |
| B3 wrong USA direction | **Fixed** (answer forms plus raw tokens) |
| B4 inconsistent word sets | **Fixed** |
| B5 expected answers not the model's | **Fixed** (belief check per swap) |
| B7 prefix matching | **Fixed** (whole-word matching; raw text saved) |
| M1 can the edit see the bridge? | **Partly.** Gate 4 was added, then made one-sided (C2, C4) |
| M2 late windows | **Fixed in form** (needed early). But the onset is late, so the restriction is weak (C5) |
| M3 spread out | **Fixed** (reported, and a verdict) |
| M4 isolation can't separate echo from attribute use | **Not fixed.** "Answer echo" is still the label (C3) |
| M5 on-topic controls | **Fixed** (fixed pool of everyday words) |
| M6 the after-vs-question test | **Dropped,** with no replacement for "separate from the question's words" (C5) |
| M7 thresholds tuned on the two examples | **Not fixed** (C4) |
| M8 small sample | **Partly.** More questions, but still ~6 positives (C8) |
| X collateral as a gate | **Reverted** to reported only (C2) |
| Swap validity before reading swaps | **Not acted on.** Swap validity is ~0 on the 27B tops and was never discussed (C2) |
| Dose-response | **Not done** |

## 4. What to pre-register before the next run

Write these down, commit them, and only then run. "Commit the script before the run" is the practical test.

1. **The question set.** Use the 38 proposed variants (`docs/proposals/similar_questions.md`). Split them at random *now* into a calibration half and a test half, and report the test half only after the rules are frozen on the calibration half.
2. **One set of rules, used everywhere.** That means one verdict function in one file, imported by the card, the bank and the experiments. Fix:
   - the onset rule (I'd report both "first window that kills" and "sustained", and call a question robust only if both agree);
   - the window size and step;
   - the positions;
   - the answer forms;
   - every threshold.
3. **A symmetric reach gate.** A positive result with reach < 0.5 is labelled "direction necessary; bridge reach unverified", not "consulted".
4. **Collateral as a gate:** ≤ 0.1 on ≥ 5 unrelated questions, not 2.
5. **Required controls, with a stated pass rule:**
   - random directions of matched size (need must beat their 95th percentile);
   - dose-response (a graded fall);
   - logit-lens removal (reported);
   - activation patching from a different-answer prompt (the answer must follow the patch in the bridge's layers);
   - a same-kind control that shares the answer (India for Italy) and one that doesn't (Spain).
6. **Paraphrases:** 3 wordings per question. A question counts only if its verdict holds in at least 2 of the 3.
7. **Known negatives that pass the gates.** At least 5 designed shortcut or echo questions that the model answers right and that the edit reaches. Write down beforehand that they must score below a set quality.
8. **Experiment 1 / 1c / 1d:**
   - Define "holds", "partly" and "fully" by P(right) before running (e.g. fully = P(right) < 0.05).
   - Fix one primary outcome: the bridge's log-odds shift minus the larger of the same-kind and random-direction shifts. Report the absolute ΔP next to it.
   - Measure the onset in each prompt.
   - Remove the peers' letter as an extra arm.
   - Run in float32 for the tail probabilities.
   - Take the paper's presence readout in the same prompts.
   - Aim for ≥ 20 questions with full give-ins (use more peers or the paper's hidden-answer setup to get them).
   - Count the sample as questions, not prompts.
9. **Decision rules, written in advance.** For example: "We call it silent dissent if, among full give-ins, the bridge-specific shift is > X in ≥ Y% of questions. We call it present-but-unused if the paper's presence score is ≥ Z while the bridge-specific shift is within the controls' 95% range."

## 5. Current conclusions: keep, weaken or drop

| Conclusion (where) | Call | Why |
|---|---|---|
| "The tool works: it can show whether the model uses a hidden middle step" (NEXT_EXPERIMENTS) | **Weaken** | "Removing the bridge's J-lens direction is necessary for the answer in ~6 letter questions." Not shown that the direction carries the bridge (C2). |
| Colosseum, Gates, Mozart letters are consulted bridges in both models | **Keep, reworded** | Robust to every onset rule. Reword as "direction necessary, comes before the output word", pending patching and random-direction controls. |
| Piaf 27B 0.96, hockey 4B 0.74 | **Weaken** | Each depends on one onset rule (C4). Hockey is a bystander in the 27B. |
| Spider legs 4B 0.95, "best case on every factor" | **Weaken** | Reach 0.10; untestable in the 27B (C2). |
| "Letter questions fix isolation completely; the shortcut fix for the paper's facts" (V3 line 62) | **Drop** | Isolation ≈ 1 follows from the geometry, and it doesn't touch the paper's facts (C1, C3). |
| "The model uses something specific to Italy, not just a country" (V3 line 122) | **Drop** | Removing the controls together kills the answer too; swaps don't plant Spain (C3). |
| "Country answers stay echoes in both models" | **Weaken** | "The effect runs through the part shared with the answer": echo vs attribute use is unresolved (C3). |
| "Letter-of-the-hidden-word is the family that holds across models" | **Weaken** | 3 of 6 questions agree. York and Federer don't (C8). |
| "Bad questions were caught" | **Weaken** | Caught by gates, not by scores. The scores' power to say "no" is untested (C8). |
| Group pilot: with agreeing peers the model copies | **Weaken** | "The bridge stops being necessary" (as in 1b). |
| Exp 1: when the agent holds, its bridge holds it | **Keep** | Expected, but correct. |
| Exp 1: partial give-in = "silent dissent, shown causally" | **Drop the label** | Keep as "where P(right) is still ~0.4, it depends on the bridge" (2 prompts). |
| Exp 1: full give-in, no bridge-specific effect | **Keep, and make it the headline** | With caveats: 3 questions, far-tail measure, no presence readout, borrowed onset. |
| "General damage ruled out" | **Keep, with exceptions** | Everyday words and unrelated questions are mostly clean. But 27B Mozart and Piaf exceed the collateral limit, and there's no random-direction control of matched size. |
