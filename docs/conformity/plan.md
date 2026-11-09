# Conformity, next steps: one plan from three

*Written 8 November 2026. Combines the v2 plan ("does the answer come from the model's own step, or from the peers?"), the broader research plan drafted the same day (audit, frozen benchmark, peer localisation, real groups), the narrow "Bridge Under Conformity" proposal, and NEXT_EXPERIMENTS 1b–1f. Ordered by value per unit of cost. Each GPU run needs Peter's yes first. Scope decisions and their justifications live in `scope.md`, the method as currently run in `methods.md`, the round-by-round process in `process.md`; the candidate questions in `src/consulted_bridges/questions/proposed/candidates.py`.*

---

## 1. Where the pieces stand

| Piece | Status | What survives into this plan |
|---|---|---|
| Narrow proposal (`bridge-conformity-proposal.html`) | **Executed.** The conformity runs (65 questions, two Qwen models, 934 prompts, plus Llama and Gemma) are what it proposed. | It is now the write-up. One open item: the made-up-subject control for Gemma. |
| v2 plan | Runnable on the current `conformity.py`. | Almost all of it: steps 1, 2, 3, 4, 5, 6, 7, 8 below. |
| Broad research plan, stage 1 (audit) | Half done: intervals by question and threshold sensitivity were committed (`conformity_stats.py`). | The interpretive fixes: ceiling, nonspecific damage, explicit denominators. Merged into Phase A. |
| Broad plan, stages 3–4 | Same experiments as v2 steps 3–6, stated more cautiously. | The cautions: a block can miss information already copied elsewhere; report every presampled case; a null is inconclusive unless the tool passes in that prompt. |
| Broad plan, stages 2 and 5 | New territory: an 80-case frozen benchmark with private-evidence tasks, real interacting groups, distributed evidence, transfer. | Phase D, as its own project. One small hybrid (peer supplies the bridge) is cheap enough to run earlier. |
| NEXT_EXPERIMENTS 1b | Not done. | Is v2 steps 3a and 5. |
| NEXT_EXPERIMENTS 1c (presence), 1e (per-prompt onset) | **Done** inside the conformity run (`presence_rank`, `onset` per prompt). | Retire. |
| NEXT_EXPERIMENTS 1d (group size 1–12, mixed groups) | Not done. | Optional add-on to Phase C (same pipeline). |
| NEXT_EXPERIMENTS 1f (swap, patching) | Swap mostly fails (10 of 240); patching open. | Swap parked. Patching returns as step 7 in a stronger form. |

**One real disagreement.** The broad plan says a question should not need to pass the solo scorecard to count. Keep the selection for the mechanistic work (a probe needs a bridge that is demonstrably used alone), but report how many candidate questions were screened out. The bank screens already record that.

---

## 2. What is most at risk, and why Phase A comes first

The quotable claim is "when peers agree, the model stops using its bridge". A check of the saved agree prompts (8 Nov) shows two problems the plans predicted:

| Run | Agree prompts not labelled "needed" | P(right) still > 0.9 after removal | Median log-odds fall from removal | Bridge drop > 0.3 with noise drop > 0.2 |
|---|---|---|---|---|
| Qwen3.6-27B neutral | 42 | 22 | 2.3 | 0 |
| Qwen3.6-27B group-aware | 61 | 40 | 1.0 | 0 |
| Qwen3.5-4B neutral | 40 | 0 | 1.5 | 8 |

- **Ceiling (27B).** Half the "not needed" verdicts leave the answer at 0.9 or better, yet the log-odds fall by about 2. That reads as "used less", not "not used". The saved `bridge_dLO` is 0 on every 27B agree row, because LO compares P(peers) with P(right) and those are the same letter there. The re-analysis must use the log-odds of P(right) instead.
- **Nonspecific damage (4B).** Up to 8 agree prompts have a large bridge drop and a large everyday-word drop too. Those are "inconclusive", not "not needed".
- **Presence is uninformative as a discriminator.** The bridge is present (J-lens rank < 100) in 100% of alone and agree prompts on both Qwen models. That already settles 1b's first question and 1c: "present but unused" stands, and presence cannot sort prompts.

Both fixes are free and may change the headline, so they come before any GPU spend.

**Done 8 Nov (A1, `results/conformity_stats_v2/`, `process.md` round 13).** It did change the headline on the 27B neutral wording: 10 of 19 still need the bridge with three agreeing peers in log-odds (5 of 19 by probability); the group-aware wording and the 4B keep the old reading. The 4B's same-kind control moves even alone, so its specificity is a probe limit, and Llama's controls start moving when peers appear.

---

## 3. Question design and selection

Everything downstream depends on this, so it gets its own rules, fixed before any new run.

### 3.1 What the gate is, and why it is defensible

The removal probe only answers questions where the model demonstrably uses an intermediate step when no peers are present. A question that fails that cannot support the *null* reading in groups ("peers made the bridge unnecessary"), because "the tool never worked here" predicts the same nothing. It can still support a positive reading (the group recruits the bridge), and it can enter every behavioural measure (F versus G peers) without any gate.

So the gate is on the **tool**, not on the outcome. It asks whether the model uses the bridge alone; the hypothesis is about what happens when peers appear, and a gated question is free to go either way (79% still need the bridge with one agreeing peer, 26% with three). What the gate costs is population: the findings are about questions where the model has a demonstrable intermediate step. That is also Silent Dissent's population: an agent cannot dissent silently if it never reasoned. State this in one paragraph and report the funnel (candidates drafted, passing each screen, selected).

**The gate itself, minimal.** Not the full scorecard. Four checks, each prespecified:
1. knows both hops alone (hop 1 names the bridge, hop 2 gives the answer from the bridge, the chain gives the answer);
2. bridge, answer, peers' answers and control words are one token each; no alias changes the answer (United States / America);
3. **need** in the solo prompt: largest early bridge drop minus largest early everyday-word drop ≥ 0.5, with the same-kind drop below the bridge drop; this is a removal against a noise floor, the plainest measurement the scorecard has;
4. **reach**: the edit demonstrably hits the bridge in this prompt.

Dropped from selection: isolation, specificity, quality, swap. They served the identity claim and the swap mostly fails.

### 3.2 Selection bias and the baseline

Selecting on solo need picks up some questions that passed by luck. Re-measured in any new prompt they look weaker on average, even if peers do nothing, so "alone versus agree" overstates the shrinkage. Unselected questions have the opposite problem: where the probe never worked, removal shows nothing and reads as "bridge never needed", understating it.

Rule: **gate on one prompt, compare two others.** Gate on the solo prompt. Measure the effect in the hidden-answer group prompt (peers write "?") and in the agreeing-peers group prompt, and compare those two. Both are re-measurements the selection never saw, so luck washes out of both equally. "Shrinks" means agree against hidden, not against alone. For the confirmation set, the gate uses one paraphrase and the comparison another.

### 3.3 Formats and relations

Two levels, both narrow today:
- **Format** (what the answer is, as a function of the bridge): first letter, and counts. The counts set has 16 questions on two facts, so it is one format in name only.
- **Relation** (hop 1): landmark → country, person → birth city, university → city. All geography.

The format restriction is forced by the probe: the bridge and the answer must share nothing inside the model, which rules out natural lookups (capital, language, demonym). So the removal arm stays with arbitrary lookups, and the claim is scoped to them. The peer block and patching have no such constraint, so the peer half of the 2×2 (visible versus blocked) can run on natural two-hop questions as well; that is how the peer-side findings generalise beyond the format.

**Candidate arbitrary-lookup formats to screen** (each needs the four gate checks; the overlap check is what kills most natural lookups):

| Bridge kind | Lookup | Example | Likely issue |
|---|---|---|---|
| country / city | first letter | Colosseum → Italy → I | in use |
| country / city | last letter | Colosseum → Italy → Y | **dropped**: untestable in the ideas run |
| month | number in the year, or days | Halloween → October → 10 / 31 | clean, small closed answer set |
| element | atomic number | the gas most of the air is → nitrogen → 7 | check the number is one token |
| planet | position from the sun, number of moons | the red planet → Mars → 4 / 2 | small set, well known |
| instrument | number of strings | Hendrix's instrument → guitar → 6 | few distinct answers |
| shape / sign | number of sides | the shape of a stop sign → octagon → 8 | **dropped**: the name carries the number |
| sport | players per side | the sport with a home run → baseball → 9 | few distinct answers |
| animal / vehicle | legs / wheels | spider → 8, bicycle → 2 | in use (spider), rebuild with distinct facts |
| country | continent, driving side | Italy → Europe / right | **overlap suspect**; screen first, expect some to fail |
| country | currency | Italy → Euro | **dropped**: every currency name is two tokens on Qwen |
| country | first letter is a vowel, yes/no | Colosseum → Italy → I → Yes | three hops, binary; Gemini's vowel/consonant idea recast as yes/no (the class words are two tokens) |

Target: three formats that pass (letter, one number format on distinct facts, one more), each with at least two relations, ten or more questions per format, letters and numbers spread so no answer dominates (today S is the right letter for 6 of 25 on the 27B). Wrong peer answers drawn by rule from the same type, never the letter or number of a plausible alternative bridge (Spain's S for Colosseum confounds "gave in" with "wavered between countries"). Report per format and per relation before pooling.

### 3.4 Development and confirmation

- **Development:** the questions used so far (19 and 20 letters, 16 counts) and every new question drafted while the method is still being adjusted. They stay in the paper in full as the main result. Nothing is given up.
- **Confirmation:** new questions, drafted now, held back, run once with every choice frozen (windows, the four thresholds, onset rule, controls, labels). The gate applies to both sets identically; the split guards against **tuning**, not selection. On development data every threshold was chosen while looking at outcomes, and enough such choices can make noise look like a finding.
- **Reading the confirmation:** with three agreeing peers the bridge was still needed in 5 of 19 development questions; with only 19 that could have been 2 or 9 by chance. On 40 confirmation questions, roughly a quarter again (about 6 to 14) means the finding holds; 30 of 40 means it was a fluke or a product of tuning.

### 3.5 How many

The unit is the question. From the saved data, in log-odds of P(right), paired within question, the alone-versus-three-agreeing-peers effect averages about 6 with a spread (SD) of about 3 across questions.

Two pre-registered tiers for the group effect, chosen where the words of the claim change:
- **stops**: after removal the right answer is no longer the model's top choice (the written answer flips), and the removal effect is within the noise floor plus 1 log-odds;
- **weakens**: the removal effect is at least 2 log-odds smaller than in the hidden prompt. Two is about a third of the full alone effect; it is a judgment, defended by being fixed in advance and by reporting the continuous values alongside.

With SD 3, detecting a 2 log-odds difference at 80% power needs about 20 questions per contrast. Reading three formats separately therefore needs about 60 gated questions per model. With a pass rate of about two thirds (25 of 37 letters passed on the 27B), draft about 90 to 100 candidates per model, more for the formats whose overlap check is uncertain.

---

## 4. Phases

### Phase A: free, now

**A1. Re-analysis in `conformity_stats.py` (done 8 Nov; see §2)** (new output directory; keep the old summaries).
- Log-odds of P(right) per window from `windows.csv`; need in log-odds = largest early bridge fall − largest early everyday-word fall. Report next to need in probability.
- A **ceiling flag**: P(right) after removal still ≥ 0.9.
- An **inconclusive** label when the everyday-word or same-kind control also drops (threshold fixed before running, e.g. noise drop ≥ 0.2), replacing "not needed" in those rows.
- The **clean subset**: reach passes, bridge present, same-kind drop < 0.2, onset within one window of the alone prompt. Redo the label table on it.
- Explicit denominators everywhere: assessed + unassessed = selected cohort; empty subgroups say "no cases".
- Keep the bootstrap by question (counts: by fact) and the threshold-scaling sensitivity already in place.

**A2. Pre-register.** Write each claim with the result that would support it and the result that would sink it. Fix every new threshold (log-odds "still needed" ≥ noise + 2, "not needed" < noise + 1, the block pass marks in step 6, the clean-subset rules). Commit before computing.

**A3. Candidate questions (§3): drafted 8 Nov.** `questions/proposed/candidates.py` holds 171 candidates in 12 families (ids `C_`), none run: 98 first-letter questions in three relations (40 landmark → country, 30 person → birth city, 28 university → city), 14 month-number, 6 atomic-number, 5 planet-position, 5 players-per-side, 2 strings, 12 continent, 11 driving-side, 14 starts-with-a-vowel yes/no, and 4 even/odd letter-count controls expected to fail. Every edited word is one token on the Qwen tokenizer (checked; the currency family and 28 two-token bridges were dropped at that check). The module's seeded split gives 114 development and 57 confirmation questions, keeping each bridge's questions together and stratifying by family; `digest()` prints the sha256 of the table and split, to quote in the commit. The current sets stay as development. The gate screen (§3.1) is the first thing each GPU run does; Llama and Gemma tokenizers are checked on the pod before their screens.

**Output:** one table per model with label counts, need in probability and in log-odds, intervals by question, the clean-subset version, and the number of ceiling and inconclusive rows. If the 27B "not needed" share collapses under log-odds, the presentation's finding 1 is reworded now.

### Phase B: code, tested on the tiny model (done 8 Nov; `process.md` round 15)

**B1. Prompt variants** in `group_user` / `prompts()`:
- hidden answers, same length: peers write "?" (check it is one token in every tokenizer);
- hidden answers, natural: "(answer submitted)" (fix the one-word regex in `peer_positions`);
- told to copy: "You are Agent 1. Repeat Agent 2's answer.";
- told to ignore: "You are Agent 1. Ignore the other agents and answer on your own.";
- a **second wrong letter G** next to F. The F-versus-G contrast is the cleanest behavioural measure of peer influence (support for F when peers switch from G to F, and the reverse), and it is what the patching in step 7 needs.
- **peer supplies the bridge**, not the answer (see §5): "Agent 2: The country is Italy." with the model asked for the letter.

**B2. A peer block.** Takes the peers' token positions and a layer range. Full-attention layers: later positions cannot attend to peer tokens. Linear-attention layers (Qwen 3.5/3.6): zero the peer tokens' contribution to the running memory. Option to block only the answer positions, to test late direct copying. **Leak test:** with the block at every layer, changing I to F in the peers' lines must leave the output bit-identical.

**B3. Patching.** Run the "?" prompt, cache the state at the peer positions, write it into the letter prompt at chosen layers (positions line up because the lengths match). Same leak test. Identity patch (same run into itself) must change nothing.

**B4. Port from the scorecard:** own-part removal (`ablate_excl`) and answer-letter removal, inside group prompts. **B5.** A removal strength setting (0.25, 0.5, 0.75, 1).

### Phase C: GPU, 4B first (cheap, finds bugs), then 27B, then Llama and Gemma

**Step 4 first: the leak test on the real model.** One forward pair (peers I, peers F) with the block at every layer; the logits must match. If they do not, hub kernels have replaced the GatedDeltaNet forward: set `USE_HUB_KERNELS=0` and rerun. Nothing in step 6 can be read without this.

**Step 4a, before the screen: choose the lens.** The J++ lens passes 26 of 56 count questions against 16, halves the 4B's same-kind collateral and removes more of the bridge on the 27B with controls unchanged (`process.md` round 14). Proposed: J++ primary on the Qwen models, J-lens as replication on the development set; Llama and Gemma need a J++ fit. Pending Peter's decision.

**Step 4b, first in every run: the gate screen.** Run the four gate checks (§3.1) on every candidate for that model, solo prompt, both paraphrases. Record the funnel. Only questions that pass continue to steps 5 to 8, and the hidden prompt becomes the baseline for every "shrinks" comparison (§3.2).

**Step 5. Does the tool work in the group layout?** (the go/no-go gate)
- Hidden answers ("?" and "(answer submitted)"), 1–3 peers, both wordings. Expected if the tool works: bridge still needed, about as much as alone. If hidden prompts also come out "not needed", the layout breaks the tool and every "not needed" claim goes.
- Own-part and same-kind removal inside group prompts (specificity, matters most for disagree).
- Dose-response: sweep strength alone, hidden, agree. Working tool: smooth curves, agree curve shifted right. Broken tool: flat.

**Step 6. Block validation, then the main 2×2.** Pass marks fixed in A2:

| Known case | Expected under the block | Failing means |
|---|---|---|
| Made-up subject, wrong peers | P(peers' letter) falls to about alone level | block doesn't remove peer influence |
| Disagree prompts where the model gave in | back to the right answer | same |
| Hidden prompts | almost no change | off-target damage; the 2×2 can't be read |
| Told to copy, wrong peers | P(peers' letter) drops | block misses the copying |

Then on each agree prompt, four runs on the identical text, scored in log-odds of P(right): A intact/visible, B removed/visible, C intact/blocked, D removed/blocked. Only D collapses: peers were the backup (strongest form of the claim). Neither collapses: a second internal route or a weaker tool, step 5 decides. C well below A: the model leaned on the peers from the start. Disagree prompts: with peers blocked and bridge removed, the answer should worsen but **not** move to the peers' letter; "lost its route" predicts the drop, not the destination.

**Step 7. Patching.** Patch "?" states into the letter prompt at every layer (should agree with step 6: triangulation by two methods). Layer sweep from L upward: the earliest layer where patching stops working is where the peers were read, before or after the bridge acts. Share of effect: A−B against A−C against both. For the F/G pair, patch F-run activations into the G run and the reverse; narrow broad regions to components before any pathway claim.

**Step 8. Known-answer cases and breadth.** Told-to-copy and told-to-ignore give a hit rate. Held-back questions: one confirmation run on the 27B. Llama and Gemma letters (counts need the counts screen first). Gemma copy-only control. Find out why presence fails on Llama group-aware before using it. If budget allows: group sizes 5, 8, 12 and mixed groups (NEXT 1d), with same-length neutral filler to separate more peers from more text.

### Phase D: later, as its own project

Private-evidence tasks with an executable oracle, real multi-round groups (2, 4, 8 agents; replay first, live second), distributed evidence against a pooled-information solo baseline, transfer to an untouched task family. One axis at a time.

---

## 5. Ideas worth keeping in view (not prioritised yet)

**Peers supply information, not an answer.** Everything above puts the answer in the peers' mouths, so copying is always available and has to be ruled out by intervention. The alternative is to give peers *clues* the model must combine, so copying is impossible by construction and combining is visible in behaviour.

*The three boxes.* A prize is in a red, blue or green box. The agent knows "not red"; a peer says "not blue"; together that supports green. Alone the agent cannot solve it. Rotate both clues:

| Agent's clue | Peer's clue | Answer |
|---|---|---|
| not red | not blue | green |
| not green | not blue | red |
| not red | not green | blue |

If the answer follows both rows and columns, the model combined the two sources. Add: a peer who states the answer directly (copying possible) for contrast; a peer whose clue conflicts with the agent's ("not red" vs "it is red"), which separates misinformation from conformity; a redundant peer clue (same as the agent's) as a control for "any peer text helps". Mechanistically the elimination step is the bridge, and removal or patching can be aimed at the peer's clue span versus the agent's clue span to ask which source the computation runs through. This is the broad plan's private-evidence task in its smallest form and belongs at the front of Phase D.

*Peer supplies the bridge.* The cheap hybrid that runs on the existing pipeline now: "Agent 2: The country is Italy." and the model is asked for the letter. The peer gives the first hop; the model must still do the second. Compare with the peer giving the letter "I". Removing the Italy direction then asks whether the model took the peer's bridge into its own computation or ignored it; the F/G analogue is a peer who names the wrong country (Spain) and whether the model's letter follows. This separates "uses the peer as information" from "uses the peer as an answer" on the same questions, with the same edits, and costs one more prompt variant (B1). Worth a slot in the Phase C runs if step 5 passes.

*Evidence quality.* A peer who supplies a verifiable fact versus an unsupported conclusion ("it's green, trust me"). Construct so the oracle says when updating is warranted; otherwise every move toward a peer counts as conformity and legitimate learning is misread.

*Source count versus repetition.* One peer repeating a claim three times against three peers saying it once, with matched length. Separates "many sources" from "more text".

---

## 6. Claims and what they rest on

| Claim | Rests on | Now |
|---|---|---|
| Alone, the answer depends on the bridge | v1 scorecard | Done |
| The tool still works in group prompts | Step 5 | Missing |
| Agreeing peers shrink the bridge's effect | A1 (log-odds) | Redo; may soften to "weaken" |
| …because the model uses the peers as a backup | Step 6, checked by step 7 | Missing |
| Disagreeing peers fill the gap when the bridge is removed | Pull labels + step 6 (disagree) | Partial |
| The peers are read after the bridge acts | Step 7 sweep | Missing |
| The model can combine its own and a peer's information | §5 clues designs | Idea |
| Holds across models, formats and new questions | Step 8 | Partial |

**Wording that stays honest:** "the answer depends on the model's own intermediate step", among the sampled cases where the measurement was validated, with the unresolved count alongside. Not "thinks for itself"; nothing about free-form or multi-turn answers until Phase D.

## 7. Order and cost

1. Phase A: free, this week. A3 is where the questions get found: drafted and split now, screened at the start of each GPU run.
2. Phase B: code, local, tiny model.
3. GPU run 1, Qwen3.5-4B: gate screen on all candidates, then steps 5–6 on the development set, plus the peer-supplies-bridge variant. Read, fix. The screen also tells us which new formats survive the overlap check.
4. GPU run 2, Qwen3.6-27B: gate screen, steps 5–7 on development, then the confirmation set once, frozen.
5. GPU run 3, Llama and Gemma: gate screen, steps 5–6, Gemma copy-only control.
6. Phase D as a separate project, starting with the three boxes.
