# Silent Dissent: next experiments

*Background: `docs/scorecard/v3_v4.md` (the scorecard, results on Qwen3.5-4B and Qwen3.6-27B, and the group-format pilot).*

## Where we are

- **The tool works.** The scorecard can show whether the model uses a hidden middle step (the bridge) to get its answer. It checks with removal edits, controls and gates.
- **Reliable questions so far:** "What letter does the name of [the city where X was born / the country where landmark Y is] start with?"
  - In Qwen3.6-27B, Colosseum (Italy → I), Gates (Seattle → S) and Piaf (Paris → P) score quality 0.96–0.98.
  - In Qwen3.5-4B, spider legs (0.94) and hockey players (0.74) also work.
- **Group format** (three peers answer first):
  - with the right answer from peers, 5 of 6 questions copy it instead of using the bridge;
  - with a wrong answer from peers, 5 of 6 give in;
  - only Colosseum keeps using its bridge (Italy) throughout.

## Next experiments, in order

> **1f's swap test is done** (re-run on Nov 7, results under 1f): the bridge's direction carries its content on both models when the question's words are edited. Patching (1f test 2) is still open. **Flagged to revisit:** person questions hold their bridge on different tokens from landmark questions (1g).

### 1. When an agent gives in, does the bridge still push back? (removal)

- **Why:** the scorecard measures P(right answer). When the agent gives in, that is about 0, so there is nothing to measure. We measure the answer the agent gives instead.
- **Test:** in the "peers disagree" prompts, remove the bridge (Seattle) and measure P(peers' wrong answer, "B"), and the balance between P("B") and P("S").
  - **No change:** giving in is pure copying; the bridge plays no part.
  - **P("B") goes up / P("S") goes down:** the bridge was still pulling toward the right answer. That is "silent dissent", shown causally.
- **Controls:**
  - removing everyday words (noise floor);
  - removing a same-kind word (Boston), which must not do the same;
  - the reach gate in this prompt.
- **Watch for the ceiling:** where P("B") ≈ 1 (Gates) it can't go up. Prefer cases where giving in is partial (Mozart P("S") 0.41, spider 0.36), and use the balance between the two answers.

**Result (Qwen3.6-27B, H100):** `scripts/giveup_remove.py`; raw `results/giveup_remove_qwen36_27b/`; early-window summary `early_summary.csv`.
- **Prompts:** 6 letter questions × 1, 2 or 3 peers giving a wrong letter.
- **Windows:** only those ending before the answer onset (layer 54). Removing the answer's own letter there barely matters (≤ 1.6), so the effects below come from the bridge.
- **Shift** = change in log P(peers' letter) − log P(right letter).

| Case | Prompts | Remove the bridge (own part only) | Everyday words | Same-kind word |
|---|---|---|---|---|
| Agent holds | 9 (Colosseum ×3, Mozart ×2, spider ×2, Gates/Piaf/York ×1) | P(right) → ~0, agent gives in; shift 2.0–13.6 | ≤ 0.8 | usually much smaller (Gates, Mozart, York partly overlap) |
| **Gives in partly** (P(right) still 0.36–0.47) | Mozart 3 peers, spider 3 peers | **P(right) 0.47 → 0.00 and 0.36 → 0.00; shift 8.4 and 4.3** | 1.3, 0.1 | 5.0, 0.4 |
| Gives in fully (P(right) ≈ 0.00–0.04) | Gates, Piaf, York × 2–3 peers | small further shift, 1.3–2.3 | 0.4–0.9 | 0.9–2.5 |

**Reading:**
- **When the agent holds,** its bridge is what holds it.
- **When it gives in partly,** the bridge is still actively pulling toward the right answer, and removing it completes the giving in. This is silent dissent, shown causally (spider is the cleanest: same-kind and everyday words do nothing).
- **When it gives in fully,** the bridge's leftover effect is small and close to the same-kind controls. Those agents mostly just follow the peers.

### 1b. Follow-up: when peers agree, is the model copying, or is the bridge just not needed?

- **Why:** in the group pilot, with peers giving the right letter, removing the bridge did nothing in 5 of 6 questions. That only shows the bridge isn't *necessary* there. It fits three pictures:
  1. the model copies the peers and doesn't think of the bridge;
  2. it still works the answer out through the bridge, but copying gets there too, so either route is enough;
  3. it thinks of the bridge, but doesn't use it for the answer.
- **Tests:**
  - **Is the bridge still there?** Read the bridge's J-lens rank in the "agree" prompts. If it's strongly present, picture 1 is out.
  - **Is the bridge a backup route?** Remove the peers' letter (its direction on the peer lines) instead of the bridge. If the answer survives, another route still produces it. Then remove both: if only removing both kills the answer, the two routes are redundant (picture 2).
  - **Does the bridge route still work on its own?** Use "agree" prompts where the peers' letter is hidden or neutral (e.g. "Agent 2: (answer submitted)"), so copying can't supply it. Check whether the bridge becomes needed again.
- **Until then,** the claim is only: *in a group, the bridge stops being necessary.* "Copying" is a hypothesis.

### 1c. Present but unused? Read out the bridge in every group prompt

- **Why:** the paper measures **presence** (the bridge ranks high in the J-lens). We measure **use** (removing the bridge hurts the answer). The two can come apart, and the paper's argument assumes they don't. Showing presence without use, in the paper's own terms, turns "their measure *could* mislead" into "here it *does*". The key cases are the full give-ins, which is exactly where the paper claims silent dissent.
- **Test:** in each group prompt (alone, agree, disagree × 1–3 peers, the 6 letter questions on Qwen3.6-27B), run the J-lens readout (`src/consulted_bridges/readout.py`). Report it next to the need we already have from the removal test:
  - the bridge's best rank at the tokens before the answer;
  - its rank in the paper's layer band.
- **Read each prompt into one cell:**

  | | Removing it hurts (used) | Removing it does nothing (not used) |
  |---|---|---|
  | **J-lens shows the bridge (present)** | consulted bridge | **present but unused: the paper's confound** |
  | **J-lens doesn't show it** | (unlikely) | not computed |

- **Controls:**
  - the same readout for a same-kind word (Vienna) and a random city (Tokyo), so "present" isn't just an Austrian context lighting up;
  - the "alone" prompt as the baseline, to see whether peers reduce the bridge's presence or leave it unchanged while use drops.
- **Cost:** one forward pass per prompt, a few minutes on a pod using the shared volume (`docs/runpod.md`). It also answers 1b's first test ("is the bridge still there?").

### 1d. How do things change with more agents?

- **Why:** experiments 1 and 1c used 1–3 peers, and already show a trend: more peers push the agent further toward giving in, and the bridge's specific pull shrinks. Three points can't show the shape of that curve. Is it a gradual slide, or a tipping point? And is there a group size where the bridge stops mattering entirely?
- **Test:** the same 6 letter questions on Qwen3.6-27B, with **1, 2, 3, 5, 8 and 12 peers**, in both "agree" and "disagree". Also mixed groups (e.g. 3 wrong + 1 right, 3 wrong + 3 right), to see whether a single dissenting peer restores the bridge's role. For each group size, report:
  - P(right letter), P(peers' letter) and what the agent answers (where it tips from holding to giving in);
  - the bridge's **use**: the removal shift from experiment 1, against the same-kind and everyday-word controls;
  - the bridge's **presence**: the J-lens readout from 1c.
- **What to look for:**
  - whether use falls with group size while presence stays (the paper's confound growing);
  - whether the tipping point differs by question (Colosseum held at 3 peers; does it give in at 8?);
  - whether a single agreeing peer is enough to keep the bridge consulted.
- **Watch for:** longer prompts as the group grows. The bridge sits further from the answer, and the scorecard's onset and windows are per prompt, so measure them in each prompt rather than borrowing from the 3-peer one. A neutral filler of the same length (agents saying something unrelated) separates the effect of more *peers* from the effect of more *text*.
- **Cost:** about 6 questions × 6 sizes × 2 conditions, plus the mixed groups. Roughly an hour on a pod with the shared volume.

### 1e. Redo experiment 1 with a per-prompt answer onset

- **Why:** experiment 1 counted only windows ending before layer 54. That is the answer onset of the Colosseum letter question asked *alone*, borrowed for all 18 group prompts. A group prompt is a different prompt, and peers may make the answer form earlier (copying is easy). If the real onset in a group prompt is earlier, some of the "pushback" windows would be removing the answer as it is written, not the bridge.
- **What already covers it:** in each group prompt, removing the answer letter in the same windows shifts the balance by at most 1.6, against 4–14 for the bridge when the agent holds or gives in partly. So the main results stand.
- **Where it is weakest:** full give-ins, where the bridge's shift (+1.5 to +2.8) is in the same range as that answer check.
- **Test:** for each group prompt:
  1. measure its own onset with the v4.1 rule (remove the answer's own part window by window; the first window from which it keeps killing the answer);
  2. recompute the pushback and its controls on windows ending before *that* onset.
- **Bonus result:** whether peers make the answer form earlier than when the question is asked alone, which would fit the "copy route" picture.
- **Cost:** about 15–20 min of A100 time on the shared volume (`docs/runpod.md`), ~$0.50.

### 1f. Does the bridge's direction carry its *content*? (swap and patching)

- **Why:**
  - Removal shows the bridge's direction is *necessary*. It doesn't show what the direction holds: the content "Italy", or something generic like "a country goes here". The critique (`docs/reviews/critique_method.md`, C2) found our swaps rarely work in the 27B (2 of 37 on letter questions).
  - Peter's Neuronpedia test says otherwise. On Qwen3.6-27B, swapping " Italy" → " Spain" in layers 9–27 turns the Colosseum answer from "I" into "S".
  - So our swap may be mis-set. Ours acts on layers 12–53 and every position from the question on; Neuronpedia's on layers 9–27.
- **Test 1 (swap, cheap):** on the letter questions, swap the bridge for 4–6 alternatives with different letters.
  - **Compare layer ranges:** our current one; Neuronpedia's 9–27; and short early and middle windows.
  - **Measure:** the share of swaps that give the swapped word's letter.
  - **Control:** swapping in a word of another kind (an everyday word) should not produce its letter.
- **Test 2 (patching, stronger):**
  - **The edit:** run the same question with another subject whose bridge differs (Eiffel Tower → France), and paste its internal state, at the bridge's positions and layers, into the Colosseum prompt.
  - **The reading:** if the answer flips to "F", the content travels in that state.
  - **The control:** patching between two subjects with the *same* bridge (Colosseum and the Trevi Fountain, both Italy) should not change the answer.
- **Use:** a "carries the content" score next to quality, so a consulted bridge is shown to be both necessary and to carry what the answer reads.

**First try (Qwen3.6-27B only, stopped before the 4B):** `scripts/swap_check.py`; log `results/swap_check_partial_27b.log` (no CSV, since the script writes it at the end).
- **The test:** on 8 letter questions, swap the bridge for 5 same-kind words with different letters, plus 3 everyday words, at every position from the question on. "Follows" = the answer becomes the swapped-in word's first letter.
- **Results** (count of the 5 same-kind swaps that follow):

  | Question | Layers 12–63 | 9–27 | 0–15 | 8–23 | 16–31 | 32–53 |
  |---|---|---|---|---|---|---|
  | Colosseum (Italy) | 1 | **2** | 0 | 0 | 0 | 0 |
  | Gates (Seattle) | 0 | 0 | 0 | 0 | 0 | 0 |
  | Piaf (Paris) | 2 | 0 | 0 | 0 | 0 | 1 |
  | Mozart (Salzburg) | 0 | 0 | 0 | 0 | 0 | 0 |
  | York University (Toronto) | 0 | 0 | 0 | 0 | 0 | 0 |
  | Eiffel Tower (France) | 2 | 0 | 0 | 0 | 0 | 0 |
  | Machu Picchu (Peru) | 2 | 0 | 0 | 0 | 0 | 0 |
  | Mount Fuji (Japan) | 0 | 0 | 0 | 0 | 0 | 0 |

- **Reading:** the swaps mostly fail at every range: 10 of 240. Layers 9–27 work only on Colosseum (France → "F", Germany → "G"; Spain stays "I"). So in our setup Peter's Neuronpedia result doesn't generalise, and it doesn't fully reproduce even on Colosseum.
- **Fix before re-running:**
  - **Range bug:** the "scorecard" range came out as layers 12–63 instead of 12 to onset−1. `sc(54)` is 108 in the 64-layer model and got clipped to 63, so it ran past the answer onset. Use `sc(6)` to `onset − 1` from the card.
  - **Neuronpedia also swaps at layers 42–54 (Peter).** On Qwen3.6-27B, swapping " Italy" → " Spain" in layers 42–54 also gives "S". That is the range where our removal says Italy matters (from layer 42; answer at 54). Our nearest range, 32–53, gave Spain → "I". So the layers aren't what separates the two; the swap itself or its positions must be.
    - **Positions (the likely cause):** Peter edits from token 27 on. Token 27 is the line break just after the end of the user's turn, so his edit covers only the 8 chat-template tokens (27–34: line break, `<|im_start|>`, `assistant`, line break, `<think>`, blank, `</think>`, blank). Our swap edits tokens 3–34, the whole question included. Same for our removals, which use the same positions (`Card.ALL = from:<question start>`). **Test first:** our swap at layers 42–54 on tokens 27–34 only. If that gives "S", re-run the swap check (and check removal) with template-only positions.
    - **The swap:** ours exchanges least-squares coefficients on the two unit directions (the mean over each word's token ids). Neuronpedia's "Swap J-Lens Readout" may scale or select differently; the gear icon in its panel has settings.
  - **Match Neuronpedia's setup:** find out which positions it edits (all tokens, or only from the subject on) and whether it swaps the coefficient or adds the direction. Then reproduce Spain → "S" on Colosseum *exactly* before testing other ranges.
  - **Add the 4B half,** then test 2 (patching), which doesn't depend on the lens directions.

**Replication (27B):** see [`docs/swap_replication.md`](docs/swap_replication.md).
- **Reproduced:** Spain → "S" (p 0.98–0.99) in layers 9–27, 32–54 and 42–54. The cause is the formula. Neuronpedia *moves* the bridge's component onto the new word (`h + (h·ŝ)(t̂ − ŝ)`), at every position. Ours (like the paper's) *exchanges* two coordinates, and here they are almost equal, so it does almost nothing. The gain, the vectors and Peter's position-27 selection don't matter: the UI applies the swap everywhere.
- **Generalises:** with the move, 2–4 of 4 swaps follow on Eiffel, Gates and Piaf. Moving a word the model isn't using does nothing. But late swaps (and any swap at the onset layer 54) also let "table" → "T" through.
- **Proposed:** removal metrics keep q→end (narrower position sets show no need at all before the onset). Swaps should use the move on the question's words, in layers before the onset, with an everyday-target control and a non-bridge-source control.

**Re-run (both models, Nov 7):** rules fixed before the run in [`docs/swap_test_rules.md`](docs/swap_test_rules.md); details in [`docs/swap_replication.md`](docs/swap_replication.md) (end); results `results/swap2_summary/`.
- **The test:** on every question where the bridge is needed and comes first (39 on the 4B, 34 on the 27B), move the bridge onto 4 same-kind words, over all layers before the answer onset. "Follows" = the answer becomes that word's own answer.
- **Rule A** (question words only): 0.70 of swaps follow on the 4B, 0.72 on the 27B. Everyday words stay out on 33/39 and 29/34 questions. Moving an unused word does nothing (1 of 73). **The bridge's direction carries its content.**
- **Rule B** (question to end, removal's tokens): follows more (0.78, 0.93), but everyday words get through on about half the questions, and moving an *unused* same-kind word follows 19 of 73 times. Not a clean content test; keep it as a check.
- **Person questions** (birth city letter) don't follow under Rule A (0.31 on the 4B, 0.00 on the 27B) but do under B. Their bridge seems to sit on the template tokens, landmarks' on the question words. Removal uses both, so quality is unaffected. **Notable and not yet understood: see 1g.**
- **Next:** test 2 (patching), which doesn't depend on the lens directions.

### 1g. Open: why does a person's birth city sit on different tokens from a landmark's country? (revisit)

> **Flagged as notable (Nov 7).** Found in the 1f swap re-run, not explained yet. It matters for Silent Dissent: the group and give-up experiments use person questions (Gates, Piaf, Mozart).

- **What we saw** (Qwen3.6-27B, swap over the whole pre-onset range; follow share of 4 same-kind swaps):

  | Questions | Rule A: question words only | Rule B: question to end (incl. template) |
  |---|---|---|
  | Landmark → country letter (`V_LM_*`, 10) | 0.98 (0.75–1.00) | 1.00 |
  | University → city letter (`V_LU_*`, 4) | 1.00 on all 4 | 1.00 |
  | Person → birth city letter (`V_LP_*`, 5) | **0.00 on all 5** | 0.65 (0.25–1.00) |
  | The bank's person questions (`L_gates`, `L_piaf`, `L_mozart`) | 0.25–0.50 | 1.00 |

  - **The 4B shows the same pattern, less sharply.** Rule A: landmarks 0.84, people 0.31.
  - **The split isn't simply "people vs places".** `L_york` (York University → Toronto) gets only 0.25 under Rule A on the 27B, though the other university questions get 1.00.
- **Why the Rule B result isn't just a leak here:** Rule B is unreliable overall (everyday words get through on about half the questions, and an unused same-kind word follows 19 of 73 times). But on the person questions both of its controls pass on the 27B: moving an unused same-kind word does nothing on all 8, and the everyday-word check passes on 7 of 8.
- **Removal agrees.** Michelle Obama → Chicago (27B): removing Chicago from the question words alone leaves "C", while removing it from the question to the end gives "A".
- **What it might mean:**
  - The model may hold a person's birth city only once it has read the whole question (on the template tokens), and a landmark's country already on the landmark's own words.
  - Or the person's name pulls the city in late, and the J-lens direction on the question tokens doesn't capture it there.

  Either way, "the bridge is on the question's words" is not true for every question type, and the swap rule that works for landmarks (Rule A) misses people.
- **Tests to run:**
  1. **Per-position readout:** the J-lens rank of the bridge at each token (subject, rest of the question, each template token) by layer, for person vs landmark questions. Where does the bridge first appear?
  2. **Swap on the template tokens only** (positions after `<|im_end|>`), with the same everyday and unused-source controls. If person questions follow there and landmarks don't, the split is confirmed.
  3. **Activation patching** (1f test 2) at question vs template positions: paste another person's state (born in a city with a different letter) and see which position set carries the answer.
  4. **Group format:** recheck whether the give-up results (experiment 1) depend on template positions for Gates, Piaf and Mozart, since the peers' lines change what comes before the template.
- **Where the numbers are:** `results/swap2_summary/questions.csv`, `results/swap2_qwen36_27b/`, `results/swap2_qwen35_4b/`; write-up in [`docs/swap_replication.md`](docs/swap_replication.md) (end).

### 2. Does strengthening the bridge make the agent hold? (steering)

- **Test:** push the bridge's direction up (steer toward "Seattle") in the "peers disagree" prompts, at a few strengths. Measure P("S"), P("B") and what the model answers.
  - **P("S") rises / the agent stops giving in:** the bridge's strength decides between holding and giving in.
  - **No effect:** the peers override the bridge regardless.
- **Controls:**
  - steer a same-kind word (Boston) and an everyday word by the same amount;
  - steer the bridge in the *alone* prompt, to check the push works at all (a dose-response check);
  - watch for broken output at high strength.
- **Note:** the paper tried something similar ("pushing toward the bridge", exploratory). Ours adds matched controls and uses questions where the bridge is known to be consulted.

**Status: not done; the method needs calibrating first.** Partial raw data: `results/giveup_steer_v2_partial_qwen36_27b/`.
- **Additive steering** (`steer`, α × |h| added in every layer of a 12-layer window) broke Qwen3.6-27B even at α = 0.02: P > 1 in the alone check, empty answers.
- **Amplifying the existing component** (`amplify`, ×1.5 to ×3) also disrupts it. Amplifying *everyday words* already pushes the agent toward the peers (shift ≥ +1.7). Only the answer's own letter (the positive control) behaves as expected.
- **Next:**
  1. **Calibrate:** sweep strength (×1.1–×1.5) and window size (1–3 layers) on the question asked alone, and pick the largest setting where amplifying everyday words changes nothing and the right answer stays.
  2. **Then run the bridge test** at that setting, with the same controls.
  3. **Do the calibration on the 4B model** (cheap, A40) before spending H100 time.

### 3. Bad questions that pass the gates (checking the scores)

- **Why:** so far the deliberately bad questions were mostly caught by the gates. Only banana → Yellow reached scoring, and it isn't clearly bad.
- **Test:** design shortcut or echo questions that the model answers correctly and where the edit reaches the bridge. Confirm their quality is low. Mozart's country (answer present from the start) is one existing example.

### 4. From single good questions to question *types* (variant sets)

**Trend questions (23 new question types, run Nov 7 with the trimmed card; `results/trend_*`, `results/swap2_summary/trend.csv`):** few are testable.
- **4B:** 3 consulted bridges (World Cup wins of Brazil, states of Australia, and the designed-bad flag stars of the USA, which should have failed), plus 3 bystanders, 3 unclear, 1 spread out and 13 untestable (the model gets 10 wrong; in 3 the edit doesn't reach the bridge).
- **27B:** 1 consulted bridge (federal states of Austria, quality 0.92), 1 category effect (Russia's dialling code), plus 2 bystanders, 2 unclear, 1 spread out and 16 untestable (in 13 the edit doesn't reach the bridge).
- **So far,** the letter family stays the reliable type. The flag-stars control passing on the 4B needs a look (experiment 3).

- **Why:** a few good questions could be lucky. We want to know whether the *type* behaves consistently.
- **Test:** once a question is reliable, repeat its exact format with small variants to build a larger set. For example, the letter of the birth city for 50–100 people the model knows (from TwoHopFact, single-token cities), and the letter of the country for the 48 landmarks.
- **Report:**
  - the distribution of quality across the set;
  - how often each gate fails;
  - whether the group-format behaviour (copy / give in / hold) is consistent within the type.
- **Rule:** the criteria and thresholds stay fixed. No re-tuning on the variants.

### 5. J++ (the improved lens on Neuronpedia)

- **Why:**
  - in Qwen3.6-27B, the single J-lens direction fails to remove the bridge in 30 of 62 questions (gate 4);
  - J++ may hold the bridge better, giving more testable questions and cleaner edits.
- **Test:** re-run the scorecard (and experiments 1–2) with J++ directions in place of J-lens ones. Compare:
  - gate 4 pass rate;
  - quality;
  - verdicts on the same questions.
  Also compare against the plain logit lens as the simplest baseline.
- **First step:** check which models have J++ files available, and how to load them in the harness.

### 6. Other models (later)

- The paper's other models: Gemma-4-E4B and Llama-3.1-8B. Run the same scorecard and group pilot to see whether the reliable question types carry over.

### 7. Stronger models with a J-lens (later)

- **Why:** Qwen3.6-27B still gets 11 of our 62 questions wrong (gate 1). That includes all 8 wordplay questions, the planet after the Red Planet, Pikachu's species and the turkey's legs. A stronger model could make those testable. It also shows whether the reliable question types (the letter family) and the group results hold as models get bigger.
- **What's available:** checked in Neuronpedia's lens repository (`neuronpedia/jacobian-lens` on Hugging Face). Models larger or stronger than Qwen3.6-27B with a J-lens file:

  | Model | Size | Lens branch | Rough GPU need (bf16) |
  |---|---|---|---|
  | **Llama-3.3-70B-Instruct** | 70B dense | `main`, `qwen-n1000` | ~140 GB: one H200 (141 GB, tight) or 2× 80 GB GPUs |
  | **Qwen3.5-122B-A10B** | 122B MoE (10B active) | `main` only | ~245 GB: 4× 80 GB or 2× H200 |
  | **DeepSeek-V4-Flash** | large MoE (the J-PACT paper's model) | `main` only | multi-GPU; check size first |
  | Gemma-4-31B | 31B dense | both | ~62 GB: one 80 GB GPU |
  | OLMo-3-32B / Qwen3-32B | 32B dense | both | ~64 GB: one 80 GB GPU |

- **Suggested order:**
  1. **Gemma-4-31B:** a different model family at about the same size, on one 80 GB GPU, using the same pod setup. It also covers item 6, since it's Gemma.
  2. **Llama-3.3-70B-Instruct:** the strongest model that still fits a modest setup. It also tests whether results carry across families (the paper used Llama-3.1-8B).
  3. **Qwen3.5-122B-A10B or DeepSeek-V4-Flash** only if 70B still fails the questions we care about.
- **Test:** first the knowledge gate alone on all 62 questions (cheap: a few generations each). Then the full scorecard on the questions it newly gets right, plus the 6 letter questions as a reference.
- **Check first:**
  - **That the harness supports each architecture.** It reuses the authors' loaders, which cover Qwen, Gemma and Llama; MoE models may need work.
  - **Lens quality.** The J-PACT review found one Neuronpedia lens file was an unfinished checkpoint.
  - **Storage.** 70B adds ~140 GB, so the shared volume (250 GB, 74 GB used) would need to grow.
- **Cost:** a 70B run on 2× 80 GB GPUs is roughly $4–8/h. Needs Peter's go-ahead.

## Housekeeping

- **Pod storage:** the stopped RunPod volumes still bill a few cents an hour. Everything is copied to the Mac, so they can be deleted when Peter says so.
