# Swap replication: why Neuronpedia's swap works and ours doesn't

*Qwen3.6-27B, J-lens `qwen-n1000`, one A100 pod for 11 minutes (~$0.33). Script `scripts/swap_np.py`; results `results/swap_np/` and `results/swap_np2/`.*

**Short answer:** we reproduce Neuronpedia's Spain → "S". Our swap failed because of its *formula*, not its positions, layers or vectors. Neuronpedia *moves* Italy's component onto Spain. Ours *exchanges* the two coordinates, and on this prompt those are almost equal, so exchanging them changes nearly nothing.

## Setup and terms

- **The question:** "What letter does the name of the country where the Colosseum is located start with? Answer with one letter." Qwen chat template, thinking off. The model says "I" (p = 0.99).
- **Positions:** token indices in the chat prompt.
  - 0–2: `<|im_start|>user\n`.
  - 3–25: the question's words. The subject "Colosseum" is 13–15.
  - 26: `<|im_end|>`.
  - 27–34: the **template** tokens: line break, `<|im_start|>`, `assistant`, line break, `<think>`, blank, `</think>`, blank.
  - **"q→end"** = 3–34. This is what the scorecard edits now (`Card.ALL`).
  - **"question"** = 3–25.
  - **"all"** = 0–34.
- **h** = the residual stream after a layer, at one position.
- **Lens vector of a word** at layer *l*: `J_lᵀ · W_U[word]`, where `W_U` is the unembedding and `J_l` the J-lens matrix. **ŝ** is the unit vector for the source word (Italy), **t̂** for the target (Spain).
- **Gain:** the final norm's per-dimension weight. Our harness multiplies it into `W_U[word]` before `J_lᵀ`; Neuronpedia and the paper don't.
- **Follows:** the model's first answer token is the swapped-in word's first letter. Every result scores the first answer token, greedy.

## How Neuronpedia's swap works (from its code)

| | Neuronpedia | The paper (Gurnee et al.) | Ours (`interventions.py`, `swap`) |
|---|---|---|---|
| Word vector | unit `J_lᵀ W_U[w]`, no gain | raw row of `W_U J_l`, not normalised | unit `J_lᵀ (gain · W_U[w])` |
| Swap | **move:** `h ← h + (h·ŝ)(t̂ − ŝ)` | **exchange:** `c = V⁺h`, `h ← h + V(σ(c) − c)` (σ swaps the two entries; α = 2 in one experiment) | exchange (least squares), same as the paper |
| Positions | every position, prompt and generated (only BOS excluded) | "at all token positions" | from the question on |

Sources:
- **Neuronpedia's inference server** (github.com/hijohnnylin/neuronpedia):
  - `apps/inference/neuronpedia_inference/schemas/lens.py`: the `swap_token` comment gives `h ← h − (h·ŝ)ŝ + (h·ŝ)t̂`.
  - `endpoints/lens/prompt.py`: `_build_steer_deltas` gives the unit `J_barᵀ w` direction, built from the raw unembedding row; `_build_lens_steering_spec` builds the swap.
- **interp-engine 1.14.3** (PyPI): `steer_specs.SwapSpec` is `h → h + (h·v̂)(t̂ − v̂)`. In `steer.py`, `position_mask` lists positions to *exclude*, and the lens passes only BOS positions.
- **The paper** (transformer-circuits.pub/2026/workspace): Methods, "Technical details of J-lens use cases". The definition was read from the first third of the page; the rest was not checked.

**What Peter's UI choices actually do:**
- **Selecting position 27:** the request has no positions field. The selection only sets the *default layers* (where the token appears in the readout, `steerScopeSubset` in `use-jlens-analysis.tsx`). The swap is applied everywhere. Neuronpedia numbers positions from 0 over the same template tokens as ours, with no BOS.
- **The gear** ("Advanced settings", `jlens-steer-panel.tsx`) has two settings:
  - "Swap Generated Tokens": on by default for a new swap.
  - "Generated Tokens": how many tokens to generate.

  Neither can change the first answer token: that token is predicted at the last prompt position, which is always edited.
- **Sampling** is greedy by default (`DEFAULT_LENS_TEMPERATURE = 0`).

## Results

**Colosseum, Italy → Spain** (first token, P of that token, directions without gain):

| Swap | Positions | 9–27 | 12–41 | 32–54 | 42–54 |
|---|---|---|---|---|---|
| **Neuronpedia (move)** | all / q→end | **S 0.99** | **S 1.00** | **S 0.99** | **S 0.98** |
| Neuronpedia | template 27–34 | I | S 0.81 | S 0.95 | S 0.98 |
| Neuronpedia | 27 only | I | I | I (S 0.24) | I (S 0.29) |
| Ours (exchange) | all / q→end | I | I | S 0.56 | S 0.81 |
| Ours | template | I | I | I | I |
| Paper (exchange, raw vectors) | all | I | I | S 0.53 | S 0.77 |
| Paper, α = 2 | all | writes "Spain" | writes "Spain" | writes "Italy" | writes "Spain" |
| Raw-projection exchange | all | I | I | I | I |

- **France, Germany, Japan** behave like Spain under the move: 4/4 follow in every range above with all positions.
  - Under our exchange: 2/4 at 9–27 and 12–41, 3/4 at 32–54, 4/4 at 42–54.
- **Other questions** (move, all positions; count of 4 cities or countries that follow):
  - Eiffel (France): 4/4 at 9–27, 32–54 and 42–54.
  - Gates (Seattle): 2/4, 4/4, 4/4.
  - Piaf (Paris): 2/4, 4/4, 3/4.
  - Our exchange gets 0/4 at 9–27 on all three questions, and 1–3/4 later.
- **The gain doesn't matter:** with and without it, the directions have cosine 0.97–0.99, and outcomes differ in 9 of 300 settings.
- **Generation:** greedy output is "S⏎" with the move at 9–27, 32–54 and 42–54, with or without editing generated tokens.

**Why the exchange fails.** Italy's and Spain's lens vectors overlap (cosine 0.80 at layers 9–32, 0.54–0.65 at 42–54). The residual reads almost equally on both: at layer 27 on the question tokens, h·ŝ = 0.028 and h·t̂ = 0.031 of |h|. Exchanging two nearly equal coordinates is close to a no-op. The move instead lowers Italy's reading and raises Spain's, each by (1 − cos)(h·ŝ).

**Controls** (Colosseum, the move):
- **Is it the bridge being moved?** Moving a word the model isn't using (table → Spain, river → Spain, Japan → Spain) never gives "S", at any layers or positions. Adding Spain at Italy's strength without removing Italy mostly breaks the reply. So the effect needs Italy's own component.
- **Everyday target (Italy → table):**
  - fails at 9–27 on all four questions;
  - fails on the question tokens at every layer range;
  - **follows ("T") at layers 32–53 only when the question and the template are edited together** (q→end: 0.74 at 32–53, 0.75 at 42–53). Template only (P(T) 0.22, 0.04) and question only (0) don't leak before layer 54. Likely the same two copies of Italy that removal found: replacing both leaves no country, and the model spells whatever word it holds.
- **Layer 54 = the answer onset.** Removing the answer's own part kills it only from layer 54 on. Swapping at layer 54 *alone* gives "S" (0.98 with the move, 0.56 even with our exchange). Our exchange works at 32–54 but not 32–53, so its late successes come from editing the onset layer. This also explains why `swap_check.py`'s 32–53 range failed.
- **Where the swap acts:**

  | Swap positions | 9–27 | 32–53 | 42–54 |
  |---|---|---|---|
  | Question words only | S 0.79 | S 0.98 | S 0.97 |
  | Subject "Colosseum" only | I | S 0.98 | I |
  | Template only | I | S 0.96 | S 0.98 |

  Italy is read at the subject around layers 32–41, then carried onward.

**Removal** (our `ablate`, Italy, 12-layer windows; drop = 1 − P after / P before):

| Positions | 30–41 | 36–47 | **42–53** | 48–59 |
|---|---|---|---|---|
| q→end (3–34) / all | 0.01 | 0.02 | **0.99** | 1.00 |
| template 27–34 | 0.00 | 0.00 | 0.01 | 0.98 |
| 27 only, last only, subject only | 0.00 | 0.00 | 0.00 | 0.00 |

- Removing on the question words only, over layers 32–53 in one go, drops 0.01.
- Before the onset, removal kills the answer only when question and template positions are edited *together*. Italy is held redundantly in both.
- Removing an everyday word (table) drops nothing in any window.

## Which positions each metric should edit

| Metric | Positions | Why |
|---|---|---|
| Need (removal) | **q→end, as now** | The only setting that shows the bridge is needed before the onset: 0.99 vs ≤ 0.02 for every narrower set. Narrower sets give false "not needed". All (0–34) gives the same as q→end. |
| Strict specificity | q→end | A removal of part of the bridge. It must use need's positions to be comparable. |
| Isolation | q→end | Same reason. |
| Answer onset | q→end (no change) | The answer's own part dies only in 54–63 for the last token, template, q→end and all alike. The choice doesn't move the onset here; keep it equal to need's. |
| Reach gate | q→end of its own prompt | It checks that need's edit reaches the bridge, so it must be the same edit. Not re-tested here. |
| **Swap** | **The question's words (3 to `<|im_end|>` − 1), Neuronpedia's move, layers ending before the onset** | Spain → "S" at 0.79–0.98. The everyday target never follows, and non-bridge sources do nothing. Editing question and template together lets "table" through at 32–53 (layers × positions grid in the explainer page); question words alone pass the control at every range before the onset. |

## Proposed changes (not made)

1. Add a `move` swap kind to `src/consulted_bridges/interventions.py` (`h + (h·ŝ)(t̂ − ŝ)`). Keep `swap` (the paper's exchange) for comparison.
2. Scorecard swap test:
   - use the move, on the question's words, in early windows (layers < onset, never the onset layer);
   - report two controls: an everyday target, and a non-bridge source.
3. Keep `Card.ALL` (q→end) for every removal metric. The choice is now backed by data rather than by default.
4. `swap_check.py` is superseded by `swap_np.py`. Its `sc(54)` range bug is not fixed (the script wasn't touched).

**Caveat:** this rests on one question for removal and four for swaps, with only 4 alternatives each. The 4B was not tested.
