# Review of the bridge scorecard

*Independent review. I checked `docs/scorecard/v1.md`, `src/consulted_bridges/legacy/bridge_card.py`, `bridge_specs.py`, `paper_set.py` and `src/consulted_bridges/` (interventions, text, readout). I recomputed numbers from the CSVs and ran new checks on the pod (Qwen3.5-4B, same J-lens, same code version: md5 of the five main files matches the local copy). My throwaway scripts and their raw output are in the session scratchpad (`review/review_checks.py`, `know_checks.py`, `legs_checks.py`, `review/pod_out/*.json|log`) and on the pod in `/workspace/review_tmp/`. No project file was changed.*

## Terms used below

- **Residual stream:** the vector the model carries at each token position from one layer to the next. Every edit here changes it.
- **J-lens direction of a word:** one fixed vector per layer such that the J-lens score for the word is the dot product with it. The code uses the first token piece of " word" (e.g. " Salzburg"; for " Funchal" it is just " F").
- **Remove (ablate):** subtract the residual's component along that direction, at chosen positions and layers.
- **Window:** a block of 6 consecutive layers (L0–5, L3–8, …, L27–31; 10 windows). The card reports the best one.
- **Drop:** 1 − P(right answer after the edit) / P(before). 1 = the answer is gone.
- **Answer onset:** the first window in which removing the *answer's own* direction gives a drop of 0.5 or more. Before that layer the answer is not yet written along its J-lens direction.
- **One-hop prompt:** a question whose answer *is* the bridge ("In which city was Yukio Mishima born?" → Tokyo).
- **Null word:** a random everyday word with no link to the question (piano, river, window, engine, castle, carpet, pencil). Its drop shows how much an edit does by chance.
- **Fragile prompt:** a prompt where the right answer's baseline probability is low (about 0.5). A small push already looks like a big drop there.

## 1. Verdict

1. The edit maths and the plumbing are correct: positions, answer-probability sums, the shard merge and the unique-part projection all check out. The bugs I found are small and flip no reported verdict on their own. But three round-2 cards test meaningless token pieces (" F", " G", " Bel"). And 5 of the 17 dataset picks fail the paper's own known-fact screen.
2. **"Spider is a consulted bridge" mostly holds.** Removing spider kills the answer in L9–26, well before the answer's own direction matters (L21+). Unrelated questions are untouched, and null words do at most 0.04. But its K pass is borderline: 0.50 at a 0.5 threshold. A clean 8-legged swap (tick) also fails.
3. **"Salzburg is an answer echo" is plausible but not shown.** U can't separate an echo from a bridge that is consulted through its country attribute. K = 0 comes with swaps that rarely work at all (F_hit 0.14–0.30). And D = 0.23 is within the null-word noise of this fragile prompt (up to 0.26).
4. **"None of the paper's set is consulted / the model follows the shortcut" holds only for a few facts.** It is supported for Mishima, Tsinghua and partly Federer, and behaviourally for the two trap questions. For country bridges (all landmark questions) the edit is blind before L18 even in a one-hop question. And a 16-layer removal reveals effects that 6-layer windows hide (Kremlin 0.65, Colosseum 0.50).
5. **Two of the three round-2 "echoes" are not echoes by the evidence.** Christ the Redeemer's N = 1.00 comes only from late windows (L21–31), where the answer is being written. CN Tower breaks under every edit: null words reach 0.35, other countries 0.72, and all swaps give "other".

## 2. Code bugs and data issues

Ordered by how much they could change a number.

| # | Where | What's wrong | Evidence | Changes a reported number? |
|---|---|---|---|---|
| B1 | `paper_set.py:45-52` (`CITIES`, picks) via `text.word_tokens` → `probe_tokens` | **Bridges whose first piece is a generic fragment.** Funchal → `" F"`, Gori → `" G"`, Belgrade → `" Bel"`. Fordham's bridge is entered as `"York"`. These are not the city's direction. Their "rank" (Ronaldo 2, Stalin 15, Djokovic 17) and their removal mean nothing. The same happens to controls and swaps: octopus `" oct"`, scorpion `" sc"`, crabs `" cr"`, giraffe `" gir"`, Innsbruck `" Inns"`, Linz `" Lin"`, Niš `" Ni"`, Kragujevac `" K"`, Tbilisi `" T"`, Yokohama `" Yok"`, Ithaca `" I"`. Apple's swap "blueberry" becomes `" blue"`, which is the colour itself. | tokenizer dump (`review/tok_check.py`) | Yes in meaning: Djokovic, Ronaldo and Stalin should be "untestable", not "bystander" (3 of 13 bystanders). Spider's scorpion swap and crabs same-kind are partly artefacts. |
| B2 | `paper_set.py` picks vs `results/known_qwen35_4b.csv` | **5 of the 17 dataset picks are not "known" by the paper's screen:** Keanu (hop 1 → "Los Angeles"), Bieber (hop 2 fails), Che (hop 1 → "Argentina", not a city), Fordham ("New York" vs "New York City"), Harvard (hop 2 fails). The scorecard cites the screen as if picks came from it. In chat format I checked: Keanu → Beirut and Beirut → Lebanon are known. Che's hop 1 still answers "Argentina", so Che's bridge is unverified. | `known_qwen35_4b.csv`; `pod_out/runK.log` | Che's "unclear" rests on an unverified first hop. The trap-question claim survives (see §5). |
| B3 | `paper_set.py:58` | **The USA answer direction is " America"**, but the model writes "United States". Removing " America" barely moves P(United States): the best window gives 0.31 (Fordham) and 0.08 (Harvard). So O and U for those two use the wrong answer direction. | `out_AB.json` → B, `answer` profiles | No verdict change (N = 0 anyway). O and U for Fordham and Harvard are not meaningful. |
| B4 | `bridge_card.py:147` vs `:153` | **N and U use different word sets.** N removes all bridge words (spider + spiders, apple + apples). U removes only `bridge[0]` minus the answer. | Recomputed with both words' unique parts: apple U 0.016 → 0.055, spider 0.99 → 0.997, elephant 0.009 → 0.016 | No verdict change. Fix for consistency. |
| B5 | `bridge_specs.py:22` (spider swaps) | **Expected answers are not the model's beliefs.** The spec says crab → "10", but the model says crabs have **Eight** legs. Both crab swaps gave 8. They were scored as failures to move (F, F_hit), but by the model's belief they are same-answer keeps (K). The model does know tick, scorpion and octopus have 8, and that Graz, Innsbruck and Linz are in Austria. | `pod_out/runL.log` | Spider K 0.50 → 0.67, F_hit 0.40 → 0.50 if scored against the model's beliefs. No verdict change. |
| B6 | `bridge_card.py:123` → `readout.Readout.summary` | **`bridge_rank` is the best rank over all 32 layers** (including layer 0 and the output layer) and all positions. The band `clip(range(12,30))` passed in is used only for band means, not for the best rank. | `readout.py:summary` (`ra.min()` over every layer) | The "rank 0–3" claims are optimistic minima over about 900 cells. That doesn't change the scorecard's point (rank is not enough). |
| B7 | `bridge_card.py:96-101` (`classify`) | **Prefix matching has false positives:** "USSR" → USA (form "US"), "Austria-Hungary" → Austria, "Eighty" → 8, "Sixteen" → 6. The feared cases are safe: Austria vs Australia, United States vs United Kingdom, New Delhi vs New York. "the United States" → "other", which would make a baseline NOT KNOWN. Raw swap outputs are not saved, so past swaps can't be audited. | `classify` replica on edge cases | None found in the saved data, but it can't be checked. Use whole-word matching and save the raw text. |
| B8 | `bridge_card.py:114-115` | **The best window isn't recorded,** so you can't see which layers produce N. That hides the late-window effect (§3, M2). | — | Reporting gap. |

**Checked and correct.**
- `ablate` projects out the span of the piece directions.
- `ablate_excl` removes only the part orthogonal to the excluded span: the excluded component is preserved to 1e-7 (`review/math_check.py`).
- `swap` exchanges the two least-squares coefficients and leaves the orthogonal complement untouched.
- `from:` and `span:` resolve to the intended tokens on the real Qwen3.5 chat prompt. `from:<|im_end|>` gives the 9 template tokens `<|im_end|> … </think> \n\n`. The question span ends at "word.".
- Edits act on prompt positions both when scoring with appended answer tokens and during greedy generation.
- `P` correctly sums the `p[{}]` and `p[**{}]` keys. The library's own `p_full` still ignores them (it sums keys starting with `p_`), but the scripts don't use it.
- No answer form is a token prefix of another, so there's no double counting: "US" and "USA" are separate tokens.
- The three shards merge into `cards_all.csv` exactly (21 rows, identical values).
- The "76% of the 545 known birth-city facts have the capital as bridge" claim reproduces (76.3%). Tokyo appears 87 times, not 91.

## 3. Method concerns, ranked by how much they could change a conclusion

### M1. No check that the edit can see the bridge at all (decides 13 of the 17 round-2 verdicts)

"Bystander" is read as "the model doesn't use it". It can also mean that removing this one direction, in these 6 layers, doesn't remove the bridge's information. I tested this in two ways.

- **One-hop validity:** remove the bridge direction in the one-hop prompt whose answer *is* the bridge.
  - City bridges: removing Tokyo, Salzburg, Basel or Beijing in windows up to L15–20 kills that city as the one-hop answer (best drops 0.92, 1.00, 0.61, 0.75). So the edit can see these.
  - **Country bridges** (Italy, Russia, India, Canada, Brazil, France) and Montreal and Cambridge: removing them in any window up to L15–20 does **nothing** (≤ 0.05; Cambridge 0.18). That holds even when the country *is* the answer. Their direction only matters from L18–21 on, where the answer is being written.
  - In a two-hop question the bridge must be used *before* the answer is written. So for every landmark question the test is blind exactly where the bridge would act.
- **Answer positive control:** remove the *answer's own* direction in the two-hop prompt. For Mishima, Tsinghua, Harvard, Fordham and elephant, even that does almost nothing before L21 (best window ending ≤ L23: 0.01, 0.03, 0.08, 0.31, 0.09). So these directions are not where the model carries the answer in the middle layers.

**What it changes.** "Bystander" is supported where one-hop validity passes and N ≈ 0. That's **Mishima** (strongest: in the one-hop prompt, removing Tokyo already hurts at L0–5 (0.36) and gives 0.70–0.92 in L6–20, yet Japan is untouched in the two-hop prompt), **Tsinghua** and **Federer** (partly: N = 0.43 with all layers). It is **untestable** for the 5 landmark questions (Colosseum, Kremlin, Taj Mahal, and in mid layers CN Tower and Christ the Redeemer), McGill and Harvard, and the 3 broken-piece cards (B1).

**What would resolve it:** criterion V below. Also try a richer edit than one direction, e.g. remove the span of the bridge's directions at all layers' J-lenses, or patch in the residual from a prompt with a different subject.

### M2. "Best of 10 windows" mixes layers where the bridge is used with layers where the answer is written

Removing the bridge in late windows also removes whatever part of the answer's output its direction overlaps. That's an output effect, not consultation.

- **Christ the Redeemer:** N = 1.00 comes from L21–31 only. The windows ending by L23 give at most 0.62, while the answer's own direction already matters from L9. Restricted to layers before the answer is written, it would be **unclear**, not an echo.
- **Che:** N = 0.66 comes from L21–29. The early windows give at most 0.40, so it would be a bystander.
- **Apple:** the bridge acts in L15–23, but "red"'s own direction already matters from L9. Apple acts after the answer is underway, which fits echo, not consultation.
- **Spider** is clean: the bridge acts from L9, the answer's direction only from L21.

**Multiple comparisons.** Taking the max over 10 windows inflates every drop. The null-word maxima show the noise floor per prompt:

| Prompt | Null-word best drop |
|---|---|
| spider | ≤ 0.04 |
| Christ the Redeemer | ≤ 0.12 |
| apple | ≤ 0.18 |
| Mozart | ≤ 0.26 |
| Che | ≤ 0.26 |
| CN Tower | ≤ 0.35 |

On fragile prompts, gaps of 0.2–0.3 (D's threshold is 0.3) are inside the noise. S and D take a max over 60 or 30 control cells against the bridge's 10, which biases them *against* the bridge (conservative). U and N are each a max of 10 (biased up).

**What would resolve it:** report the full window profile. Score N only in windows that end before the answer onset (criterion N_pre below). Compare every drop to a null floor (Z).

### M3. Six-layer windows miss representations spread over many layers

| Fact | Best 6-layer window | 16 layers (L8–23) | All 32 layers |
|---|---|---|---|
| Kremlin | 0.13 | **0.65** | 0.54 |
| Colosseum | 0.01 | 0.16 | **0.50** |
| Federer | 0.05 | 0.14 | 0.43 |
| Taj Mahal | 0.03 | 0.05 | 0.19 |

Kremlin would move from bystander to unclear. The model can rebuild the bridge after a short window because the subject's tokens are left intact.

**What would resolve it:** add the 16-layer and all-layer drops to the card as R. A low window drop with a high R means "spread out", not "bystander".

### M4. The echo criteria (U, D, K) don't isolate "echo"

**U.** U is high whenever the bridge's direction has little overlap with the answer's, almost regardless of use.
- Random word pairs have |cos| of 0.15–0.20 at L6–18. Spider–eight at 0.22 is at chance, so "spider minus eight" ≈ spider and U ≈ N automatically.
- For a city → country hop, the bridge's useful attribute *is* its country. A bridge consulted through its "Austrian" feature would fail U just like an echo.
- So low U says the effect runs through the shared part. It doesn't say the bridge wasn't consulted.
- Note too that Mozart **passes** U (0.58 ≥ 0.5). Its echo label rests on D (0.23 < 0.3) and K. The text "its own part does little" overstates this: Salzburg-minus-Austria drops P by 0.25 absolute, Vienna-minus-Austria by 0.14.

**K.** Low K is predicted when swaps don't work at all, not only by echo.
- Mozart's swaps mostly return the fallback "Italy" (F_hit 0.14–0.30), even though the model knows Graz, Innsbruck and Linz are Austrian. So K = 0 there is "the swap inserts nothing usable", which is not specific to echo.
- K also passes trivially when the edit does nothing (every bystander has K = 1).
- With 2–4 swap cells it moves in steps of 0.25–0.5. Spider's K is 0.50 with {octopus, scorpion}, 0.67 if crab is scored by the model's belief, and 0.50 with tick (both tick swaps → "Four").

**D.** D depends on hand-picked same-kind words, some of them broken pieces (B1). Its 0.3 threshold sits exactly between the two calibration examples (spider 0.40, Mozart 0.23).

### M5. The automatic controls behind S are on-topic, not unrelated

They come from the prompt's own J-lens top list, so by construction they are about the question. Examples: "Musik" (drop 0.61) and "Cities" (0.35) for Mozart; "столиц" ("capital", 0.91) and "Bogotá" (0.40) for CN Tower; "Brasília" (0.64) for Taj Mahal; "famously" and "iconic" (0.41, 0.37) for Che. A cosine below 0.4 excludes only 2–5% of random word pairs, so it doesn't make a word unrelated.

**Fix:** take controls from a fixed list of null words matched on strength, or from another fact's bridge (the `random` role in `src/consulted_bridges/silent_dissent/facts.py`).

### M6. T saturates and uses a different edit from N

T is the difference of two 16-layer drops. For Mozart both are near 1 (after 0.94, question 1.00), so T = −0.06 "fails" from saturation. The data don't support "not after them": removing Salzburg only after the question also kills the answer.

There's also a leak in both directions. Information can move to the template tokens before L8. And question tokens at L24+ are untouched. So "after" vs "on the question" isn't cleanly separated.

### M7. Thresholds tuned on the two examples they then classify

Round 1's spider and Mozart verdicts are true by construction, not evidence. Round 2 is the first real test. There, 13 of 17 cards were decided by N alone, which M1–M3 show is the weakest link.

### M8. Small, uneven sample

There are 17 scored facts, and many groups have 2 picks. The B1 conclusion ("76% of the dataset is capital-bridge, all bystanders") rests on Mishima and Djokovic, and Djokovic's card is invalid (B1). Fragile prompts (Mozart 0.50, CN Tower 0.52, Stalin 0.49) dominate the interesting verdicts.

### Ruled out: general damage

Every bridge and answer edit I tried (spider, Salzburg, Brazil, Canada, apple, Rosario, Austria, Ottawa, Brasília) leaves two unrelated questions intact: gold → Au and largest planet → Jupiter, max drop 0.06. So high N values are not the model being broken in general. CN Tower's fragility is specific to that prompt.

## 4. Criteria: keep, drop, change, add

| | Now | Recommendation |
|---|---|---|
| **N** | best of 10 six-layer windows | **Change to N_pre** (below). Report the full window profile and R. |
| **S** | N minus the max drop of on-topic J-lens words | **Change:** use null words or other facts' bridges matched on strength, and subtract the null floor Z instead of a max over on-topic words. |
| **O** | cosine to the answer | **Keep** as reported. Also print the random-pair baseline (|cos| about 0.15–0.17 at L12–18). |
| **U** | unique part, `bridge[0]` only | **Keep as descriptive, not verdict-deciding.** Compute it on the same word set as N. Read low U as "acts through the answer-shared subspace", not as "echo". |
| **D** | U minus the max same-kind U | **Change:** use single-token same-kind words the model knows, and compare to Z. Use it only when U ≥ 0.5. |
| **T** | after minus question, 16 layers | **Drop** in favour of P (precedence). If kept, report both raw drops and score them only when neither is above 0.9. |
| **K** | same-answer swaps that keep the answer | **Score only when SV ≥ 0.5;** otherwise report "swaps uninformative". Use ≥ 4 single-token, model-verified words. |
| **F** | swaps that move the answer | **Merge into SV.** |

**New criteria with exact definitions.** Let W be the 10 windows. dropₓ(w) is the drop when removing direction x in window w at positions `from:q_start`.

- **A, answer control:** A = max over w of drop_answer(w), where the answer's directions are the pieces of the form the model actually writes (" United", " States" for USA, plus the ' Word' piece). **Answer onset L\*** = the first layer of the first window with drop_answer ≥ 0.5. If A < 0.5, the card is **untestable**.
- **V, edit validity (one-hop):** in the one-hop prompt whose answer is the bridge, V = max over windows that end before L\* of drop_bridge(w) on P(bridge). Require **V ≥ 0.5**, else **untestable**. Here V passes for spider, Salzburg, Tokyo, Basel and Beijing, and fails for every country bridge, Montreal and Cambridge.
- **N_pre, necessity before the answer:** N_pre = max of drop_bridge(w) over windows that end before L\*. It replaces N in the verdict rules.
- **P, precedence:** the bridge onset (first window start with drop_bridge ≥ 0.5) must come at least 3 layers before L\*. Here only spider passes (bridge L9, answer L21). Mozart (3 vs 0), CN Tower (0 vs 0), Christ (12 vs 9), apple (15 vs 9) and Che (21 vs 3) fail.
- **Z, null floor:** the 95th percentile of the best-window drops for ≥ 10 null words (single-token, matched on strength to the bridge at mid layers). Require N_pre − Z ≥ 0.5, and compare D and S against Z.
- **X, unrelated damage:** the max drop on 2 unrelated questions under the same edit. Require X ≤ 0.1. All edits here passed.
- **R, redundancy:** the drop with L8–23 and with all layers removed. If N_pre < 0.5 but R ≥ 0.5, the verdict is **spread out**, not bystander.
- **SV, swap validity:** the share of different-answer swaps that land on the swap word's own answer, as the model states it in a one-hop check ("How many legs does a crab have?"), excluding the fallback. Interpret K only when SV ≥ 0.5.
- **Q, token quality:** the bridge (and every control or swap word) must be a single token when written with a leading space, or be skipped. That's a hard gate.
- **Dose response (optional):** remove a fraction α ∈ {0.25, 0.5, 0.75, 1} of the component. The drop should rise with α. A cliff at α = 1 alone suggests an off-distribution artefact.

**Suggested verdict rules:**
- Untestable if A < 0.5, V < 0.5 or Q fails.
- Consulted if N_pre − Z ≥ 0.5, P holds and X ≤ 0.1, with U ≥ 0.5 and SV/K as supporting evidence.
- Shared-component effect (instead of "echo") if N_pre − Z ≥ 0.5 but P fails, or U < 0.5.
- Spread out if R ≥ 0.5 and N_pre < 0.5.
- Bystander if R < 0.2 and V ≥ 0.5.

## 5. Which stated conclusions hold

| Claim in docs/scorecard/v1.md | Status | Why |
|---|---|---|
| Spider is a consulted bridge | **Holds, with caveats** | Acts in L9–26 before the answer onset (L21), N_pre = 1.00, null ≤ 0.04, unrelated damage ≤ 0.06, one-hop V = 0.68, bee → Six on both swaps. Caveats: K borderline (tick fails), one wording, thresholds tuned on it. "Works it out after reading" is one reading of T. Another: spider-ness read early and held at the final positions. |
| Salzburg is an answer echo | **Weakened** | What's shown: the Salzburg and Austria directions both become necessary at the same early layers (L3–8), and Vienna does nearly the same. So the shared "Austrian place" part carries the effect. What isn't shown: that the model doesn't consult Salzburg. U can't separate the two, K is confounded by swaps that don't work, and D is inside the noise. "Acts on the question's words, not after them" is contradicted by T_after = 0.94. |
| Mozart direct: an echo | **Weakened** (same reasons) | U = 0.24 is cleaner here, but P fails as above. |
| Apple acts like an echo | **Holds** | Its bridge acts only after red's own direction is already needed (L15 vs L9), and U ≈ 0 with either word set. |
| Elephant and Eiffel are bystanders | **Eiffel untestable; elephant uninformative** | France can't be removed even as a one-hop answer before L18 (V ≈ 0). For elephant, even removing the answer's own direction barely matters (A = 0.27). |
| Round 2: none of the paper's set is consulted | **Unsupported as stated** | Of the 17: 3 have broken pieces, 7 are untestable by V, and 2 (Kremlin, Colosseum) show 0.5–0.65 drops when removed over 16+ layers. Only Mishima, Tsinghua and partly Federer are genuine bystanders by these checks. Better: "no fact passes; most can't be tested with this edit". |
| Three round-2 echoes (Mozart, CN Tower, Christ) | **Only Mozart survives** (as a shared-component effect) | Christ: N comes only from late windows (N_pre = 0.62), so unclear. CN Tower: every edit breaks it (null words 0.35, other countries 0.72, all swaps → "other", identical bridge and answer profiles), so non-specific fragility. |
| The trap questions show the shortcut directly | **Holds behaviourally** | I checked both hops in the same chat format: Australia's capital → Canberra and Sydney Opera House → Australia, yet the composed question → Sydney. Keanu → Beirut and Beirut → Lebanon, yet → Canada, which is his nationality. This shows the model doesn't *compose* here; it doesn't show the mechanism. The scorecard didn't run these hop checks, and the paper's screen marks Keanu's hop 1 as failed. |
| 76% of the 545 known birth-city facts have a capital bridge | **Holds** (76.3%; Tokyo 87, not 91) | "That's group B1, all bystanders" rests on 2 cards, one invalid (Djokovic " Bel"). |
| Ranking first in J-space is not enough | **Holds** | Mishima ranks 0 and passes V, yet removal does nothing. |
| A presence readout can't tell consulted from echo from bystander | **Holds** | But the scorecard itself still can't reliably tell echo from attribute use (M4), or bystander from "the edit can't see it" (M1). |
| K, F and T carry no information when N ≈ 0 | **Holds** | Also true when swaps don't work (SV low). |

**What you would need to believe the round-2 headline:** the J-lens direction is where the model carries the bridge in the layers before the answer is written. The one-hop check shows that's true for some city bridges (Tokyo, Basel, Beijing, Salzburg) and false for country bridges. So the headline can be made for capital-city birth facts like Mishima, but only after redoing the cards with V, N_pre, Z and R on a larger, token-clean sample.

## Appendix: key numbers from my checks (pod)

Drop per window, two-hop prompt. "Ans onset" is the first window start where removing the answer's own direction gives ≥ 0.5. "One-hop V" is the best drop on P(bridge) in the one-hop prompt, windows up to L15–20.

| Sentence | N (card) | N, windows ≤ L23 | L8–23 | All layers | Bridge onset | Ans onset | One-hop V |
|---|---|---|---|---|---|---|---|
| spider | 1.00 | 1.00 | 1.00 | 1.00 | L9 | L21 | 0.68 |
| mozart | 1.00 | 1.00 | 1.00 | 1.00 | L3 | L0 | 1.00 |
| mozart_direct | 1.00 | 1.00 | 1.00 | 1.00 | L3 | L3 | — |
| apple | 0.60 | 0.60 | 0.62 | 0.48 | L15 | L9 | 0.87 |
| elephant | 0.00 | 0.00 | 0.00 | 0.00 | — | — (A = 0.27) | 0.15 |
| eiffel | 0.00 | 0.00 | 0.02 | 0.02 | — | L18 | 0.00 |
| Colosseum | 0.01 | 0.01 | 0.16 | 0.50 | — | L18 | 0.01 |
| Kremlin | 0.13 | 0.04 | 0.65 | 0.54 | — | L12 | 0.00 |
| CN Tower | 0.99 | 0.99 | 1.00 | 1.00 | L0 | L0 | 0.00 |
| Christ Redeemer | 1.00 | 0.62 | 0.75 | 1.00 | L12 | L9 | 0.00 |
| Taj Mahal | 0.03 | 0.03 | 0.05 | 0.19 | — | L18 | 0.00 |
| Mishima | 0.00 | 0.00 | 0.00 | 0.00 | — | L24 | 0.92 |
| Djokovic (" Bel") | 0.00 | 0.00 | 0.00 | 0.00 | — | L6 | — |
| Federer | 0.05 | 0.05 | 0.14 | 0.43 | — | L18 | 0.61 |
| Ronaldo (" F") | 0.02 | 0.02 | 0.03 | 0.02 | — | L0 | — |
| Stalin (" G") | 0.16 | 0.02 | 0.00 | 0.01 | — | L6 | — |
| Bieber | 0.00 | 0.00 | 0.00 | 0.00 | — | L15 | — |
| Che | 0.66 | 0.40 | 0.48 | 0.56 | L21 | L3 | (one-hop base 0.01) |
| Tsinghua | 0.01 | 0.01 | 0.06 | 0.02 | — | L21 | 0.75 |
| Fordham (" York") | 0.00 | 0.00 | 0.00 | 0.00 | — | L21 (" America") | — |
| McGill | 0.00 | 0.00 | 0.19 | 0.04 | — | L18 | 0.05 |
| Harvard | 0.00 | 0.00 | 0.00 | 0.00 | — (A = 0.17, " America") | — | 0.18 |

**J-lens geometry:** mean |cos| between 300 random word directions is 0.20 at L6, 0.17 at L12, 0.15 at L18 and 0.08 at L24 (95th percentile 0.41, 0.35, 0.30 and 0.16). The share of pairs above 0.4 is 5.5%, 2.1%, 0.6% and 0%.

**Spider swaps** (mid 12–21 / L8–23):

| Swap word | Result |
|---|---|
| tick | Four / Four |
| octopus | Eight / Eight |
| scorpion | Four / Six |
| bee | Six / Six |
| dog | Four / Four |

**Model beliefs (chat, one word):**
- Leg counts: tick, scorpion, octopus, crab and spider all → Eight; bee and ant → Six; snake → Zero; bird → Two.
- Graz, Innsbruck, Linz and Salzburg → Austria.
