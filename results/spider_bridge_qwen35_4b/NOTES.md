# Is "spider" a real bridge? (Qwen3.5-4B)

Script: `src/consulted_bridges/legacy/spider_bridge.py`. Full output: `run.log`; table: `edits.csv`.
**Prompt:** "How many legs do the animals that spin webs have? Answer in one word." **Baseline:** "Eight", P(Eight or 8) = 0.89.

**Question:** does the model consult "spider" in the middle layers to decide the answer?

## 1. Is spider needed? Yes, in layers ~9–23
Removing the spider direction, 6 layers at a time, at every position from the question on:

| Layers | 0–5 | 3–8 | 6–11 | 9–14 | 12–17 | 15–20 | 18–23 | 21–26 | 24–29 | 27–31 |
|---|---|---|---|---|---|---|---|---|---|---|
| Answer | Eight | Eight | Eight | **Four** | **Four** | **Four** | **Four** | **Six** | Four | Eight |
| P(8) | 0.81 | 0.71 | 0.42 | 0.12 | 0.01 | 0.00 | 0.00 | 0.03 | 0.40 | 0.91 |

**Controls:** removing other words that are about as strongly present (animals, legs, humans, dogs, sentence, word, Answer, ways…) left the answer at Eight (P(8) 0.80–0.96).
- **Exceptions:** "webs" and "insects", which are on the path to spider. Their directions also overlap with spider's.

## 2. Does the answer follow what replaces spider? Partly
Swapping spider for another animal (middle layers 12–21):

| Swapped in | Expected legs | Answer | P(expected) |
|---|---|---|---|
| octopus (control: also 8) | 8 | **Eight** | 0.55 |
| bee | 6 | **Six** | 0.89 |
| dog | 4 | **Four** | 0.97 |
| cat | 4 | **Four** | 0.97 |
| ant | 6 | Four | 0.29 |
| bird | 2 | Four | 0.15 |
| human | 2 | Eight | 0.00 |
| snake, fish | 0 | Eight | ≈0 |
| crab | 10 | Four | 0.00 |

- **Matches:** the answer matched the new animal in 12 of 30 swaps (over 3 layer ranges).
- **Caveat on dog and cat:** "Four" is also what the model says when spider is simply removed. So those matches don't prove it read "dog".
- **The clean positives are bee → Six and octopus → Eight.** In most other swaps the model just loses spider and falls back to Four.

## 3. Does any swap just plant the new word? No
Swapping 13 other strongly present words for "bird" never gave "2" (P ≤ 0.09). The exception is "insects" → bird, which gives Four, the same as removing insects. So unlike Mozart's Vienna → Budapest, the swaps here don't blindly plant.

## 4. Where in the prompt? After the question
- **Question's own words:** editing spider there barely matters (mostly still Eight).
- **Template tokens after it** (the line break and the assistant-start tokens, where "spiders" ranks first): editing spider there breaks the answer.
- **So the model works out spider after reading the question,** then uses it.

## Bottom line
- **Spider is a real, needed bridge:** removing it in the middle layers, after the question, breaks "eight", and matched controls don't.
- **Its identity is only partly what the answer reads off:** a swapped-in animal sets the count for bee and octopus, but usually the model just falls back to "Four".
- **Caveats:**
  - One prompt only; our own definitions of remove and swap.
  - The "spider rank after" column in `edits.csv` stays near 1. That's because it takes the best rank over many layers and positions, and spider comes back in layers we didn't edit. Read it as "spider is recomputed later", not "the edit failed".
