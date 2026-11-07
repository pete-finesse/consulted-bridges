# Spider legs on Qwen3.5-4B

Script: `src/consulted_bridges/legacy/spider_legs.py`. Full output: `run.log`; tables: `edits.csv`, `tracked_words.csv`, `discovery_top_words.csv`.

**Prompt** (chat, no prefill): "How many legs do the animals that spin webs have? Answer in one word."
**Baseline:** the model answers "Eight". P(Eight) = 0.73, P(8) = 0.16, P(Four) = 0.07.

## What is in J-space
- **"spiders" ranks first** at the newline right after the user's message, from layer 7 to layer 25. The word "spider" is never in the prompt.
- **"eight" appears later:** at the assistant-start tokens, layers 28–30.
- So the order is webs → spider (middle layers) → eight (last layers).
- "arachnid" is weak (best rank 180 at the last position), as are "nest" and "ant".

## Edits (spider/spiders never edited; positions = the question onward)
Layer ranges: early 4–13, mid 12–21, late 20–29, all 0–31.

| Edit | Answer | P(Eight) | "spider" rank after (last position) |
|---|---|---|---|
| none | Eight | 0.73 | 1 |
| swap eight→six, late | **Six** | 0.12 | 1 |
| swap eight→four, late | **Four** | 0.07 | 1 |
| swap eight→two, late | **Two** | 0.02 | 1 |
| swap eight→ten, late | **Ten** | 0.03 | 1 |
| same swaps, early or mid | Eight | 0.58–0.78 | 1 |
| remove eight, late | **4** | ≈0 | 3 |
| remove web/webs, mid | **Four** | 0.21 | 7 (weakened) |
| swap web→nest (any) | Eight | 0.63–0.74 | 1–3 |
| swap arachnid→insect (any) | Eight | 0.52–0.70 | 1–2 |
| swap legs→wings (any) | Eight | 0.67–0.75 | 1 |
| push toward "insect", mid, small (0.05) | **Six** | 0.18 | 4 |
| push toward "bird", mid, small (0.05) | **2** | ≈0 | 104 (spider pushed out) |
| larger pushes (0.1–0.2) | anything / nonsense ("Buzzard") | ≈0 | often 100s–1000s |

The answer changed away from eight in 37 of the 78 edits.

## Reading
1. **Cleanest result:** in the late layers, swapping the number "eight" for another number gives that number as the answer, while "spider" stays the top word. The answer comes from a separate number slot computed after "spider". It isn't read straight off "spider".
2. **Timing matters:** the same swap in early or middle layers does nothing, because "eight" isn't computed there yet.
3. **The upstream route also works, but goes through spider:** removing "web" in the middle layers weakens "spider" (rank 1 → 7), and the model falls back to "Four", a generic animal.
4. **Steering is blunt:** small pushes give sensible answers (insect → Six, bird → 2), but bigger ones knock "spider" out or break the output.
5. **Why it matters for Silent Dissent:** a middle step (here, spider) can be fully present in J-space while the model gives a different answer. So "the bridge is present" doesn't show the bridge is what decides the answer.

## Caveats
- **One prompt only.** Our ablate and swap are defined in `src/consulted_bridges/interventions.py` and may differ from Neuronpedia's.
- **"Spider untouched" is approximate.** The edited words' J-lens directions are not perpendicular to spider's (largest cosine 0.33 for eight, 0.56 for web, 0.61 for insect). The "spider rank after" column is the real check.
