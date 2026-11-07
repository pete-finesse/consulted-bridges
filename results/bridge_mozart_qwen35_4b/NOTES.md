# Is "Salzburg" a real bridge? Mozart vs spider (Qwen3.5-4B)

Script: `src/consulted_bridges/legacy/bridge_test.py --spec mozart` (and `--spec mozart_direct`). The same tests as the spider run (`../spider_bridge_qwen35_4b/NOTES.md`). Full output: `run.log`, `edits.csv`; the direct-question run is in `../bridge_mozart_direct_qwen35_4b/`.

## Prompts (chat, no prefill, one-word answer)

| Name | User message | Baseline |
|---|---|---|
| **two-hop** | "In which country is the city where Wolfgang Amadeus Mozart was born? Answer in one word." | **Austria**, but only P = 0.46 (Italy 0.23, Germany 0.23) |
| **direct** (shortcut check) | "Which country was Wolfgang Amadeus Mozart from? Answer in one word." | **Austria**, P = 0.77 |

"Salzburg" ranks first in J-space in both, so it shows up even when the question never asks about a city.

## 1. Removal: removing Salzburg kills "Austria", but removing Vienna does too

P(Austria) after removing a word, 6-layer windows, two-hop prompt:

| Removed | L0–5 | L6–11 | L12–17 | L18–23 | L24–29 |
|---|---|---|---|---|---|
| **Salzburg** | 0.27 | 0.00 | 0.00 | 0.00 | 0.05 |
| **Vienna** (another Austrian city) | 0.26 | 0.04 | 0.01 | 0.01 | 0.03 |
| Wolfgang (in the question) | 0.30 | 0.07 | 0.04 | 0.11 | 0.45 |
| Graz, Innsbruck (weakly present) | 0.4 | 0.2–0.4 | 0.2–0.4 | 0.2–0.4 | 0.3–0.5 |
| Tokyo, music, composer, city, town… | ≈0.45 | ≈0.45 | ≈0.45 | ≈0.45 | ≈0.45 |

The direct prompt gives the same picture: Salzburg 0.00–0.02, Vienna 0.04–0.15, the other controls 0.6–0.8.

**Why Vienna does the same:** the J-lens directions overlap.
- Salzburg vs Vienna: cosine 0.60.
- Salzburg vs **Austria itself**: 0.63.
- Vienna vs Austria: 0.72.

So "remove Salzburg" also removes much of "Austria". For spider, the matching overlaps were smaller (spider vs octopus 0.35, vs dog 0.41), apart from insects (0.60).

## 2. Swaps: they don't follow the new city's country

| Swap Salzburg for | Should give | Two-hop prompt | Direct prompt |
|---|---|---|---|
| Munich | Germany | **Germany** (0.48–0.84) | **Germany** (all layers) |
| Milan | Italy | **Italy** (0.9) | **Italy** (0.8–0.9) |
| Paris | France | Italy (France only with all layers) | (not run) |
| Budapest, Zurich, Prague, Krakow, Madrid, Osaka | their countries | Italy or Austria | Austria (Budapest) |
| **Innsbruck, Graz, Linz** (Austrian, should stay Austria) | Austria | **Italy** | **Austria** (Innsbruck) |

- **The controls fail in the two-hop prompt:** swapping in another *Austrian* city still breaks "Austria".
- **The two "hits" are the runner-ups:** Germany and Italy are the model's second choices anyway (0.23 each). Like "Four" for spider, they're the fallback, not a sign that the model read "Munich".

## 3. Planting check

Swapping 16 controls for Budapest never gave Hungary. That rules out blind planting, but Salzburg → Budapest never gave Hungary either.

## 4. Where in the prompt: the opposite of spider

- **Mozart:** removing Salzburg on the **question's own words** kills "Austria" (P = 0.00), in both prompts. Removing it on the template tokens after the question hurts less in the direct prompt (0.65 in the middle layers).
- **Spider:** it was the other way round. "Spider" mattered after the question, where the model works it out.

## Bottom line

**We can't make the spider claim for Salzburg.**
1. **Removing another Austrian city** (Vienna) has the same effect as removing Salzburg.
2. **Swapping in other Austrian cities** breaks "Austria" instead of keeping it.
3. **The direct question** ("which country was Mozart from?") depends on "Salzburg" just as much, though no city is asked about.
4. **The "Salzburg" effect sits on the question's own words** (Mozart's name), not after them.

**Best reading:** the J-lens "Salzburg" direction here is largely "Austria / Austrian place", stored with Mozart. It isn't a separate step the model computes and then consults. Spider passed the equivalent checks: its controls didn't break the answer, and it mattered after the question.

## Caveats and next step

- **One fact, one model;** our own definitions of remove and swap.
- **Mozart is famous, so a direct Mozart → Austria link is likely.** Less famous people may behave differently.
- **Next step:** remove only the part of "Salzburg" that is *not* shared with "Austria" (project the Austria direction out first), and see whether that still hurts the answer. Then repeat over many facts.
