# Proposed questions from the scorecard's trends

*Proposals only: nothing here has been run. Specs: `src/consulted_bridges/questions/proposed/trend.py` (23 questions, ids `T_…`). Data: `src/consulted_bridges/bank.py` → `results.csv` and `tests.csv` (62 questions × 2 models).*

## Terms used below

- **Bridge:** the middle word the model has to think of but never writes ("Italy" in "the country where the Colosseum is").
- **The edit:** removing the bridge word's J-lens direction (its "meaning direction" inside the model) in a few layers.
- **Gates:** the checks a question must pass before it is scored. Gate 1: the model gets the question and both single steps right. Gate 2: the edited words are single tokens. Gate 3: we can find the layer where the answer forms. Gate 4 ("reach"): the edit really removes the bridge (asked to name the bridge, the model can't).
- **Need:** how much removing the bridge lowers P(right answer), above the noise from removing everyday words.
- **Specificity:** whether the bridge's *own* part does the work, rather than a part it shares with similar words (Vienna for Salzburg).
- **Isolation:** whether the effect survives when the part the bridge shares with the answer is left alone (rules out an "echo").
- **Quality** = need × specificity × isolation (0 to 1). We call ≥ 0.5 "high quality".
- **Fallback:** what the model answers when the bridge is removed in all layers before the answer forms. In practice, its default guess.

## 1. Trends: what separates the passing questions from the rest

Counts are questions out of the total in each group, per model. "Passed gates 1–3" means the edit-reach check (gate 4) was run.

| Trend | Qwen3.5-4B | Qwen3.6-27B | What it means |
|---|---|---|---|
| **Answer is a letter** (first letter of a hidden name, 9 questions) | gate 1: 5/9; quality ≥ 0.5: **4/9**; mean isolation 0.89 | gate 1: **9/9**; quality ≥ 0.5: **4/9**; mean isolation 0.97 | A spelling can't be an echo of the bridge's meaning |
| Answer is a count or number (23) | gate 1: 12/23; ≥ 0.5: 3/23 (spider ×2, hockey); mean quality of gate-1 passers 0.26 | gate 1: 20/23; ≥ 0.5: **1/23** (Thanksgiving days); mean 0.04 | Isolation is fine (0.75 / 0.61), but need is usually low |
| Answer is a country or continent (4) | ≥ 0.5: 0/4; mean isolation **0.18** | ≥ 0.5: 0/4; mean isolation **0.06** | The answer shares its meaning with the bridge: an echo |
| Answer is a month, day, colour, direction or state of matter (12) | ≥ 0.5: 0/12; mean isolation 0.24 | ≥ 0.5: 0/12; mean isolation 0.53 | Closed sets of related words overlap the bridge; and the bridge is often one too |
| **Bridge is a named place** (city or country, 11 questions): edit reaches it | **8/8** of those tested | **8/11** | The only bridge class the edit reliably removes in the 27B |
| Bridge is anything else (animal, month, colour, element, object…): edit reaches it | 10/26 | **8/40** | Animals 2/10, months/holidays 1/7, colours 0/3, objects 1/10 in the 27B |
| **Answer forms late** (27B: the reach window is the last one before the answer, layers 42–53) | — | reach passes **10/16** vs **6/35** for earlier windows | The same bridge is removable or not depending on when the answer forms: Salzburg 1.00 (letter question, answer at layer 54) vs 0.00 (country question, answer at layer 30); Tokyo 0.46 vs 0.00 |
| Letter answers form at the very end | onset layer 24–27 of 32 | onset layer **54 of 64** in 9/9 | Spelling is a last step after the bridge, which leaves the edit room |
| **Default guess = right answer** (fallback with the bridge removed is still right), among questions that reached gate 4 | 23 such: mean need **0.18**, ≥ 0.5: 1 | 37 such: mean need **0.11**, ≥ 0.5: 2 | If the model would say it anyway, the bridge isn't needed |
| Default guess ≠ right answer | 11 such: mean need **0.75**, ≥ 0.5: **6** | 14 such: mean need **0.52**, ≥ 0.5: **3** | |
| Typical default-guess failures | kidney, duck, bicycle → "Two"; bee → "Six"; Halloween → "October" | the same, plus spider legs → "Eight" | Small numbers like "Two" are the model's reflex |
| Model knows both steps but can't chain an **operation on the spelling** other than the first letter (last letter, reverse, rhyme, syllables, letter count, alphabet order; 9 questions) | gate 1: **1/9** | gate 1: **1/9** | Lookups chain; operations on spelling don't |
| Model can't chain overall | gate 1: 34/62 | gate 1: 51/62 | The bigger model knows more, but the edit misses more |
| A shortcut in the question (trick-or-treat → October; "make honey" → insects → six) | Halloween: need 0.00; bee: 0.12 | Halloween: 0.00; bee: 0.00 | Caught as bystanders, as designed |
| Bridge is a single token | all 62 checked in advance | same | A precondition, not a discriminator here |

**What the winners have in common** (Colosseum, Gates, Piaf, Mozart, York letters; spider legs and hockey in the 4B; Thanksgiving days in the 27B):

1. **The bridge is a named place.** Only place bridges are reliably removable in the 27B (8/11 vs 8/40).
2. **The answer is a generic token** (a letter, a small number) that carries no meaning of the bridge, so it can't be an echo (isolation 0.97–1.00).
3. **The answer is a fact about that one bridge,** not about its category: Vienna, Graz and Munich all give different letters from Salzburg.
4. **The answer needs a step after the bridge,** so it forms late, and the edit acts before the model can re-derive the bridge from the question.
5. **The model's default guess is a different answer** (removing Italy gives "U" or "T", not "I").
6. **Both steps are famous facts, and the second is a lookup,** not an operation like counting letters.

## 2. Proposals: new question types with those properties

None of these asks for the first letter of a name or for a leg count. Each keeps properties 1–6 but changes what the answer is. The predicted verdicts are for the bank's verdict rule ("consulted bridge" = needed early, beats noise, comes first, specificity and isolation ≥ 0.5).

### A. A named place → a number fact about that place (8)

Keeps properties 1–5. The answer is a small number, like the counts, but it's an arbitrary fact about a place, like the letters. Default guesses differ from the answer (World Cup: "Five" (Brazil); stars on a flag: "Fifty" (USA); stripes: "Thirteen").

| Id | Question → bridge → answer | Why the trends predict a pass | Biggest risk | Predicted |
|---|---|---|---|---|
| `T_place_wc_italy` | How many times has the country where the Leaning Tower of Pisa stands won the men's football World Cup? → **Italy** → **Four** | Place bridge (reach 8/11); the number belongs to Italy only (Spain 1, France 2); the default is Brazil's "Five"; Germany (4) gives a same-answer swap | The 4B may not know or chain it | 27B consulted; 4B gate 1 at risk |
| `T_place_wc_brazil` | … the country where the city of Rio de Janeiro is located … World Cup? → **Brazil** → **Five** | As above | Brazil may be the *default* World Cup country, so removing it may leave "Five" (low need) | consulted, lower quality than Italy |
| `T_place_hills_rome` | On how many hills was the city where the Trevi Fountain stands built? → **Rome** → **Seven** | City bridge; "Trevi Fountain" says nothing about hills; Seven isn't a default | "Seven hills" is a famous phrase, so Seven may overlap Rome a little (isolation); few swaps have a known hill count, so the swap test will say little | consulted in both |
| `T_place_stars_china` | How many stars are on the flag of the country where the Forbidden City is located? → **China** → **Five** | Country bridge; the default for "stars on a flag" is Fifty, not Five | The 4B might fall back to "Fifty" even with China present | consulted (27B); likely (4B) |
| `T_place_stripes_greece` | How many stripes are on the flag of the country where the Parthenon stands? → **Greece** → **Nine** | Country bridge; the default is the US flag's 13; Uruguay (9) is a same-answer swap | The 4B probably doesn't know Greece's 9 stripes | 27B consulted; 4B gate 1 fail |
| `T_place_langs_switzerland` | How many national languages does the country where the city of Geneva is located have? → **Switzerland** → **Four** | Country bridge; Four is unusual (Belgium 3, Canada 2, most 1); Singapore is a same-answer swap | Some sources say three official languages; the model may say "Three" | consulted (27B) |
| `T_place_states_australia` | How many states does the country where the Sydney Opera House stands have? → **Australia** → **Six** | Country bridge; the default "Fifty" (USA) differs | Six states vs eight including territories | consulted |
| `T_place_states_austria` | How many federal states does the country where Wolfgang Amadeus Mozart was born have? → **Austria** → **Nine** | The paper's own fact (Mozart → Austria) with an answer that can't be an echo. Mozart's country scored only 0.14 / 0 because "Austria" was the answer | The 4B may not know 9; the model may think via Salzburg, with Austria as a second bridge | 27B consulted |

### B. A named place → an arbitrary code (1)

| Id | Question → bridge → answer | Why | Biggest risk | Predicted |
|---|---|---|---|---|
| `T_code_dial_russia` | What is the international dialling code of the country where Red Square is? (number only) → **Russia** → **7** | A code is as arbitrary as a spelling: it can't be an echo. Russia is the only well-known country whose code starts with a digit its neighbours don't share (Ukraine 380, Poland 48, China 86). That matters because Qwen splits numbers into digits, so P("7") reads only the first digit | The model writes "+7" despite the instruction; the 4B may not know it | 27B consulted; 4B gate 1 at risk |

*Not proposed for this reason:* other dialling codes share their first digit with their neighbours (France 33 / Spain 34 / Italy 39; Japan 81 / China 86).

### C. A named place → an arbitrary emblem (3)

The answer is a word, but it's linked to the bridge by convention, not by meaning or category (Scotland's animal is a unicorn; Ireland's isn't).

| Id | Question → bridge → answer | Why | Biggest risk | Predicted |
|---|---|---|---|---|
| `T_emblem_animal_scotland` | What is the national animal of the country whose capital is Edinburgh? → **Scotland** → **Unicorn** | Country bridge; the answer differs across similar countries (Wales: dragon, England: lion); a unicorn is not a property of "country" | Like banana → Yellow (27B: isolation 1.00 but specificity 0.26), a strong association may score as a category effect; the 4B may not know it | consulted or category effect |
| `T_emblem_animal_canada` | What is the national animal of the country where the city of Montreal is located? → **Canada** → **Beaver** | As above | Beaver is a close associate of Canada (an echo risk) | consulted or category effect |
| `T_emblem_flag_wales` | What creature is shown on the flag of the country whose capital is Cardiff? → **Wales** → **Dragon** | Country bridge; other flags show eagles, leaves, trees | "Dragon" may be the default guess for "creature on a flag" | uncertain |

### D. A named place → a two-way rule whose default is the other side (2)

A two-way answer is only useful if the default is the *wrong* side. Most countries drive on the right, so a "Left" answer has to come from the bridge.

| Id | Question → bridge → answer | Why | Biggest risk | Predicted |
|---|---|---|---|---|
| `T_rule_drive_japan` | On which side of the road do people drive in the country where the city of Osaka is located? → **Japan** → **Left** | Country bridge; default "Right" (so the fallback differs); China and Korea, its nearest similar words, drive on the right | 50% chance level makes need noisy; Left/Right are a closed set (0/12 for those answers so far, though those failed because the *bridge* was from the set too) | consulted |
| `T_rule_drive_india` | … in the country where the Taj Mahal stands? → **India** → **Left** | As above | As above | consulted |

### E. A named thing that isn't a place → a number fact (4)

These test whether reach comes from the bridge being a *name* rather than specifically a place. If they pass gate 4 in the 27B, the bank can grow well beyond geography.

| Id | Question → bridge → answer | Why | Biggest risk | Predicted |
|---|---|---|---|---|
| `T_named_logo_audi` | How many rings are in the logo of the car company based in Ingolstadt? → **Audi** → **Four** | Proper-noun bridge; the default for "rings in a logo" is the Olympics' Five | The 4B may not link Ingolstadt to Audi; few swaps with known ring counts | 27B consulted if names reach like places |
| `T_named_logo_mercedes` | How many points does the star in the logo of the car maker that builds the S-Class have? → **Mercedes** → **Three** | Proper-noun bridge; the default star has five points | Few swaps (Chrysler's five-pointed star) | consulted |
| `T_named_fingers_simpsons` | How many fingers are on each hand of the members of the yellow cartoon family that lives in Springfield? → **Simpsons** → **Four** | Proper-noun bridge; the default for fingers is Five | "Cartoon characters have four fingers" is a category rule: a likely category effect | uncertain (category effect) |
| `T_named_wives_henry` | How many wives did the English king who broke with the Pope to found the Church of England have? → **Henry** → **Six** | Person bridge (only one tested so far: Harry, which failed); the default ("One") differs | The token "Henry" is not the entity Henry VIII; "VIII" may push toward "Eight" | uncertain |

### F. A letter given by a fact, not by spelling (1)

| Id | Question → bridge → answer | Why | Biggest risk | Predicted |
|---|---|---|---|---|
| `T_fact_vitamin_carrot` | Which vitamin is the orange root vegetable that rabbits love best known for? (one letter) → **carrot** → **A** | Keeps the letter answer's isolation, but the letter comes from a fact (carrot starts with C, so spelling can't supply A); "rabbits" says nothing about vitamins | Food bridges rarely reach in the 27B (potato 0.00); " A" is also the article "a" | 4B consulted; 27B untestable (reach) |

## 3. Controls expected to FAIL

Each one is built to fail for one named reason. Three of them reuse the hop "Colosseum → Italy", which passes everywhere, so the only thing that changes is the factor being tested.

| Id | Question → bridge → answer | Expected failure | Reason (the trend it tests) |
|---|---|---|---|
| `T_bad_len_italy` | How many letters are in the name of the country where the Colosseum is located? → Italy → Five | **Untestable at gate 1** | Operations on spelling other than the first letter pass gate 1 in 1/9 questions per model |
| `T_bad_stars_usa` | How many stars are on the flag of the country where the Statue of Liberty stands? → America → Fifty | **Bystander** (need ≈ 0) | Default guess = right answer: "Fifty" comes without the bridge. Pairs with `T_place_stars_china`, which has the same format and a different default |
| `T_bad_continent_italy` | On which continent is the country where the Colosseum is located? → Italy → Europe | **Answer echo** (isolation < 0.5) | The answer is a category containing the bridge, like country answers (isolation 0.00–0.40; mean 0.18 in the 4B, 0.06 in the 27B) |
| `T_bad_vitamin_scurvy` | Which vitamin is the fruit that sailors ate to prevent scurvy rich in? → lemon/lime → C | **Bystander** | Shortcut: "scurvy → vitamin C" is stored directly, like trick-or-treat → October. Pairs with `T_fact_vitamin_carrot` |

**What a good outcome looks like:** groups A and B pass in the 27B at a rate close to the letter family (≈ 4–5 of 9), and all four controls fail for the predicted reason. If group E also passes gate 4, "named entity" is the real driver of reach, not "place".

## 4. Notes for whoever runs these

- **Token check.** Every bridge, same-kind control, swap and answer-direction word is one token with a leading space in the Qwen3.5-4B tokenizer (checked; the script is in the module docstring). The Qwen3.6-27B tokenizer is not cached on this machine, so it is unchecked. No question in the bank failed gate 2 in the 27B, which suggests the two tokenizers agree.
- **Dropped on token grounds:** Beethoven, Curie, Brahms (2+ tokens), so no composer or scientist questions; "₹" (2 tokens).
- **A promising type held back by the harness:** currency symbols (England → £, Japan → ¥). A symbol is as generic as a letter. But `Card.match` in `bridge_card3.py` ends its pattern with `\b`, and `\b` never matches after "£" at the end of a reply, so every symbol answer would classify as "other" and fail gate 1. A one-line change (`(?!\w)` instead of `\b`) would make them testable.
- **Swap tests will be thin** for Rome's hills and the two logos: few other cities or brands have a known count. That only weakens the reported swap tests, not the score.
- **The group format** (`NEXT_EXPERIMENTS.md`): only questions with a high single-question quality are worth carrying into the peer-pressure prompts. On the evidence so far, run these alone first. Group A's numbers would also give the group experiment non-letter wrong answers for the peers ("Agent 2: Five").
- **Suggested order:** the 27B first (where the letter family alone holds up). Start with `T_place_wc_italy`, `T_place_stars_china` + `T_bad_stars_usa` (a matched pair), `T_place_states_austria`, `T_code_dial_russia`, `T_named_logo_audi`, and the four controls.
