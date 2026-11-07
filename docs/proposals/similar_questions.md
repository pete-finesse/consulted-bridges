# Proposed: close variants of the questions that already work

*Proposed only: not added to any question set. Run on both models; results at the end. Specs: `src/consulted_bridges/questions/proposed/similar.py` (38 questions, ids `V_…`); single-token check: `scripts/check_similar.py` (all pass).*

## Why these

Each family copies the exact format of a question that already scores well. Only the subject changes. Controls are words of the bridge's kind with a *different* answer, as Vienna is for Salzburg.

| Family | Modelled on (quality: 4B / 27B) | Proposed | Why it should work |
|---|---|---|---|
| **Letter of a landmark's country** | Colosseum → Italy → I (0.95 / 0.98) | 14 | The answer is the bridge's spelling, so there's no shortcut. Country letters don't overlap country directions (isolation ≈ 1) |
| **Letter of a person's birth city** | Gates → Seattle → S (0.73 / 0.96), Piaf → Paris → P (– / 0.96), Mozart → Salzburg → S (0.66 / 0.56) | 8 | Same reasons. People taken from the facts the 4B knows |
| **Letter of a university's city** | York University → Toronto → T (0.91 / 0.40) | 6 | Same format, other universities the 4B knows |
| **Players per team in a sport, by description** | hockey → Six (0.74 / bystander) | 5 | The count is a property of the sport only. The description names no number, or a misleading one |
| **Days in a holiday's month** | Thanksgiving → November → 30 (– / 0.52) | 3 | The day count belongs to the month, not the holiday |
| **Legs of an animal, by description** | spider → Eight (0.95 / untestable) | 2 | Same format as spider |

## The proposals

**Letter of a landmark's country** ("What letter does the name of the country where X is located start with?"):
- Eiffel Tower → France → F
- Leaning Tower of Pisa → Italy → I *(trap: Pisa starts with P)*
- Taj Mahal → India → I
- Kremlin → Russia → R
- Sydney Opera House → Australia → A *(trap: Sydney starts with S)*
- Christ the Redeemer → Brazil → B
- Machu Picchu → Peru → P
- Pyramids of Giza → Egypt → E
- Mount Fuji → Japan → J
- Acropolis → Greece → G
- Sagrada Familia → Spain → S
- Angkor Wat → Cambodia → C
- Brandenburg Gate → Germany → G
- Petronas Towers → Malaysia → M

**Letter of a person's birth city** ("…the city where X was born…"):
- Michelle Obama → Chicago → C
- Beyoncé → Houston → H
- Chris Hemsworth → Melbourne → M
- Muhammad Ali → Louisville → L
- Justin Timberlake → Memphis → M
- Christopher Plummer → Toronto → T
- Jimi Hendrix → Seattle → S
- Kareena Kapoor → Mumbai → M

**Letter of a university's city** ("…the city where X has its headquarters…"):
- Berklee College of Music → Boston → B
- Fudan University → Shanghai → S
- Pompeu Fabra University → Barcelona → B
- Johns Hopkins University → Baltimore → B
- University of Alberta → Edmonton → E
- Dalhousie University → Halifax → H

**Players per team:**
- "…the sport where players skate and shoot a puck" → hockey → Six *(a close variant of the hockey question)*
- "…the sport invented by James Naismith" → basketball → Five
- "…the sport played with a bat, a ball and wickets" → cricket → Eleven
- "…the sport where players hit home runs and run around four bases" → baseball → Nine *(the question says "four")*
- "…the sport played on horseback with mallets" → polo → Four

**Days in a holiday's month:**
- Valentine's Day → February → 28
- Christmas → December → 31
- American Independence Day → July → 31

**Legs, by description:**
- "the animals whose bites can spread Lyme disease" → tick → Eight
- "the animals that farm aphids and live in colonies" → ants → Six

## Risks to watch

- **27B gate 4 (the edit can't reach the bridge)** hit many city, country and month bridges, so expect some of these to come out untestable in the 27B. The letter family passed gate 4 in both models, though.
- **Gate 1 (the model knows it):** the 4B got 4 of 9 letter questions wrong. The person and university picks are facts it answered in the paper's own screen, but in a different format.
- **Days:** "31" and "30" start with the same token, "3", so the answer's own direction is less distinct. February's "28" doesn't have this problem.
- **Team sizes:** "Six" for hockey worked in the 4B but was a bystander in the 27B. The bigger model may know team sizes from the description directly.
- **Same-kind controls:** for letters they deliberately have a different first letter. For Fudan University only Beijing qualified (Shenzhen also starts with S).

## How to run them later

```bash
cd harness && uv run python scripts/bridge_card4.py --config qwen35_4b ...   # after adding a --set for scripts/proposed/similar_specs.py
```

They are kept out of every existing set until you decide to add them.

## Results (scorecard v4.1, both models)

Per-question data: `results/prop_all.csv` (from `prop_qwen35_4b/` and `prop_qwen36_27b/`). "Good" = quality ≥ 0.5.

| Family | Good in 4B | Good in 27B | Good in both |
|---|---|---|---|
| Letter of a landmark's country | **12 / 14** | **10 / 14** | **9** |
| Letter of a person's birth city | 0 / 8 (4 untestable) | 3 / 8 | 0 |
| Letter of a university's city | 2 / 6 | 2 / 6 | 1 (Berklee) |
| Players per team | 3 / 5 | 0 / 5 | 0 |
| Days in a holiday's month | 0 / 3 | 0 / 3 | 0 |
| Legs, by description | 0 / 2 | 0 / 2 | 0 |
| **All** | **17 / 38** | **15 / 38** | **10** |

- **Good in both models (10):** Eiffel Tower, Leaning Tower of Pisa, Kremlin, Sydney Opera House, Machu Picchu, Pyramids of Giza, Mount Fuji, Angkor Wat, Petronas Towers (all ≥ 0.76 in both), and Berklee (0.70 / 0.54). Both "trap" questions (Pisa and Sydney start with another letter) pass.
- **Good in 27B only:** Sagrada Familia, Michelle Obama, Chris Hemsworth, Muhammad Ali, Fudan. In the 4B these come out untestable, spread out, category effects or unclear.
- **Good in 4B only:** Taj Mahal, Christ the Redeemer, Acropolis, Beyoncé, Johns Hopkins, and three sports (basketball 0.95, hockey variant 0.66, baseball 0.58). In the 27B these are untestable, spread out or bystanders, as hockey was.
- **What failed:** days and legs (untestable or bystander in both); team sizes in the 27B, where the description alone seems to give the count.
- **Trend:** the landmark-letter format carries over to new subjects in both models. Formats whose answer is a number don't carry over to the 27B.
