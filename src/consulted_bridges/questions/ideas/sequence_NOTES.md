# Sequence, order and arithmetic question ideas (sequence_specs.py)

All bridge, swap, control and answer-direction words checked as one token (with a leading space) for Qwen3.5-4B.
"Echo risk" = the answer sits next to the bridge in a memorised sequence (Mars, Jupiter; February, March), so the
bridge's direction may carry some of the answer's direction too.

- **Q_planet_red_next** (Mars -> Jupiter): good, both hops are well known. Weakness: Jupiter's Great Red Spot is a small direct path "red" -> Jupiter; also echo risk (neighbours in the planet list).
- **Q_month_valentine_next** (February -> March): good, no word in the question points at March. Weakness: echo risk (neighbouring months).
- **Q_day_moon_next** (Monday -> Tuesday): good, Moon gives Monday, but nothing gives Tuesday. Weakness: the 4B model may not know the "named after the Moon" step well; echo risk.
- **Q_letter_last_before** (Z -> Y): good, the hops are trivial. Weakness: "the letter before the last" may be stored as its own fact (penultimate = Y) that skips Z; echo risk.
- **Q_planet_rings_position** (Saturn -> 6 / Sixth): good, the answer is a position, not a word linked to Saturn. Weakness: the model may number planets wrongly (Saturn as 5th or 7th); the answer may come out as "Six" or "Sixth" (both counted).
- **Q_month_halloween_number** (October -> 10): good, there are swaps for all month numbers. Weakness: "Halloween" -> "10/31" is a date string the model may know directly (a small shortcut); counting months is a weak skill for a 4B model.
- **Q_month_thanksgiving_days** (November -> 30): good, the only question with SAME-answer swaps (April, June). Weakness: 30 or 31 may be the model's default guess for any month, and days per month is known less reliably than the other facts here.
- **Q_num_week_plus1** (seven -> Eight): good, the arithmetic is easy. Weakness: strong echo (seven, eight are neighbours in the number list); the bridge is a number word, so removing it might hit all number directions.
- **Q_num_triangle_double** (three -> Six): good, "triangle" has no link to six. Weakness: "twice" may push toward a default even answer; same number-direction risk as above.
- **Q_father_wizard** (Harry -> James): good, nothing in "boy wizard, scar" points at James, and James is not a strong associate of Harry as a word. Weakness: James is a common default name; for swaps the check "father of Harry" is ambiguous (Prince Harry -> Charles), and the swap words come from different stories (Luke, Bart, Thor).
- **Q_month_christmas_next** (December -> January): BAD on purpose, a shortcut: "after Christmas" -> New Year -> January without December. The scores should show the answer survives removing December.
- **Q_color_snow_opposite** (white -> Black): BAD on purpose, an echo: black is the top word associate of white, and "opposite colour" -> black is a default. The scores should show that the bridge and answer directions cannot be separated.
