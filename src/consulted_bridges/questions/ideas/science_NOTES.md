# Science and nature bridge ideas (science_specs.py)

All bridge, swap, same-kind and answer_dirs words are one Qwen3.5-4B token with a leading space (checked). "octopus", "kangaroo", "penguin", "zebra", "Uranus", "Hg", "mammal" are NOT one token, so they were avoided.

- **S_element_gold** (metal for first-place medals → gold → Au): good; "Au" is not a spelling echo and nothing in the question gives it. Weakness: no same-answer swap can exist (symbols are unique); gold/Au are moderately linked as words.
- **S_element_helium** (gas that floats balloons → helium → atomic number 2): good; "balloons" has no link to "2". Weakness: the model may pick hydrogen (answer 1); no same-answer swap exists.
- **S_planet_saturn** (planet famous for rings → Saturn → Sixth): good; ring words have no link to "sixth". Weakness: the model may answer "6", "Six" or "6th" (all listed as forms); no same-answer swap exists.
- **S_hearts_squid** (served as calamari → squid → 3 hearts): good; the default for hearts is "One", so swaps (fish, dog...) give clear changes. Weakness: a 4B model may not know squid have 3 hearts (better known for octopus, which is 2 tokens).
- **S_organ_kidney** (organ that makes urine → kidney → Two): good; same-answer swaps lung/eye, different-answer liver/heart/brain. Weakness: "Two" may be the default guess for body-part counts; "filters the blood" could also call up liver.
- **S_continent_tiger** (big cat with orange fur and black stripes → tiger → Asia): good; nothing in the words points to Asia, and Africa is the likely default for big cats. Weakness: swap answers (camel, elephant) are ambiguous, and the model may say "India".
- **S_state_mercury** (metal in old thermometers → mercury → Liquid): risky; mercury is a strong associate of "liquid" (echo), and "thermometer" may lead to "liquid" directly. Same-answer swap is water (not a metal).
- **S_wings_bee** (animals that make honey → bee → Four wings): good; default for wings is "Two", so all swaps except butterfly flip. Weakness: the model may wrongly say bees have two wings.
- **S_element_sodium** (metal in table salt → sodium → Na): risky by design; "salt" → NaCl might give "Na" without going through "sodium". Compare with gold to see if the score notices.
- **S_bad_helium_state** (BAD, shortcut): the question says "the gas that…", so "Gas" comes straight from the question words; removing "helium" should not move the answer.
- **S_bad_cow_sound** (BAD, echo): "Moo" is a strong word associate of "cow" (and of "milk"), so the bridge and answer directions overlap and the bridge is not separable from the answer.
