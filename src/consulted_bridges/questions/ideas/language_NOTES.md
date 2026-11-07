# Language / wordplay bridge questions: expectations

- **W_last_cow** (moo -> cow -> W): good; W is rare and unrelated to "moo"/"animal". Weakness: 4B last-letter spelling can slip; "crow" is the same-answer swap.
- **W_last_owl** (hoots at night -> owl -> L): good; no L-final word in the question. Weakness: model may say "Owls" and spell the plural (S).
- **W_rhyme_cat** (purrs -> cat -> hat, from a 6-word list): good; answer set fixed by the list, rat/bat keep "hat". Weakness: model may pick an option by position or at random, so the base accuracy may be low.
- **W_syll_elephant** (trunk -> elephant -> 3): good on paper; buffalo/butterfly/dinosaur keep 3. Weakness: "Three" may be a default count, and 4B syllable counting is shaky.
- **W_alpha_tiger** (orange fur, black stripes -> tiger -> After M): good; "cat" in the question points the other way (Before), so a shortcut would hurt. Weakness: two-way answer near chance; weak alphabet ordering in small models.
- **W_rev_dog** (barks -> dog -> god): good; "god" comes only from the letters. Weakness: reversal is hard for 4B except famous pairs; no same-answer swap exists; non-word reversals (tac, gip) may come out garbled.
- **W_homo_bear** (hibernates -> bear -> bare): good; "bare" only via the sound. Weakness: echo risk (answers "Bear"), no same-answer swap, homophone knowledge of swaps (hare/hair, flea/flee) uneven.
- **W_len_jupiter** (largest planet -> Jupiter -> 7): good; Mercury/Neptune keep 7. Weakness: letter counting is unreliable in small models, so hop2 may fail.
- **W_tr_green** (grass -> green -> verde): risky; the translation is a strong associate of the bridge (property 2), and grass -> verde may go through the colour concept without the English word. Expect a weak or blurred bridge score.
- **W_bad_west** (sun rises -> east -> West): DELIBERATELY BAD, shortcut; "the sun sets in the west" links the question to the answer without east. Expect removing east to leave West standing.
- **W_bad_mice** (loves cheese -> mouse -> mice): DELIBERATELY BAD, echo; "cheese" evokes "mice" directly and "mice" is a near copy of the bridge. Expect the answer to survive removing mouse or the bridge and answer to be inseparable.
