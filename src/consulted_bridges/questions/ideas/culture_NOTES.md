# Culture idea set (culture_specs.py): why each should (or should not) work

All bridge, swap and control words are one token with a leading space for Qwen3.5-4B (checked).

- **C_chess_knight** (L-shape -> knight -> Three points): "L-shape" has no tie to 3, and knight and "three" are weakly linked. Weakness: chess has only 6 pieces and " rook" is 2 tokens, so there are just 3 swaps with a different answer (pawn, queen, castle), and "castle" may not read as a piece.
- **C_sport_hockey** (puck -> hockey -> Six): "puck" and "ice" point to hockey, not to six. Weakness: the model may count skaters only ("Five"). Volleyball is the same-answer swap.
- **C_sport_messi** (Messi -> football/soccer -> Eleven): Messi's own number is 10, so there is no shortcut to 11. Weakness: the bridge has two words (football/soccer), which splits the direction. Cricket is the same-answer swap.
- **C_month_thanksgiving** (Thanksgiving -> November -> Eleven): Thanksgiving falls on a Thursday, not the 11th, so the words do not give 11. Weakness: the model may answer "November" (the bridge) instead of a number. The only same-answer swap is "Nov". Days per month was rejected because "Thirty" and "Thirty-one" share their first token.
- **C_music_ringo** (Ringo Starr -> drums -> Percussion): the answer follows only through the drums. Weakness: drums and percussion are fairly close as words (a mild echo), and the answer word is 2 tokens (its direction uses " percussion"). Triangle is the same-answer swap.
- **C_char_pikachu** (Ash's yellow electric partner -> Pikachu -> Mouse): Pikachu and "mouse" are weakly linked as words. Weakness: the model may answer "Electric" or "Rodent", or may not know the species. "Tom" and "Bugs" are ambiguous outside cartoons.
- **C_food_fries** (French fries -> potato -> Below): fries are not linked to underground. Weakness: both options ("above or below") are written in the question, so the answer word is in the prompt, and a 2-way answer gives a 50% baseline.
- **C_food_turkey** (Thanksgiving dinner -> turkey -> Two): "meat" and "dinner" suggest four-legged animals if anything, so Two must come through the bird. Weakness: "turkey" is also the country, and chicken and duck (same answer) are close neighbours of turkey.
- **C_bad_banana** (BAD, echo; monkeys -> banana -> Yellow): yellow is banana's top associate, so removing banana should also remove yellow (an answer-direction effect, not use of the bridge). It is also close to a default colour guess.
- **C_bad_halloween** (BAD, shortcut; trick-or-treating -> Halloween -> October): "trick-or-treating" co-occurs with October directly, so the model need not go through Halloween. It also has no same-answer swap (" Oktoberfest" is 2 tokens).
