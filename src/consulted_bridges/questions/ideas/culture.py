"""Idea set: culture, games, sports, food and everyday life (group "C").

Two-step questions whose middle word (the bridge) is never written, in the format of card3_specs.py
(built with its `base(...)` helper). Every bridge, swap and control word was checked to be one token with a
leading space for Qwen3.5-4B. Two questions are deliberately bad (predict starts with "bad:"), as checks that the
scores catch a shortcut and an echo. Reasons and weaknesses per question: culture_NOTES.md.
"""
from __future__ import annotations

from consulted_bridges.questions.v2 import NUM, NUM_DIRS
from consulted_bridges.questions.core import ONE, base

# number answers spelled as words. Keys are chosen so no two answers share their first token
# (e.g. no "Thirty" next to "Thirty-one"), since a one-token form also collects the longer answer's mass.
NUMW = {"1": ["One", "1"], "2": ["Two", "2"], "3": ["Three", "3"], "4": ["Four", "4"], "5": ["Five", "5"],
        "6": ["Six", "6"], "7": ["Seven", "7"], "8": ["Eight", "8"], "9": ["Nine", "9"], "10": ["Ten", "10"],
        "11": ["Eleven", "11"], "12": ["Twelve", "12"], "15": ["Fifteen", "15"]}
NUMW_DIRS = {"3": ["Three", "three"], "6": ["Six", "six"], "11": ["Eleven", "eleven"]}
FAMILY = {"Percussion": ["Percussion"], "String": ["Strings", "String", "Stringed"], "Brass": ["Brass"],
          "Woodwind": ["Woodwinds", "Woodwind"], "Keyboard": ["Keyboards", "Keyboard"]}
SPECIES = {"Mouse": ["Mouse", "Rodent", "Rat"], "Duck": ["Duck"], "Dog": ["Dog"], "Rabbit": ["Rabbit", "Bunny", "Hare"],
           "Cat": ["Cat"], "Hedgehog": ["Hedgehog"], "Pig": ["Pig"], "Frog": ["Frog"], "Human": ["Human", "Plumber"]}
GROUND = {"Below": ["Below", "Underground", "Under", "Beneath"], "Above": ["Above", "Aboveground", "Over"]}
COLOR = {c: [c] for c in ["Yellow", "Red", "Green", "Orange", "Purple", "Brown", "White", "Black", "Blue"]}
MONTH = {m: [m] for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                          "October", "November", "December"]}


def _hf(bridge):
    return list(dict.fromkeys([b[0].upper() + b[1:] for b in bridge] + bridge))


SPORT_CHECK = "How many players does each team have playing at once in {}?" + ONE

SPECS = dict([
    # chess piece by how it moves -> its point value
    ("C_chess_knight", base(
        "How many points is the chess piece that moves in an L-shape worth?" + ONE,
        ["knight", "knights"], "piece", "3", NUMW, NUMW_DIRS["3"],
        "Which chess piece moves in an L-shape?" + ONE, _hf(["knight", "knights"]),
        "How many points is a knight worth in chess?" + ONE, "How many points is a {} worth in chess?" + ONE,
        ["queen", "pawn"], ["pawn", "queen", "castle", "bishop"], "C", "consulted", "How many")),
    # sport by its equipment -> players per team
    ("C_sport_hockey", base(
        "How many players does each team have on the ice at once in the sport played with a puck?" + ONE,
        ["hockey"], "sport", "6", NUMW, NUMW_DIRS["6"],
        "Which sport is played with a puck?" + ONE, _hf(["hockey"]),
        "How many players does each team have playing at once in hockey?" + ONE, SPORT_CHECK,
        ["tennis", "golf"], ["soccer", "basketball", "baseball", "rugby", "cricket", "polo", "volleyball"], "C",
        "consulted", "How many")),
    # sport by a famous player -> players per team
    ("C_sport_messi", base(
        "How many players does each team have on the field at once in the sport Lionel Messi plays?" + ONE,
        ["football", "soccer"], "sport", "11", NUMW, NUMW_DIRS["11"],
        "Which sport does Lionel Messi play?" + ONE, _hf(["football", "soccer"]),
        "How many players does each team have playing at once in football?" + ONE, SPORT_CHECK,
        ["tennis", "golf"], ["basketball", "baseball", "hockey", "rugby", "volleyball", "polo", "cricket"], "C",
        "consulted", "How many")),
    # holiday -> its month -> the month's number
    ("C_month_thanksgiving", base(
        "What is the number of the month in which Americans celebrate Thanksgiving?" + ONE,
        ["November"], "month", "11", NUMW, NUMW_DIRS["11"],
        "In which month do Americans celebrate Thanksgiving?" + ONE, ["November"],
        "What is the number of the month November?" + ONE, "What is the number of the month {}?" + ONE,
        ["March", "August"], ["February", "October", "December", "January", "July", "May", "June", "Nov"], "C",
        "consulted", "What is")),
    # instrument through a famous player -> instrument family
    ("C_music_ringo", base(
        "To which family of instruments does the instrument Ringo Starr played in the Beatles belong?" + ONE,
        ["drums", "drum"], "instrument", "Percussion", FAMILY, ["percussion"],
        "Which instrument did Ringo Starr play in the Beatles?" + ONE, _hf(["drums", "drum"]),
        "To which family of instruments do the drums belong?" + ONE,
        "To which family of instruments does the {} belong?" + ONE,
        ["piano", "organ"], ["guitar", "trumpet", "flute", "violin", "sax", "horn", "bass", "triangle"], "C",
        "consulted", "To which")),
    # fictional character -> species
    ("C_char_pikachu", base(
        "What kind of animal is Ash Ketchum's yellow electric partner in Pokémon?" + ONE,
        ["Pikachu"], "character", "Mouse", SPECIES, ["mouse", "Mouse"],
        "Which Pokémon is Ash Ketchum's yellow electric partner?" + ONE, ["Pikachu"],
        "What kind of animal is the cartoon character Pikachu?" + ONE,
        "What kind of animal is the cartoon character {}?" + ONE,
        ["Mario", "Daisy"], ["Donald", "Pluto", "Bugs", "Sonic", "Babe", "Pepe", "Tom", "Mickey", "Jerry"], "C",
        "consulted", "What kind")),
    # dish -> main ingredient -> where it grows
    ("C_food_fries", base(
        "Where does the main ingredient of French fries grow, above or below the ground?" + ONE,
        ["potato", "potatoes"], "vegetable", "Below", GROUND, ["Below", "below", "underground"],
        "What is the main ingredient of French fries?" + ONE, _hf(["potato", "potatoes"]),
        "Where does the potato grow, above or below the ground?" + ONE,
        "Where does the {} grow, above or below the ground?" + ONE,
        ["tomato", "corn"], ["tomato", "apple", "corn", "rice", "banana", "wheat", "grape", "carrot", "onion"], "C",
        "consulted", "Where does")),
    # dish -> the animal it is made from -> its legs
    ("C_food_turkey", base(
        "How many legs does the animal whose meat is the centrepiece of a Thanksgiving dinner have?" + ONE,
        ["turkey"], "animal", "2", NUM, NUM_DIRS["2"],
        "Which animal's meat is the centrepiece of a Thanksgiving dinner?" + ONE, _hf(["turkey"]),
        "How many legs does a turkey have?" + ONE, "How many legs does a {} have?" + ONE,
        ["pig", "sheep"], ["spider", "dog", "ant", "horse", "snake", "cow", "chicken", "duck"], "C",
        "consulted", "How many")),
    # --- deliberately BAD ---
    # echo: yellow is banana's top associate, so the bridge direction and the answer are hard to separate
    ("C_bad_banana", base(
        "What color is the fruit that monkeys are famous for eating?" + ONE,
        ["banana", "bananas"], "fruit", "Yellow", COLOR, ["Yellow", "yellow"],
        "Which fruit are monkeys famous for eating?" + ONE, _hf(["banana", "bananas"]),
        "What color is a banana?" + ONE, "What color is a {}?" + ONE,
        ["apple", "mango"], ["apple", "lime", "orange", "grape", "plum", "cherry", "strawberry", "lemon"], "C",
        "bad: echo (yellow is a top associate of banana; removing banana also removes yellow)", "What color")),
    # shortcut: trick-or-treating itself co-occurs with October, so the holiday can be skipped
    ("C_bad_halloween", base(
        "In which month is the holiday when children go trick-or-treating?" + ONE,
        ["Halloween"], "holiday", "October", MONTH, ["October"],
        "On which holiday do children go trick-or-treating?" + ONE, ["Halloween"],
        "In which month is Halloween?" + ONE, "In which month is {}?" + ONE,
        ["Easter", "Ramadan"], ["Christmas", "Thanksgiving", "Valentine", "Easter", "Ramadan"], "C",
        "bad: shortcut (trick-or-treating -> October directly, no need for Halloween)", "In which")),
])
