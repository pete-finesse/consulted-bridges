"""Language, words and wordplay questions (beyond first letter), in the card3 format.

Each answer is a property of the hidden word's SPELLING or SOUND (last letter, rhyme, syllables,
alphabet position, reversal, homophone, letter count), so the description in the question can only
reach it through the bridge word. Two exceptions are deliberate:
  W_bad_west  shortcut: "the sun sets in the west" gives the answer straight from the question's words.
  W_bad_mice  echo: "cheese" already evokes "mice", and "mice" is the plural (a near copy) of the bridge.
W_tr_green is a risky one: a translation is a strong associate of the bridge (property 2).

All bridge, swap and same-kind words checked as ONE token with a leading space (Qwen3.5-4B tokenizer),
except where noted (W_bad_mice: "geese"/"oxen" are answers, not edited words).
Load with sys.path containing scripts/ and scripts/ideas/.
"""
from __future__ import annotations

from consulted_bridges.questions.core import LETTER_ONE, LETTERS, ONE, base


def _h1(words):
    """hop1_forms: capitalised and lower-case, with and without an article."""
    out = []
    for w in words:
        for v in (w.capitalize(), w):
            out += [v, f"A {v}", f"An {v}", f"The {v}", f"a {v}", f"an {v}", f"the {v}"]
    return out


def _words(*ws):
    return {w: [w] for w in ws}


NUMW = {"1": ["One", "1"], "2": ["Two", "2"], "3": ["Three", "3"], "4": ["Four", "4"], "5": ["Five", "5"],
        "6": ["Six", "6"], "7": ["Seven", "7"], "8": ["Eight", "8"], "9": ["Nine", "9"]}
NUMW_DIRS = {k: [v[0], v[0].lower()] for k, v in NUMW.items()}


def last_letter(name, predict, phrase, bridge, kind, same_kind, swaps, hop1):
    L = bridge[0][-1].upper()
    user = f"What is the last letter of the name of {phrase}?" + LETTER_ONE
    return name, base(user, bridge, kind, L, LETTERS, [L], hop1 + ONE, _h1(bridge),
                      f"What is the last letter of the word {bridge[0]}?" + LETTER_ONE,
                      "What is the last letter of the word {}?" + LETTER_ONE, same_kind, swaps, "W", predict,
                      "What is")


RHYME_OPTS = "fog, tree, hat, house, boat, box"
RHYME = _words("fog", "tree", "hat", "house", "boat", "box")

SPECS = dict([
    # last letter of the hidden word
    last_letter("W_last_cow", "good: W only via the spelling of 'cow'", "the animal that says moo", ["cow"], "animal",
                ["sheep", "goat"], ["dog", "cat", "pig", "duck", "horse", "sheep", "owl", "crow"],
                "Which animal says moo?"),
    last_letter("W_last_owl", "good: L only via the spelling of 'owl'", "the bird that hoots at night", ["owl"], "bird",
                ["eagle", "hawk"], ["cat", "dog", "duck", "cow", "hen", "pig", "seal", "camel"],
                "Which bird hoots at night?"),

    # rhyme, chosen from a fixed list (keeps the answer set small)
    ("W_rhyme_cat", base(
        f"Which of these words rhymes with the name of the animal that purrs: {RHYME_OPTS}?" + ONE,
        ["cat", "cats"], "animal", "hat", RHYME, ["hat", "Hat"], "Which animal purrs?" + ONE, _h1(["cat", "cats"]),
        f"Which of these words rhymes with cat: {RHYME_OPTS}?" + ONE,
        "Which of these words rhymes with {}: " + RHYME_OPTS + "?" + ONE,
        ["rabbit", "horse"], ["dog", "bee", "mouse", "goat", "fox", "rat", "bat", "frog"], "W",
        "good: 'hat' only via the sound of 'cat'; risk the model just picks the first or a random option",
        "Which of")),

    # number of syllables
    ("W_syll_elephant", base(
        "How many syllables are in the name of the animal with a trunk?" + ONE,
        ["elephant", "elephants"], "animal", "3", NUMW, NUMW_DIRS["3"], "Which animal has a trunk?" + ONE,
        _h1(["elephant", "elephants"]), "How many syllables are in the word elephant?" + ONE,
        "How many syllables are in the word {}?" + ONE, ["lion", "tiger"],
        ["cat", "dog", "horse", "tiger", "rabbit", "monkey", "turtle", "buffalo", "butterfly", "dinosaur"], "W",
        "good if the model can count syllables; risk 'Three' is a default and 4B syllable counts are shaky",
        "How many")),

    # alphabet position relative to M
    ("W_alpha_tiger", base(
        "Does the name of the big cat with orange fur and black stripes come before or after M in the alphabet?" + ONE,
        ["tiger"], "animal", "After", _words("Before", "After"), ["After", "after"],
        "Which big cat has orange fur and black stripes?" + ONE, _h1(["tiger", "tigers"]),
        "Does the word tiger come before or after M in the alphabet?" + ONE,
        "Does the word {} come before or after M in the alphabet?" + ONE, ["lion", "leopard"],
        ["cat", "dog", "lion", "horse", "bee", "fox", "snake", "rabbit", "wolf"], "W",
        "good: the letter T is reached only via 'tiger' ('cat' in the question points the wrong way); "
        "risk two-way answer near chance and weak alphabet ordering",
        "Does the")),

    # spelled backwards
    ("W_rev_dog", base(
        "What word do you get when you spell the name of the animal that barks backwards?" + ONE,
        ["dog", "dogs"], "animal", "god",
        {"god": ["god"], "tar": ["tar"], "tab": ["tab"], "mar": ["mar"], "kay": ["kay"], "tac": ["tac"],
         "gip": ["gip"], "woc": ["woc"], "eeb": ["eeb"]},
        ["god", "God"], "Which animal barks?" + ONE, _h1(["dog", "dogs"]),
        "What word do you get when you spell dog backwards?" + ONE,
        "What word do you get when you spell {} backwards?" + ONE, ["wolf", "fox"],
        ["rat", "bat", "ram", "yak", "cat", "pig", "cow", "bee"], "W",
        "good: 'god' only via the letters of 'dog'; risk 4B reversal is unreliable, no same-answer swap exists",
        "What word")),

    # homophone
    ("W_homo_bear", base(
        "Which word sounds exactly like the name of the large furry animal that hibernates in winter?" + ONE,
        ["bear", "bears"], "animal", "bare",
        {"bare": ["bare"], "dear": ["dear"], "flee": ["flee"], "hair": ["hair"], "wail": ["wail", "wale"],
         "hoarse": ["hoarse"], "sew": ["sew", "so"]},
        ["bare", "Bare"], "Which large furry animal hibernates in winter?" + ONE, _h1(["bear", "bears"]),
        "Which word sounds exactly like bear?" + ONE, "Which word sounds exactly like {}?" + ONE,
        ["wolf", "lion"], ["deer", "flea", "hare", "whale", "horse", "sow"], "W",
        "good: 'bare' only via the sound of 'bear'; risk echo (model answers 'Bear'), no same-answer swap exists",
        "Which word")),

    # number of letters
    ("W_len_jupiter", base(
        "How many letters are in the name of the largest planet in our solar system?" + ONE,
        ["Jupiter"], "planet", "7", NUMW, NUMW_DIRS["7"], "Which is the largest planet in our solar system?" + ONE,
        _h1(["Jupiter"]), "How many letters are in the word Jupiter?" + ONE,
        "How many letters are in the word {}?" + ONE, ["Saturn", "Mars"],
        ["Mars", "Venus", "Earth", "Saturn", "Pluto", "Mercury", "Neptune"], "W",
        "good: 7 only via the spelling of 'Jupiter' (Mercury, Neptune keep it); risk 4B letter counting",
        "How many")),

    # translation (risky on property 2)
    ("W_tr_green", base(
        "What is the Spanish word for the colour of grass?" + ONE,
        ["green"], "colour", "verde",
        _words("verde", "rojo", "azul", "blanco", "negro", "amarillo", "naranja", "morado", "rosa", "gris"),
        ["verde", "Verde"], "What colour is grass?" + ONE, _h1(["green"]),
        "What is the Spanish word for green?" + ONE, "What is the Spanish word for {}?" + ONE,
        ["blue", "brown"], ["red", "blue", "white", "black", "yellow", "orange", "pink"], "W",
        "risky: 'verde' is the bridge's translation, a strong associate; grass->verde may skip the English word",
        "What is")),

    # deliberately bad
    ("W_bad_west", base(
        "What is the opposite of the direction in which the sun rises?" + ONE,
        ["east", "East"], "direction", "West",
        _words("North", "South", "East", "West", "Up", "Down", "Left", "Right"), ["West", "west"],
        "In which direction does the sun rise?" + ONE, _h1(["east"]), "What is the opposite of east?" + ONE,
        "What is the opposite of {}?" + ONE, ["north", "south"], ["north", "south", "west", "left", "up", "down"],
        "W", "bad: shortcut, 'the sun sets in the west' links sun to West without going through east",
        "What is")),
    ("W_bad_mice", base(
        "What is the plural of the name of the small animal that loves cheese?" + ONE,
        ["mouse"], "animal", "mice",
        {"mice": ["mice"], "geese": ["geese"], "oxen": ["oxen"], "wolves": ["wolves"], "cats": ["cats"],
         "dogs": ["dogs"], "sheep": ["sheep"], "rats": ["rats"]},
        ["mice"], "Which small animal loves cheese?" + ONE, _h1(["mouse"]),
        "What is the plural of mouse?" + ONE, "What is the plural of {}?" + ONE, ["rat", "squirrel"],
        ["goose", "ox", "wolf", "cat", "dog", "sheep", "rat"], "W",
        "bad: echo, 'cheese' already evokes 'mice' and 'mice' is a near copy of the bridge",
        "What is")),
])
