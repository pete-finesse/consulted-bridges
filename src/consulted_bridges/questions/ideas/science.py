"""Science and nature bridge questions (ideas, not yet run). Same fields as card3_specs.py, built with its base().

Every bridge, swap, same-kind and answer_dirs word was checked to be ONE Qwen3.5-4B token with a leading space.
Two questions are deliberately bad (predict starts with "bad:") so we can check the scores catch them; one more
is marked "risky" (a possible shortcut we are unsure about).

Note: for answers that are unique to one bridge (chemical symbols, atomic numbers, planet positions) no swap with
the SAME answer can exist; those questions have none.
"""
from __future__ import annotations

from consulted_bridges.questions.core import ONE, base

NUMS = {"1": ["One", "1"], "2": ["Two", "2"], "3": ["Three", "3"], "4": ["Four", "4"], "5": ["Five", "5"],
        "6": ["Six", "6"], "7": ["Seven", "7"], "8": ["Eight", "8"], "10": ["Ten", "10"], "0": ["Zero", "None", "0"]}
NUM_DIRS = {"1": ["One", "one"], "2": ["Two", "two"], "3": ["Three", "three"], "4": ["Four", "four"]}
SYM = {s: [s] for s in ["Au", "Ag", "Fe", "Cu", "Pb", "Sn", "Na", "Zn", "Pt", "Hg", "K", "Cl"]}
ORD = {"1": ["First", "1st", "1", "One"], "2": ["Second", "2nd", "2", "Two"], "3": ["Third", "3rd", "3", "Three"],
       "4": ["Fourth", "4th", "4", "Four"], "5": ["Fifth", "5th", "5", "Five"], "6": ["Sixth", "6th", "6", "Six"],
       "7": ["Seventh", "7th", "7", "Seven"], "8": ["Eighth", "8th", "8", "Eight"]}
STATE = {"Solid": ["Solid"], "Liquid": ["Liquid"], "Gas": ["Gas", "Gaseous"]}
CONT = {"Asia": ["Asia"], "Africa": ["Africa"], "Europe": ["Europe"], "Australia": ["Australia", "Oceania"],
        "South America": ["South America", "South"], "North America": ["North America", "North"],
        "America": ["America", "Americas"], "Antarctica": ["Antarctica"]}
SOUND = {"Moo": ["Moo", "Mooing", "Moos"], "Woof": ["Woof", "Bark", "Barking", "Barks"], "Meow": ["Meow", "Miaow"],
         "Quack": ["Quack"], "Baa": ["Baa", "Bleat"], "Oink": ["Oink"], "Roar": ["Roar"], "Neigh": ["Neigh"]}


def _hf(bridge):
    return [b.capitalize() for b in bridge] + bridge


def q(name, predict, user, bridge, kind, answer, answers, answer_dirs, hop1, hop2, check, same_kind, swaps):
    return name, base(user + ONE, bridge, kind, answer, answers, answer_dirs, hop1 + ONE, _hf(bridge), hop2 + ONE,
                      check + ONE, same_kind, swaps, "S", predict, " ".join(user.split()[:2]))


SPECS = dict([
    # element -> chemical symbol (symbol is not a spelling echo of "gold")
    q("S_element_gold", "consulted",
      "What is the chemical symbol of the metal that first-place Olympic medals are named after?",
      ["gold"], "metal", "Au", SYM, ["Au"],
      "Which metal are first-place Olympic medals named after?",
      "What is the chemical symbol of gold?", "What is the chemical symbol of {}?",
      ["silver", "platinum"], ["silver", "iron", "copper", "lead", "tin", "sodium", "zinc"]),
    # element -> atomic number
    q("S_element_helium", "consulted",
      "What is the atomic number of the gas that makes party balloons float?",
      ["helium"], "gas", "2", NUMS, NUM_DIRS["2"],
      "Which gas makes party balloons float?",
      "What is the atomic number of helium?", "What is the atomic number of {}?",
      ["neon", "hydrogen"], ["hydrogen", "lithium", "carbon", "nitrogen", "oxygen", "neon"]),
    # planet -> position from the Sun
    q("S_planet_saturn", "consulted",
      "What position from the Sun is the planet famous for its big bright rings?",
      ["Saturn"], "planet", "6", ORD, ["Sixth", "sixth"],
      "Which planet is famous for its big bright rings?",
      "What position from the Sun is Saturn?", "What position from the Sun is {}?",
      ["Jupiter", "Neptune"], ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Neptune"]),
    # animal -> number of hearts (octopus is 2 tokens, so squid via calamari)
    q("S_hearts_squid", "consulted",
      "How many hearts does the sea animal that is served as calamari have?",
      ["squid"], "animal", "3", NUMS, NUM_DIRS["3"],
      "Which sea animal is served as calamari?",
      "How many hearts does a squid have?", "How many hearts does a {} have?",
      ["crab", "shrimp"], ["fish", "dog", "cat", "frog", "horse", "worm", "crab"]),
    # organ -> how many a person has
    q("S_organ_kidney", "consulted",
      "How many does a person have of the organ that filters the blood and makes urine?",
      ["kidney", "kidneys"], "organ", "2", NUMS, NUM_DIRS["2"],
      "Which organ filters the blood and makes urine?",
      "How many kidneys does a person have?", "How many {}s does a person have?",
      ["liver", "bladder"], ["liver", "heart", "brain", "stomach", "bladder", "lung", "eye"]),
    # animal -> continent where it lives in the wild
    q("S_continent_tiger", "consulted",
      "On which continent does the big cat with orange fur and black stripes live in the wild?",
      ["tiger"], "animal", "Asia", CONT, ["Asia"],
      "Which big cat has orange fur and black stripes?",
      "On which continent do tigers live in the wild?", "On which continent does the {} live in the wild?",
      ["lion", "cougar"], ["lion", "llama", "elephant", "camel", "cougar", "panda", "yak"]),
    # metal -> state of matter at room temperature
    q("S_state_mercury", "consulted (risky: mercury is a strong associate of 'liquid')",
      "At room temperature, what state of matter is the metal used in old thermometers?",
      ["mercury", "Mercury"], "substance", "Liquid", STATE, ["Liquid", "liquid"],
      "Which metal is used in old thermometers?",
      "At room temperature, what state of matter is mercury?", "At room temperature, what state of matter is {}?",
      ["iron", "copper"], ["iron", "gold", "copper", "oxygen", "helium", "nitrogen", "water"]),
    # insect -> number of wings (default guess for wings is two)
    q("S_wings_bee", "consulted",
      "How many wings do the animals that make honey have?",
      ["bee", "bees"], "animal", "4", NUMS, NUM_DIRS["4"],
      "Which animals make honey?",
      "How many wings does a bee have?", "How many wings does a {} have?",
      ["ant", "fly"], ["bird", "fly", "duck", "bat", "chicken", "eagle", "butterfly"]),
    # element -> symbol, through table salt: salt -> NaCl may give "Na" without the bridge
    q("S_element_sodium", "risky: 'table salt' -> NaCl may give Na without the bridge",
      "What is the chemical symbol of the metal found in table salt?",
      ["sodium"], "metal", "Na", SYM, ["Na"],
      "Which metal is found in table salt?",
      "What is the chemical symbol of sodium?", "What is the chemical symbol of {}?",
      ["potassium", "lithium"], ["gold", "silver", "iron", "copper", "lead", "tin", "zinc"]),
    # DELIBERATELY BAD 1: the question says "gas", which gives the answer directly
    q("S_bad_helium_state", "bad: shortcut, the question's word 'gas' names the answer; helium is skippable",
      "At room temperature, what state of matter is the gas that makes party balloons float?",
      ["helium"], "substance", "Gas", STATE, ["Gas", "gas"],
      "Which gas makes party balloons float?",
      "At room temperature, what state of matter is helium?", "At room temperature, what state of matter is {}?",
      ["neon", "hydrogen"], ["iron", "gold", "copper", "mercury", "water", "hydrogen", "oxygen"]),
    # DELIBERATELY BAD 2: echo, "moo" is a strong word associate of "cow" (and of "milk")
    q("S_bad_cow_sound", "bad: echo, the answer 'Moo' is a strong associate of the bridge 'cow' (and of 'milk')",
      "What sound does the farm animal that gives us milk make?",
      ["cow", "cows"], "animal", "Moo", SOUND, ["Moo", "moo"],
      "Which farm animal gives us milk?",
      "What sound does a cow make?", "What sound does a {} make?",
      ["sheep", "pig"], ["dog", "cat", "duck", "sheep", "pig", "lion", "horse", "bull"]),
])
