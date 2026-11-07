"""Round 3 questions: families built so the answer is a property of the bridge only (like spider).

  L  first letter of the hidden word. The answer comes from the bridge's spelling, so neither the
     subject (Mozart) nor the bridge's category (Austrian city) can supply it; the same-kind control
     (Vienna) has a different letter. Reuses the paper's own facts with the shortcut removed.
  R  leg-count riddles with no class word in the question (not "insects that...").
  O  counts of an object's parts, reached through a description.
  X  reference questions from round 2 (spider, York University, Mozart's and Mishima's country).

Fields as in card2_specs.py, plus:
  name_ask, name_prefill  the same question, asked to name the bridge first ("... First name the city." +
                          "The city is"): edits act on identical question tokens, so this checks whether the
                          edit reaches the bridge IN THIS PROMPT
  kind_word               used in name_ask / name_prefill
"""
from __future__ import annotations

from consulted_bridges.questions.v2 import CITIES, COUNTRY, COUNTRY_DIRS, NUM, NUM_DIRS, SWAP_CITIES

LETTER_ONE = " Answer with one letter."
ONE = " Answer in one word."
LETTERS = {c: [c] for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}


def _forms(words):
    out = []
    for w in words:
        out += [w, f"a {w}", f"an {w}", f"the {w}"]
    return out


def base(user, bridge, kind, answer, answers, answer_dirs, hop1, hop1_forms, hop2, check, same_kind, swaps, group,
         predict, q_start):
    return dict(user=user, bridge=bridge, answer=answer, answers=answers, answer_dirs=answer_dirs, hop1=hop1,
                hop1_forms=hop1_forms, hop2=hop2, check=check, same_kind=same_kind, swaps=swaps, group=group,
                predict=predict, q_start=q_start, name_ask=f" First name the {kind}.", name_prefill=f"The {kind} is",
                name_forms=_forms(bridge))


def letter(name, predict, subject_phrase, bridge, kind, same_kind, swaps, hop1):
    """What letter does the name of <subject phrase> start with? -> first letter of the bridge."""
    L = bridge[0][0].upper()
    user = f"What letter does the name of {subject_phrase} start with?" + LETTER_ONE
    return name, base(user, bridge, kind, L, LETTERS, [L], hop1 + ONE, bridge,
                      f"What letter does the word {bridge[0]} start with?" + LETTER_ONE,
                      "What letter does the word {} start with?" + LETTER_ONE, same_kind, swaps, "L", predict,
                      "What letter")


def riddle_legs(name, predict, description, bridge, ans, same_kind, swaps, hop1):
    user = f"How many legs do {description} have?" + ONE
    return name, base(user, bridge, "animal", ans, NUM, NUM_DIRS[ans], hop1 + ONE, [b.capitalize() for b in bridge] + bridge,
                      f"How many legs does a {bridge[0]} have?" + ONE, "How many legs does a {} have?" + ONE,
                      same_kind, swaps, "R", predict, "How many")


def parts(name, predict, question, bridge, kind, ans, part, same_kind, swaps, hop1):
    return name, base(question + ONE, bridge, kind, ans, NUM, NUM_DIRS.get(ans, [NUM[ans][0]]), hop1 + ONE,
                      [b.capitalize() for b in bridge] + bridge, f"How many {part} does a {bridge[0]} have?" + ONE,
                      f"How many {part} does a {{}} have?" + ONE, same_kind, swaps, "O", predict,
                      " ".join(question.split()[:2]))


def born_country(name, predict, person, city, country):
    user = f"In which country is the city where {person} was born?" + ONE
    return name, base(user, [city], "city", country, COUNTRY, COUNTRY_DIRS.get(country, [COUNTRY[country][0]]),
                      f"In which city was {person} born?" + ONE, [city], f"In which country is {city}?" + ONE,
                      "In which country is {}?" + ONE, [c for c in CITIES.get(country, []) if c != city],
                      [c for c in SWAP_CITIES if c != city][:8], "X", predict, "In which")


CITY_SWAPS = ["Munich", "Paris", "Tokyo", "Berlin", "Rome", "Madrid", "Chicago", "Lyon"]

SPECS = dict([
    # L: first letter of the hidden word
    letter("L_mozart", "consulted", "the city where Wolfgang Amadeus Mozart was born", ["Salzburg"], "city",
           ["Vienna", "Graz"], CITY_SWAPS, "In which city was Wolfgang Amadeus Mozart born?"),
    letter("L_mishima", "consulted", "the city where Yukio Mishima was born", ["Tokyo"], "city",
           ["Osaka", "Kyoto"], ["Munich", "Paris", "Berlin", "Rome", "Madrid", "Chicago", "Lyon", "Osaka"],
           "In which city was Yukio Mishima born?"),
    letter("L_maldini", "consulted", "the city where Paolo Maldini was born", ["Milan"], "city",
           ["Rome", "Naples"], ["Munich", "Paris", "Tokyo", "Berlin", "Madrid", "Chicago", "Lyon", "Rome"],
           "In which city was Paolo Maldini born?"),
    letter("L_piaf", "consulted", "the city where Édith Piaf was born", ["Paris"], "city", ["Lyon", "Marseille"],
           ["Munich", "Tokyo", "Berlin", "Rome", "Madrid", "Chicago", "Lyon"], "In which city was Édith Piaf born?"),
    letter("L_gates", "consulted", "the city where Bill Gates was born", ["Seattle"], "city", ["Boston", "Chicago"],
           CITY_SWAPS, "In which city was Bill Gates born?"),
    letter("L_federer", "consulted", "the city where Roger Federer was born", ["Basel"], "city", ["Zurich", "Geneva"],
           CITY_SWAPS, "In which city was Roger Federer born?"),
    letter("L_york", "consulted", "the city where York University has its headquarters", ["Toronto"], "city",
           ["Vancouver", "Ottawa"], ["Munich", "Paris", "Berlin", "Rome", "Madrid", "Chicago", "Lyon", "Vancouver"],
           "In which city does York University have its headquarters?"),
    letter("L_spider", "consulted", "the animals that spin webs", ["spider", "spiders"], "animal",
           ["ants", "bees"], ["bee", "ant", "dog", "bird", "horse", "cat", "duck", "snake"], "Which animals spin webs?"),
    letter("L_colosseum", "untestable (country bridge)", "the country where the Colosseum is located", ["Italy"],
           "country", ["Spain", "France"], ["Germany", "Japan", "Spain", "France", "Brazil", "Canada"],
           "In which country is the Colosseum located?"),
    # R: leg riddles, no class word
    riddle_legs("R_spider", "consulted", "the animals that spin webs", ["spider", "spiders"], "8", ["ants", "bees"],
                ["bee", "ant", "dog", "bird", "snake", "horse", "duck"], "Which animals spin webs?"),
    riddle_legs("R_bee", "consulted", "the animals that make honey", ["bee", "bees"], "6", ["ants", "spiders"],
                ["spider", "dog", "bird", "snake", "horse", "duck", "ant"], "Which animals make honey?"),
    riddle_legs("R_duck", "consulted", "the animals that quack", ["duck", "ducks"], "2", ["geese", "chickens"],
                ["spider", "ant", "dog", "snake", "horse", "bee", "chicken"], "Which animals quack?"),
    # O: parts of an object, via a description
    parts("O_guitar", "consulted", "How many strings does the instrument that Jimi Hendrix played have?",
          ["guitar", "guitars"], "instrument", "6", "strings", ["piano", "violin"], ["violin", "piano", "drum", "guitar"],
          "Which instrument did Jimi Hendrix play?"),
    parts("O_bicycle", "consulted", "How many wheels does the vehicle that you ride by pedalling have?",
          ["bicycle", "bike"], "vehicle", "2", "wheels", ["car", "bus"], ["car", "bus", "motorcycle"],
          "Which vehicle do you ride by pedalling?"),
    # X: round-2 references, with the new same-prompt reach check
    ("X_spider", dict(riddle_legs("X", "consulted (reference)", "the animals that spin webs", ["spider", "spiders"],
                                  "8", ["ticks"], ["bee", "ant", "dog", "bird", "snake", "horse", "beetle", "tick"],
                                  "Which animals spin webs?")[1], group="X")),
    born_country("X_mozart_country", "answer from the start (round 2)", "Wolfgang Amadeus Mozart", "Salzburg", "Austria"),
    born_country("X_mishima_country", "bystander (round 2)", "Yukio Mishima", "Tokyo", "Japan"),
])
SPECS["X_york_country"] = dict(SPECS["X_mishima_country"])
from consulted_bridges.questions.v2 import SPECS as S2  # noqa: E402
_y = S2["C2_york"]
SPECS["X_york_country"] = base(_y["user"], _y["bridge"], "city", _y["answer"], _y["answers"], _y["answer_dirs"], _y["hop1"],
                               _y["hop1_forms"], _y["hop2"], _y["check"], _y["same_kind"], _y["swaps"], "X",
                               "consulted (round 2)", _y["q_start"])
