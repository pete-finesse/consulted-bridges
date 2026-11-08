"""PROPOSED count questions (ids N_): a second answer type for the conformity experiment, so the result is not
only about spelling. A letter answer is about the bridge's *spelling*; a count is a *fact about* the bridge.

What the bank already showed for counts (data/artifacts/question_bank/results.csv) and how this set is built:
  - On Qwen3.6-27B nearly every count question failed (need ~0, or the edit did not reach the bridge): legs of an
    animal from a description, players in a sport from a description, flag stars, hills, languages. The ones that
    passed: Mozart -> Austria -> 9 federal states (0.92), and two month-day questions. The 4B passed more
    (spider legs, four sports, Australia's states, Brazil's World Cups, US flag stars).
  - So the drafts lean on the shapes that pass, and screen three ways before any check runs:
      1. the bridge's name or meaning does not contain the number (no Pentagon, tricycle, octopus);
      2. the answer is not the default for the bridge's kind (not 4 wheels for a vehicle);
      3. the description does not carry the number itself (the subject is a person or a landmark, not the
         defining trait of the bridge, as "the animals that make honey" carries six legs).
  - Right answers are 1-9 only: Qwen splits digits, so 10+ shares a first token with a small number.

Families:
  N_LM  landmark -> country -> a count about that country   (same landmarks as the passing V_LM letter questions:
                                                             same bridge, only the second hop changes)
  N_BP  person -> birth country -> a count about that country (the Mozart -> Austria -> 9 shape)
  N_BB  person -> band -> number of members                  (new)
  N_X   one-off shapes: a game's board, an instrument's strings, a planet's moons

Same-kind controls are words of the bridge's kind with a DIFFERENT answer (as Spain, one win, for France, two).

Screened 2026-11-08 (trimmed scorecard, results/counts_<config>/):
  Qwen3.6-27B consulted: N_BP_schwarzenegger 0.94, N_BP_jackman 0.80, N_LM_eiffel 0.78, N_BP_gisele 0.39 (weak
    specificity). Most others fail knows-it (World Cup counts answered 0) or the edit never reaches the bridge.
  Qwen3.5-4B consulted: N_LM_fuji 0.35 only.
"""
from __future__ import annotations

from consulted_bridges.questions.core import ONE, base
from consulted_bridges.questions.proposed.trend import NUMT, _cap, _dirs

WC = "How many times has {} won the men's football World Cup?"
WC_COUNTRIES = ["Brazil", "Germany", "Italy", "Argentina", "France", "Uruguay", "England", "Spain"]


def count(name, predict, user, bridge, kind, ans, hop1, hop2, check, same_kind, swaps, group, q_start="How many"):
    return name, base(user + ONE, [bridge], kind, ans, NUMT, _dirs(ans), hop1 + ONE, _cap([bridge]), hop2 + ONE,
                      check + ONE, same_kind, [s for s in swaps if s != bridge][:8], group, predict, q_start)


def wc_landmark(name, predict, place, country, ans, same_kind, verb="is located"):
    return count(name, predict,
                 f"How many times has the country where {place} {verb} won the men's football World Cup?",
                 country, "country", ans,
                 f"In which country is {place} located?" if verb == "is located" else f"In which country does {place} stand?",
                 WC.format(country), WC, same_kind, WC_COUNTRIES, "N_LM")


def wc_born(name, predict, person, country, ans, same_kind):
    return count(name, predict, f"How many times has the country where {person} was born won the men's football World Cup?",
                 country, "country", ans, f"In which country was {person} born?", WC.format(country), WC, same_kind,
                 WC_COUNTRIES, "N_BP")


def born_count(name, predict, person, country, ans, what, same_kind, swaps):
    """person -> birth country -> 'How many <what> does <country> have?'"""
    return count(name, predict, f"How many {what} does the country where {person} was born have?", country, "country",
                 ans, f"In which country was {person} born?", f"How many {what} does {country} have?",
                 f"How many {what} does {{}} have?", same_kind, swaps, "N_BP")


BANDS = ["Queen", "Police", "Oasis", "Muse", "Rush", "Blur", "Kiss", "Eagles", "Genesis"]


def band(name, predict, person, bridge, ans, same_kind, role="fronted", did="did"):
    return count(name, predict, f"How many members {did} the band that {person} {role} have?", bridge, "band", ans,
                 f"Which band {'was' if did == 'did' else 'is'} {person} a member of?",
                 f"How many members {did} {bridge} have?", f"How many members {did} {{}} have?", same_kind, BANDS, "N_BB")


SPECS = dict([
    # ---- N_LM: same landmarks as the passing letter questions; the second hop is a World Cup count -----------
    wc_landmark("N_LM_eiffel", "consulted", "the Eiffel Tower", "France", "2", ["Spain", "Germany"]),
    wc_landmark("N_LM_sagrada", "consulted", "the Sagrada Familia", "Spain", "1", ["France", "Italy"]),
    wc_landmark("N_LM_brandenburg", "consulted", "the Brandenburg Gate", "Germany", "4", ["France", "Spain"]),
    wc_landmark("N_LM_christ", "consulted (lower: Brazil may be the default World Cup country)",
                "the Christ the Redeemer statue", "Brazil", "5", ["Argentina", "France"]),
    wc_landmark("N_LM_colosseum", "consulted (the Pisa version failed knows-it on both models)", "the Colosseum",
                "Italy", "4", ["Spain", "France"]),
    wc_landmark("N_LM_bigben", "consulted", "Big Ben", "England", "1", ["France", "Germany"], verb="stands"),
    count("N_LM_fuji", "consulted", "How many main islands does the country where Mount Fuji is located have?",
          "Japan", "country", "4", "In which country is Mount Fuji located?", "How many main islands does Japan have?",
          "How many main islands does {} have?", ["Britain", "Indonesia"],
          ["Britain", "Indonesia", "Greece", "Italy", "Canada", "Norway", "Philippines", "Denmark"], "N_LM"),
    count("N_LM_taj", "consulted (cricket, not football: India has won the cricket World Cup twice)",
          "How many times has the country where the Taj Mahal is located won the men's cricket World Cup?", "India",
          "country", "2", "In which country is the Taj Mahal located?",
          "How many times has India won the men's cricket World Cup?",
          "How many times has {} won the men's cricket World Cup?", ["Pakistan", "England"],
          ["Australia", "England", "Pakistan", "Bangladesh", "Afghanistan", "Ireland", "Kenya", "Zimbabwe"], "N_LM"),

    # ---- N_BP: person -> birth country -> a count (the Mozart -> Austria -> 9 shape) ---------------------------
    born_count("N_BP_schwarzenegger", "consulted (same answer as the Mozart reference, another person)",
               "Arnold Schwarzenegger", "Austria", "9", "federal states", ["Germany", "Switzerland"],
               ["Germany", "Switzerland", "Australia", "Brazil", "Mexico", "India", "Nigeria", "Malaysia"]),
    born_count("N_BP_jackman", "consulted", "Hugh Jackman", "Australia", "6", "states", ["Austria", "Germany"],
               ["Austria", "Germany", "Brazil", "Mexico", "India", "Nigeria", "Malaysia", "Venezuela"]),
    born_count("N_BP_dion", "consulted (unsure: provinces vs territories)", "Celine Dion", "Canada", "3",
               "territories", ["Australia", "India"], ["Australia", "India", "Brazil", "Argentina", "Mexico", "Russia",
                                                       "Pakistan", "Nigeria"]),
    born_count("N_BP_joyce", "consulted", "James Joyce", "Ireland", "4", "provinces", ["Canada", "Spain"],
               ["Canada", "Spain", "Italy", "Argentina", "Pakistan", "Belgium", "Netherlands", "Finland"]),
    born_count("N_BP_kurosawa", "consulted", "Akira Kurosawa", "Japan", "4", "main islands", ["Britain", "Indonesia"],
               ["Britain", "Indonesia", "Greece", "Italy", "Canada", "Norway", "Philippines", "Denmark"]),
    born_count("N_BP_herge", "consulted (unsure the model knows the count)", "Hergé", "Belgium", "3",
               "official languages", ["Switzerland", "Canada"],
               ["Switzerland", "Canada", "France", "Germany", "Netherlands", "Spain", "Italy", "Finland"]),
    wc_born("N_BP_picasso", "consulted", "Pablo Picasso", "Spain", "1", ["France", "Italy"]),
    wc_born("N_BP_monet", "consulted", "Claude Monet", "France", "2", ["Spain", "Germany"]),
    wc_born("N_BP_beethoven", "consulted", "Ludwig van Beethoven", "Germany", "4", ["France", "Spain"]),
    wc_born("N_BP_borges", "consulted", "Jorge Luis Borges", "Argentina", "3", ["Brazil", "Uruguay"]),
    wc_born("N_BP_gisele", "consulted (lower: Brazil may be the default)", "Gisele Bündchen", "Brazil", "5",
            ["Argentina", "Italy"]),

    # ---- N_BB: person -> band -> members. Default for a rock band is four, so most answers are not four --------
    band("N_BB_bellamy", "consulted", "Matt Bellamy", "Muse", "3", ["Queen", "Oasis"]),
    band("N_BB_geddy", "consulted", "Geddy Lee", "Rush", "3", ["Queen", "Oasis"]),
    band("N_BB_sting", "consulted", "Sting", "Police", "3", ["Queen", "Oasis"]),
    band("N_BB_gallagher", "consulted (lineup changed; five is the classic one)", "Liam Gallagher", "Oasis", "5",
         ["Muse", "Queen"]),
    band("N_BB_jungkook", "consulted", "Jungkook", "BTS", "7", ["Queen", "Muse"], role="is a member of", did="does"),
    band("N_BB_mercury", "consulted (answer is the default four)", "Freddie Mercury", "Queen", "4", ["Muse", "Oasis"]),
    band("N_BB_albarn", "consulted (answer is the default four)", "Damon Albarn", "Blur", "4", ["Muse", "Oasis"]),

    # ---- N_X: one-off shapes ---------------------------------------------------------------------------------
    count("N_X_carlsen", "consulted", "How many rows does the board of the game that Magnus Carlsen plays have?",
          "chess", "game", "8", "Which game does Magnus Carlsen play?", "How many rows does a chess board have?",
          "How many rows does a {} board have?", ["Go", "poker"],
          ["Go", "poker", "bridge", "tennis", "golf", "cricket", "rugby", "hockey"], "N_X"),
    count("N_X_perlman", "consulted (four strings is common)",
          "How many strings does the instrument that Itzhak Perlman plays have?", "violin", "instrument", "4",
          "Which instrument does Itzhak Perlman play?", "How many strings does a violin have?",
          "How many strings does a {} have?", ["guitar", "piano"],
          ["guitar", "piano", "flute", "drum", "trumpet", "organ", "horn", "bass"], "N_X"),
    count("N_X_mars", "consulted", "How many moons does the planet named after the Roman god of war have?",
          "Mars", "planet", "2", "Which planet is named after the Roman god of war?", "How many moons does Mars have?",
          "How many moons does {} have?", ["Earth", "Venus"],
          ["Earth", "Venus", "Mercury", "Jupiter", "Saturn", "Neptune", "Pluto"], "N_X"),
])

# Reference questions already in the bank, re-run as anchors (they reproduce the saved results or the pod differs):
#   T_place_states_austria (27B consulted 0.92), T_place_states_australia (4B consulted 0.57), V_T_basketball (4B 0.96)
