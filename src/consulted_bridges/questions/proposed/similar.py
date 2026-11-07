"""PROPOSED, not yet run or added to any set: close variants of the questions that already score well.

Families (ids prefixed V_):
  V_LM  letter of the country where a landmark is            (like the Colosseum letter: 0.95 4B / 0.98 27B)
  V_LP  letter of the city where a person was born           (like Gates 0.73/0.96, Piaf -/0.96, Mozart 0.66/0.56)
  V_LU  letter of the city where a university is             (like York University 0.91 4B)
  V_T   players per team in a sport reached by a description (like hockey 0.74 4B)
  V_D   days in the month of a holiday                       (like Thanksgiving days 0.52 27B)
  V_R   legs of an animal reached by a description           (like spider 0.95 4B)

Every edited word is a single token with a leading space (checked by scripts/proposed/check_similar.py).
Same-kind controls are words of the bridge's kind with a DIFFERENT answer (as Vienna for Salzburg).
"""
from __future__ import annotations

import sys
from pathlib import Path

from consulted_bridges.questions.core import CITY_SWAPS, ONE, base, letter  # noqa: E402

COUNTRY_SWAPS = ["Germany", "Japan", "Spain", "Brazil", "Canada", "Egypt", "Mexico", "Russia"]


def landmark_letter(name, predict, place, country, same_kind):
    swaps = [c for c in COUNTRY_SWAPS if c[0] != country[0]][:6]
    return letter(name, predict, f"the country where {place} is located", [country], "country", same_kind, swaps,
                  f"In which country is {place} located?")


def born_letter(name, predict, person, city, same_kind):
    swaps = [c for c in CITY_SWAPS if c[0] != city[0]][:7]
    return letter(name, predict, f"the city where {person} was born", [city], "city", same_kind, swaps,
                  f"In which city was {person} born?")


def univ_letter(name, predict, uni, city, same_kind):
    swaps = [c for c in CITY_SWAPS if c[0] != city[0]][:7]
    return letter(name, predict, f"the city where {uni} has its headquarters", [city], "city", same_kind, swaps,
                  f"In which city does {uni} have its headquarters?")


TEAM = {"5": ["Five", "5"], "6": ["Six", "6"], "9": ["Nine", "9"], "11": ["Eleven", "11"], "4": ["Four", "4"],
        "2": ["Two", "2"], "1": ["One", "1"]}
TEAM_DIRS = {"5": ["Five", "five"], "6": ["Six", "six"], "9": ["Nine", "nine"], "11": ["Eleven", "eleven"],
             "4": ["Four", "four"]}


def team(name, predict, question, sport, ans, hop1, same_kind, swaps):
    return name, base(question + ONE, [sport], "sport", ans, TEAM, TEAM_DIRS[ans], hop1 + ONE, [sport.capitalize(), sport],
                      f"How many players does each team have playing at once in {sport}?" + ONE,
                      "How many players does each team have playing at once in {}?" + ONE, same_kind, swaps,
                      "V_T", predict, " ".join(question.split()[:2]))


DAYS = {"28": ["28", "Twenty-eight"], "30": ["30", "Thirty"], "31": ["31", "Thirty-one"], "29": ["29"]}


def days(name, predict, holiday_phrase, month, ans, same_kind, swaps):
    user = f"How many days are in the month of {holiday_phrase}?" + ONE
    return name, base(user, [month], "month", ans, DAYS, ["Thirty"] if ans != "28" else ["Twenty"],
                      f"In which month is {holiday_phrase}?" + ONE, [month],
                      f"How many days are in {month}?" + ONE, "How many days are in {}?" + ONE,
                      same_kind, swaps, "V_D", predict, "How many")


def legs(name, predict, description, animal, ans, hop1, same_kind, swaps):
    nums = {"8": ["Eight", "8"], "6": ["Six", "6"], "4": ["Four", "4"], "2": ["Two", "2"], "0": ["Zero", "None", "0"]}
    dirs = {"8": ["Eight", "eight"], "6": ["Six", "six"]}
    user = f"How many legs do {description} have?" + ONE
    return name, base(user, [animal], "animal", ans, nums, dirs[ans], hop1 + ONE, [animal.capitalize(), animal],
                      f"How many legs does a {animal} have?" + ONE, "How many legs does a {} have?" + ONE,
                      same_kind, swaps, "V_R", predict, "How many")


SPECS = dict([
    # V_LM: letter of a landmark's country. Same-kind controls: neighbouring countries with other letters.
    landmark_letter("V_LM_eiffel", "consulted", "the Eiffel Tower", "France", ["Germany", "Spain"]),
    landmark_letter("V_LM_pisa", "consulted (trap: Pisa starts with P)", "the Leaning Tower of Pisa", "Italy", ["Spain", "France"]),
    landmark_letter("V_LM_taj", "consulted", "the Taj Mahal", "India", ["China", "Nepal"]),
    landmark_letter("V_LM_kremlin", "consulted", "the Kremlin", "Russia", ["Poland", "China"]),
    landmark_letter("V_LM_sydney", "consulted (trap: Sydney starts with S)", "the Sydney Opera House", "Australia", ["Indonesia", "Japan"]),
    landmark_letter("V_LM_christ", "consulted", "the Christ the Redeemer statue", "Brazil", ["Argentina", "Peru"]),
    landmark_letter("V_LM_machu", "consulted", "Machu Picchu", "Peru", ["Chile", "Brazil"]),
    landmark_letter("V_LM_pyramids", "consulted", "the Pyramids of Giza", "Egypt", ["Morocco", "Jordan"]),
    landmark_letter("V_LM_fuji", "consulted", "Mount Fuji", "Japan", ["China", "Vietnam"]),
    landmark_letter("V_LM_acropolis", "consulted", "the Acropolis", "Greece", ["Italy", "Turkey"]),
    landmark_letter("V_LM_sagrada", "consulted", "the Sagrada Familia", "Spain", ["Portugal", "France"]),
    landmark_letter("V_LM_angkor", "consulted", "Angkor Wat", "Cambodia", ["Thailand", "Vietnam"]),
    landmark_letter("V_LM_brandenburg", "consulted", "the Brandenburg Gate", "Germany", ["Austria", "Poland"]),
    landmark_letter("V_LM_petronas", "consulted", "the Petronas Towers", "Malaysia", ["Thailand", "Indonesia"]),
    # V_LP: letter of a person's birth city. Same-kind controls: other cities of the same country, other letters.
    born_letter("V_LP_obama_m", "consulted", "Michelle Obama", "Chicago", ["Boston", "Denver"]),
    born_letter("V_LP_beyonce", "consulted", "Beyoncé", "Houston", ["Dallas", "Austin"]),
    born_letter("V_LP_hemsworth", "consulted", "Chris Hemsworth", "Melbourne", ["Sydney", "Perth"]),
    born_letter("V_LP_ali", "consulted", "Muhammad Ali", "Louisville", ["Nashville", "Denver"]),
    born_letter("V_LP_timberlake", "consulted", "Justin Timberlake", "Memphis", ["Atlanta", "Denver"]),
    born_letter("V_LP_plummer", "consulted", "Christopher Plummer", "Toronto", ["Vancouver", "Ottawa"]),
    born_letter("V_LP_hendrix", "consulted", "Jimi Hendrix", "Seattle", ["Boston", "Denver"]),
    born_letter("V_LP_kapoor", "consulted", "Kareena Kapoor", "Mumbai", ["Delhi", "Chennai"]),
    # V_LU: letter of a university's city.
    univ_letter("V_LU_berklee", "consulted", "Berklee College of Music", "Boston", ["Chicago", "Denver"]),
    univ_letter("V_LU_fudan", "consulted", "Fudan University", "Shanghai", ["Beijing"]),
    univ_letter("V_LU_pompeu", "consulted", "Pompeu Fabra University", "Barcelona", ["Madrid", "Valencia"]),
    univ_letter("V_LU_hopkins", "consulted", "Johns Hopkins University", "Baltimore", ["Chicago", "Denver"]),
    univ_letter("V_LU_alberta", "consulted", "the University of Alberta", "Edmonton", ["Calgary", "Vancouver"]),
    univ_letter("V_LU_dalhousie", "consulted", "Dalhousie University", "Halifax", ["Toronto", "Vancouver"]),
    # V_T: players per team. Same-kind controls: other sports with other team sizes.
    team("V_T_hockey2", "consulted (close variant of hockey)", "How many players does each team have on the ice at once in the sport where players skate and shoot a puck?",
         "hockey", "6", "Which sport has players skate and shoot a puck?", ["tennis", "golf"], ["basketball", "baseball", "cricket", "polo", "tennis"]),
    team("V_T_basketball", "consulted", "How many players does each team have on the court at once in the sport invented by James Naismith?",
         "basketball", "5", "Which sport did James Naismith invent?", ["tennis", "baseball"], ["hockey", "baseball", "cricket", "polo", "tennis"]),
    team("V_T_cricket", "consulted", "How many players does each team have on the field at once in the sport played with a bat, a ball and wickets?",
         "cricket", "11", "Which sport is played with a bat, a ball and wickets?", ["baseball", "golf"], ["hockey", "basketball", "baseball", "polo", "tennis"]),
    team("V_T_baseball", "consulted (the question says four bases; the answer is nine)", "How many players does each team have on the field at once in the sport where players hit home runs and run around four bases?",
         "baseball", "9", "Which sport has players hit home runs and run around four bases?", ["cricket", "golf"], ["hockey", "basketball", "cricket", "polo", "tennis"]),
    team("V_T_polo", "consulted", "How many players does each team have on the field at once in the sport played on horseback with mallets?",
         "polo", "4", "Which sport is played on horseback with mallets?", ["hockey", "golf"], ["hockey", "basketball", "baseball", "cricket"]),
    # V_D: days in a holiday's month. Same-kind controls: other months with other day counts.
    days("V_D_valentine", "consulted (28 differs from 30/31 in its first token)", "Valentine's Day", "February", "28",
         ["April", "June"], ["April", "June", "January", "March", "September"]),
    days("V_D_christmas", "consulted (risk: 31 and 30 share their first token)", "Christmas", "December", "31",
         ["November", "February"], ["November", "February", "April", "June", "March"]),
    days("V_D_july4", "consulted (risk: as above)", "American Independence Day", "July", "31",
         ["June", "February"], ["June", "February", "April", "September", "August"]),
    # V_R: legs of an animal reached by a description (4B only so far).
    legs("V_R_tick", "consulted", "the animals whose bites can spread Lyme disease", "tick", "8",
         "Which animals spread Lyme disease with their bites?", ["ants", "dogs"], ["ant", "dog", "bird", "beetle", "horse"]),
    legs("V_R_ant", "consulted", "the animals that farm aphids and live in colonies", "ants", "6",
         "Which animals farm aphids and live in colonies?", ["spiders", "dogs"], ["spider", "dog", "bird", "horse", "tick"]),
])
