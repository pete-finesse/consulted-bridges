"""Idea set: order, sequences, relations and simple arithmetic on a hidden item.

The hidden bridge is a planet, month, weekday, letter, number or character; the answer is its neighbour in a
sequence, its position, a count about it, a small sum on it, or a relative of it. Fields and the base() helper
as in card3_specs.py. Two questions are deliberately bad (predict starts with "bad:"):
  Q_month_christmas_next  shortcut: "after Christmas" -> New Year -> January without going through December
  Q_color_snow_opposite   echo: black is the top word associate of white (the bridge), and a default for "opposite"
Note the sequence questions (next month/day/planet/letter, +1) have an answer adjacent to the bridge in a
sequence: a possible echo risk, flagged per question in sequence_NOTES.md.
"""
from __future__ import annotations

from consulted_bridges.questions.core import LETTER_ONE, LETTERS, ONE, base

PLANET_LIST = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
PLANETS = {p: [p] for p in PLANET_LIST}
MONTH_LIST = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
MONTHS = {m: [m] for m in MONTH_LIST}
DAY_LIST = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAYS = {d: [d] for d in DAY_LIST}
COLORS = {c: [c] for c in ["Black", "White", "Red", "Green", "Blue", "Orange", "Yellow", "Purple", "Gray", "Grey",
                           "Brown", "Pink"]}
FATHERS = {"James": ["James"], "Homer": ["Homer"], "Anakin": ["Anakin", "Darth Vader", "Darth", "Vader"],
           "Abraham": ["Abraham"], "Odin": ["Odin"], "Lucius": ["Lucius"], "Arthur": ["Arthur"],
           "Vernon": ["Vernon"]}

_CARD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
         11: "Eleven", 12: "Twelve", 16: "Sixteen", 20: "Twenty", 28: "Twenty-eight", 30: "Thirty", 31: "Thirty-one"}
_ORD = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth", 6: "Sixth", 7: "Seventh", 8: "Eighth",
        9: "Ninth", 10: "Tenth", 11: "Eleventh", 12: "Twelfth"}
_SUF = {1: "st", 2: "nd", 3: "rd"}


def _num_forms(n):
    f = [_CARD[n], str(n)]
    if "-" in _CARD[n]:
        f.append(_CARD[n].replace("-", " "))
    if n in _ORD:
        f += [_ORD[n], f"{n}{_SUF.get(n, 'th')}"]  # ordinals only go up to 12
    return f


# Order matters for classify(): keys whose words start with another key's word go first
# ("thirty-one" before "thirty", "twenty-eight" before "twenty").
NUMS = {str(n): _num_forms(n) for n in [31, 28, 30, 20, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}


def _dirs(n):
    return [_CARD[n], _CARD[n].lower()]


SPECS = dict([
    # --- sequences: the neighbour of a hidden item ---------------------------------------------------------
    ("Q_planet_red_next", base(
        "Which planet comes right after the Red Planet in order from the Sun?" + ONE, ["Mars"], "planet",
        "Jupiter", PLANETS, ["Jupiter"], "Which planet is known as the Red Planet?" + ONE, ["Mars"],
        "Which planet comes right after Mars in order from the Sun?" + ONE,
        "Which planet comes right after {} in order from the Sun?" + ONE, ["Neptune", "Venus"],
        ["Mercury", "Venus", "Earth", "Jupiter", "Saturn"], "S", "consulted", "Which planet")),
    ("Q_month_valentine_next", base(
        "Which month comes right after the month of Valentine's Day?" + ONE, ["February"], "month",
        "March", MONTHS, ["March"], "In which month is Valentine's Day?" + ONE, ["February"],
        "Which month comes right after February?" + ONE, "Which month comes right after {}?" + ONE,
        ["July", "May"], ["January", "March", "June", "September", "October", "December"], "S", "consulted",
        "Which month")),
    ("Q_day_moon_next", base(
        "Which day of the week comes right after the day named after the Moon?" + ONE, ["Monday"], "day",
        "Tuesday", DAYS, ["Tuesday"], "Which day of the week is named after the Moon?" + ONE, ["Monday"],
        "Which day of the week comes right after Monday?" + ONE, "Which day of the week comes right after {}?" + ONE,
        ["Friday", "Wednesday"], ["Sunday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], "S",
        "consulted", "Which day")),
    ("Q_letter_last_before", base(
        "Which letter of the alphabet comes right before the last letter?" + LETTER_ONE, ["Z"], "letter",
        "Y", LETTERS, ["Y"], "What is the last letter of the alphabet?" + LETTER_ONE, ["Z"],
        "Which letter of the alphabet comes right before the letter Z?" + LETTER_ONE,
        "Which letter of the alphabet comes right before the letter {}?" + LETTER_ONE, ["X", "Q"],
        ["B", "D", "N", "T", "K", "P"], "S", "consulted", "Which letter")),
    # --- positions and counts of a hidden item -------------------------------------------------------------
    ("Q_planet_rings_position", base(
        "Counting outward from the Sun, what number planet is the one famous for its rings?" + ONE, ["Saturn"],
        "planet", "6", NUMS, ["Sixth", "sixth", "Six"], "Which planet is famous for its rings?" + ONE, ["Saturn"],
        "Counting outward from the Sun, what number planet is Saturn?" + ONE,
        "Counting outward from the Sun, what number planet is {}?" + ONE, ["Jupiter", "Neptune"],
        ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Neptune"], "P", "consulted", "Counting outward")),
    ("Q_month_halloween_number", base(
        "What number month of the year is the month of Halloween?" + ONE, ["October"], "month",
        "10", NUMS, _dirs(10), "In which month is Halloween?" + ONE, ["October"],
        "What number month of the year is October?" + ONE, "What number month of the year is {}?" + ONE,
        ["November", "August"], ["January", "March", "May", "July", "September", "December"], "P", "consulted",
        "What number")),
    ("Q_month_thanksgiving_days", base(
        "How many days are in the month when Americans celebrate Thanksgiving?" + ONE, ["November"], "month",
        "30", NUMS, _dirs(30), "In which month do Americans celebrate Thanksgiving?" + ONE, ["November"],
        "How many days are in the month of November?" + ONE, "How many days are in the month of {}?" + ONE,
        ["October", "May"], ["January", "February", "March", "July", "December", "April", "June"], "P",
        "consulted", "How many")),
    # --- arithmetic on a hidden number ---------------------------------------------------------------------
    ("Q_num_week_plus1", base(
        "What number is one more than the number of days in a week?" + ONE, ["seven", "Seven"], "number",
        "8", NUMS, _dirs(8), "How many days are in a week?" + ONE, ["Seven", "seven", "7"],
        "What number is one more than seven?" + ONE, "What number is one more than {}?" + ONE,
        ["three", "six"], ["two", "four", "five", "nine", "ten", "eleven"], "A", "consulted", "What number")),
    ("Q_num_triangle_double", base(
        "What number is twice the number of sides of a triangle?" + ONE, ["three", "Three"], "number",
        "6", NUMS, _dirs(6), "How many sides does a triangle have?" + ONE, ["Three", "three", "3"],
        "What number is twice three?" + ONE, "What number is twice {}?" + ONE,
        ["seven", "nine"], ["two", "four", "five", "six", "eight", "ten"], "A", "consulted", "What number")),
    # --- family relations of a fictional figure ------------------------------------------------------------
    ("Q_father_wizard", base(
        "What is the first name of the father of the boy wizard with a lightning-shaped scar?" + ONE, ["Harry"],
        "character", "James", FATHERS, ["James"], "Who is the boy wizard with a lightning-shaped scar?" + ONE,
        ["Harry", "Harry Potter"], "What is the first name of Harry Potter's father?" + ONE,
        "What is the first name of the father of {}?" + ONE, ["Ron", "Hermione"],
        ["Bart", "Luke", "Isaac", "Thor", "Draco", "Lisa"], "F", "consulted", "What is")),
    # --- deliberately BAD ----------------------------------------------------------------------------------
    ("Q_month_christmas_next", base(
        "Which month comes right after the month of Christmas?" + ONE, ["December"], "month",
        "January", MONTHS, ["January"], "In which month is Christmas?" + ONE, ["December"],
        "Which month comes right after December?" + ONE, "Which month comes right after {}?" + ONE,
        ["July", "May"], ["January", "February", "March", "June", "September", "October"], "S",
        "bad: shortcut, 'after Christmas' leads to New Year and January without December", "Which month")),
    ("Q_color_snow_opposite", base(
        "What color is the opposite of the color of snow?" + ONE, ["white", "White"], "color",
        "Black", COLORS, ["Black", "black"], "What color is snow?" + ONE, ["White", "white"],
        "What color is the opposite of white?" + ONE, "What color is the opposite of {}?" + ONE,
        ["red", "blue"], ["black", "red", "green", "blue", "yellow"], "B",
        "bad: echo, black is the top associate of white (and the default 'opposite' colour)", "What color")),
])
