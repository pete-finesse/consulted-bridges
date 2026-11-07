"""Trend-based proposals (group "T"): new question TYPES that share what made the passing questions pass.

Not run yet. Reasoning, trends and per-question risks: docs/proposals/trend_questions.md.

What the passing questions share (numbers in the write-up):
  1. the bridge is a named place (city or country): the only bridge class the edit reliably removes in the 27B;
  2. the answer is a generic token with no meaning shared with the bridge (letters, small numbers), so it can't be
     an echo, and it is a fact about the bridge alone, not about its category;
  3. the answer needs a lookup after the bridge, so it forms late and the edit has room to act;
  4. the default guess (what the model says with the bridge removed) is a different answer;
  5. both hops are famous facts and the second is a lookup, not an operation on spelling.

None of these is "the first letter of the name of X" or "how many legs". Groups:
  T_place_*    place -> a number fact about that place (World Cup wins, hills, flag stars, languages, states)
  T_code_*     place -> an arbitrary code number (dialling code)
  T_emblem_*   place -> an arbitrary emblem word (national animal, flag creature)
  T_rule_*     place -> a two-way convention whose default is the other side (driving side)
  T_named_*    a named non-place (brand, cartoon family, king) -> a number fact
  T_fact_*     common noun -> a letter label attached by fact, not by spelling (vitamin)
  T_bad_*      controls designed to FAIL, one named reason each (predict starts with "bad:")

Every bridge, same-kind, swap and answer-direction word was checked to be one token with a leading space for the
Qwen3.5-4B tokenizer (the Qwen3.6-27B tokenizer is not cached locally, so it is unchecked). Run as
  cd harness && uv run python -c "import sys; sys.path[:0]=['scripts','scripts/proposed']; import trend_specs as s; print(len(s.SPECS))"
"""
from __future__ import annotations

from consulted_bridges.questions.core import LETTER_ONE, LETTERS, ONE, base

# number answers: word and digit forms. Only the right answer's forms feed P(answer); the rest are there so swaps
# and the fallback can be classified. Digit forms of 10+ start with the same first token as small digits
# (Qwen splits digits), so no right answer below is 10 or more except in the controls.
_W = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine",
      13: "Thirteen", 14: "Fourteen", 16: "Sixteen", 22: "Twenty-two", 26: "Twenty-six", 27: "Twenty-seven",
      28: "Twenty-eight", 31: "Thirty-one", 32: "Thirty-two", 41: "Forty-one", 50: "Fifty"}
NUMT = {str(k): [w, str(k)] for k, w in _W.items()}
NUMT["0"] += ["None", "No", "Never"]
NUMT["1"] += ["Once"]
NUMT["2"] += ["Twice"]


def _dirs(n):
    return [_W[int(n)], _W[int(n)].lower()]


DIAL = {k: [k] for k in ["7", "1", "33", "34", "39", "44", "48", "49", "55", "81", "86", "91", "380"]}
DIAL["7"] += ["Seven"]
EMBLEM = {"Unicorn": ["Unicorn"], "Kangaroo": ["Kangaroo"], "Panda": ["Panda", "Giant panda"],
          "Eagle": ["Eagle", "Bald eagle", "Golden eagle"], "Beaver": ["Beaver"],
          "Rooster": ["Rooster", "Gallic rooster", "Cockerel", "Cock"], "Tiger": ["Tiger", "Bengal tiger"],
          "Dragon": ["Dragon", "Red dragon", "Welsh dragon"], "Lion": ["Lion"], "Bear": ["Bear", "Brown bear"],
          "Leaf": ["Leaf", "Maple leaf", "Maple"], "Sun": ["Sun", "Red sun", "Circle", "Red circle", "Disc"],
          "Tree": ["Tree", "Cedar", "Cedar tree"], "Harp": ["Harp"], "Cross": ["Cross"]}
SIDE = {"Left": ["Left"], "Right": ["Right"]}
CONT = {"Europe": ["Europe"], "Asia": ["Asia"], "Africa": ["Africa"], "North America": ["North America"],
        "South America": ["South America"], "Oceania": ["Oceania", "Australia"]}


def _cap(words):
    return list(dict.fromkeys([w[0].upper() + w[1:] for w in words] + words))


def place_count(name, predict, user, bridge, kind, ans, hop1, hop2, check, same_kind, swaps, q_start="How many"):
    """place -> a number fact about that place."""
    return name, base(user + ONE, bridge, kind, ans, NUMT, _dirs(ans), hop1 + ONE, _cap(bridge), hop2 + ONE,
                      check + ONE, same_kind, swaps, "T", predict, q_start)


WC = "How many times has {} won the men's football World Cup?"
STARS = "How many stars are on the flag of {}?"

SPECS = dict([
    # ---- T_place: place -> a number fact about that place --------------------------------------------------
    place_count("T_place_wc_italy", "consulted",
                "How many times has the country where the Leaning Tower of Pisa stands won the men's football World Cup?",
                ["Italy"], "country", "4", "In which country does the Leaning Tower of Pisa stand?", WC.format("Italy"),
                WC, ["Spain", "France"],
                ["Brazil", "Argentina", "France", "Uruguay", "England", "Spain", "Germany", "Netherlands"]),
    place_count("T_place_wc_brazil", "consulted (lower need: Brazil may be the default World Cup country)",
                "How many times has the country where the city of Rio de Janeiro is located won the men's football World Cup?",
                ["Brazil"], "country", "5", "In which country is the city of Rio de Janeiro located?",
                WC.format("Brazil"), WC, ["Argentina", "Portugal"],
                ["Italy", "Argentina", "France", "Uruguay", "England", "Spain", "Germany", "Mexico"]),
    place_count("T_place_hills_rome", "consulted",
                "On how many hills was the city where the Trevi Fountain stands built?",
                ["Rome"], "city", "7", "In which city does the Trevi Fountain stand?",
                "On how many hills was Rome built?", "On how many hills was {} built?", ["Milan", "Naples"],
                ["Lisbon", "Istanbul", "Moscow", "Paris", "Athens", "Prague", "Madrid", "Seattle"], q_start="On how"),
    place_count("T_place_stars_china", "consulted",
                "How many stars are on the flag of the country where the Forbidden City is located?",
                ["China"], "country", "5", "In which country is the Forbidden City located?", STARS.format("China"),
                STARS, ["Japan", "India"],
                ["Vietnam", "Australia", "Japan", "Turkey", "Chile", "Ghana", "France", "America"]),
    place_count("T_place_stripes_greece", "consulted",
                "How many stripes are on the flag of the country where the Parthenon stands?",
                ["Greece"], "country", "9", "In which country does the Parthenon stand?",
                "How many stripes are on the flag of Greece?", "How many stripes are on the flag of {}?",
                ["Italy", "Turkey"], ["Thailand", "Cuba", "Uruguay", "Japan", "Malaysia", "America", "Chile"]),
    place_count("T_place_langs_switzerland", "consulted",
                "How many national languages does the country where the city of Geneva is located have?",
                ["Switzerland"], "country", "4", "In which country is the city of Geneva located?",
                "How many national languages does Switzerland have?", "How many national languages does {} have?",
                ["Austria", "Germany"],
                ["Belgium", "Canada", "Ireland", "Finland", "Singapore", "France", "Japan", "Germany"]),
    place_count("T_place_states_australia", "consulted",
                "How many states does the country where the Sydney Opera House stands have?",
                ["Australia"], "country", "6", "In which country does the Sydney Opera House stand?",
                "How many states does Australia have?", "How many states does {} have?", ["Canada", "Brazil"],
                ["Austria", "America", "Germany", "India", "Brazil", "Mexico", "Malaysia"]),
    place_count("T_place_states_austria", "consulted (Mozart -> Austria is the paper's own fact, new answer)",
                "How many federal states does the country where Wolfgang Amadeus Mozart was born have?",
                ["Austria"], "country", "9", "In which country was Wolfgang Amadeus Mozart born?",
                "How many federal states does Austria have?", "How many federal states does {} have?",
                ["Switzerland", "Germany"], ["Germany", "Australia", "America", "India", "Brazil", "Belgium", "Mexico"]),

    # ---- T_code: place -> an arbitrary code number --------------------------------------------------------
    ("T_code_dial_russia", base(
        "What is the international dialling code of the country where Red Square is? Answer with the number only, "
        "without a plus sign.", ["Russia"], "country", "7", DIAL, ["Seven", "seven"],
        "In which country is Red Square?" + ONE, ["Russia"],
        "What is the international dialling code of Russia? Answer with the number only, without a plus sign.",
        "What is the international dialling code of {}? Answer with the number only, without a plus sign.",
        ["Ukraine", "Poland"], ["Britain", "France", "Germany", "Japan", "Brazil", "India", "Italy", "Spain"], "T",
        "consulted (27B); 4B may not know the code", "What is")),

    # ---- T_emblem: place -> an arbitrary emblem word ------------------------------------------------------
    ("T_emblem_animal_scotland", base(
        "What is the national animal of the country whose capital is Edinburgh?" + ONE, ["Scotland"], "country",
        "Unicorn", EMBLEM, ["Unicorn", "unicorn"], "Which country has Edinburgh as its capital?" + ONE, ["Scotland"],
        "What is the national animal of Scotland?" + ONE, "What is the national animal of {}?" + ONE,
        ["Ireland", "Wales"], ["Australia", "China", "America", "Canada", "France", "India", "Wales", "England"], "T",
        "consulted (risk: category effect, as banana -> Yellow)", "What is")),
    ("T_emblem_animal_canada", base(
        "What is the national animal of the country where the city of Montreal is located?" + ONE, ["Canada"],
        "country", "Beaver", EMBLEM, ["Beaver"], "In which country is the city of Montreal located?" + ONE, ["Canada"],
        "What is the national animal of Canada?" + ONE, "What is the national animal of {}?" + ONE,
        ["Mexico", "Australia"], ["Australia", "China", "America", "Scotland", "France", "India", "Wales", "England"],
        "T", "consulted (risk: category effect)", "What is")),
    ("T_emblem_flag_wales", base(
        "What creature is shown on the flag of the country whose capital is Cardiff?" + ONE, ["Wales"], "country",
        "Dragon", EMBLEM, ["Dragon", "dragon"], "Which country has Cardiff as its capital?" + ONE, ["Wales"],
        "What creature is shown on the flag of Wales?" + ONE, "What creature is shown on the flag of {}?" + ONE,
        ["Scotland", "Ireland"], ["Mexico", "Albania", "Scotland", "England", "Canada", "Lebanon", "America"], "T",
        "consulted (risk: Dragon may be the default guess for a flag creature)", "What creature")),

    # ---- T_rule: place -> a two-way convention whose default is the other answer ---------------------------
    ("T_rule_drive_japan", base(
        "On which side of the road do people drive in the country where the city of Osaka is located?" + ONE,
        ["Japan"], "country", "Left", SIDE, ["Left", "left"], "In which country is the city of Osaka located?" + ONE,
        ["Japan"], "On which side of the road do people drive in Japan?" + ONE,
        "On which side of the road do people drive in {}?" + ONE, ["China", "Korea"],
        ["Britain", "India", "Australia", "France", "America", "Germany", "Brazil", "Italy"], "T",
        "consulted (two-way answer: 'Right' is the default, so removal should flip it)", "On which")),
    ("T_rule_drive_india", base(
        "On which side of the road do people drive in the country where the Taj Mahal stands?" + ONE,
        ["India"], "country", "Left", SIDE, ["Left", "left"], "In which country does the Taj Mahal stand?" + ONE,
        ["India"], "On which side of the road do people drive in India?" + ONE,
        "On which side of the road do people drive in {}?" + ONE, ["China", "Russia"],
        ["Britain", "Japan", "Australia", "France", "America", "Germany", "Brazil", "Italy"], "T",
        "consulted (two-way answer)", "On which")),

    # ---- T_named: a named non-place -> a number fact (does "named entity" carry the reach, not "place"?) ----
    ("T_named_logo_audi", base(
        "How many rings are in the logo of the car company based in Ingolstadt?" + ONE, ["Audi"], "company", "4",
        NUMT, _dirs("4"), "Which car company is based in Ingolstadt?" + ONE, ["Audi"],
        "How many rings are in the Audi logo?" + ONE, "How many rings are in the logo of {}?" + ONE,
        ["BMW", "Mercedes"], ["Olympics", "Mercedes", "BMW", "Toyota", "Chrysler"], "T",
        "consulted (default for rings in a logo is Five, the Olympics)", "How many")),
    ("T_named_logo_mercedes", base(
        "How many points does the star in the logo of the car maker that builds the S-Class have?" + ONE,
        ["Mercedes"], "company", "3", NUMT, _dirs("3"), "Which car maker builds the S-Class?" + ONE,
        ["Mercedes", "Mercedes-Benz"], "How many points does the star in the Mercedes logo have?" + ONE,
        "How many points does the star in the logo of {} have?" + ONE, ["BMW", "Audi"],
        ["Chrysler", "Subaru", "Tesla", "Audi"], "T",
        "consulted (default star has Five points)", "How many")),
    ("T_named_fingers_simpsons", base(
        "How many fingers are on each hand of the members of the yellow cartoon family that lives in Springfield?"
        + ONE, ["Simpsons", "Simpson"], "family", "4", NUMT, _dirs("4"),
        "Which yellow cartoon family lives in Springfield?" + ONE, ["Simpsons", "The Simpsons", "Simpson"],
        "How many fingers are on each hand of the Simpsons characters?" + ONE,
        "How many fingers are on each hand of {}?" + ONE, ["Mickey", "Bugs"],
        ["humans", "people", "monkeys", "Mickey", "children"], "T",
        "uncertain (risk: 'cartoons have four fingers' is a category shortcut)", "How many")),
    ("T_named_wives_henry", base(
        "How many wives did the English king who broke with the Pope to found the Church of England have?" + ONE,
        ["Henry"], "king", "6", NUMT, _dirs("6"),
        "Which English king broke with the Pope to found the Church of England?" + ONE,
        ["Henry", "Henry VIII", "King Henry VIII", "King Henry"], "How many wives did King Henry VIII have?" + ONE,
        "How many wives did King {} have?" + ONE, ["Edward", "Charles"],
        ["Edward", "George", "Charles", "William", "Richard", "John"], "T",
        "uncertain (person bridge; the token 'Henry' is not the entity Henry VIII)", "How many")),

    # ---- T_fact: a letter label attached by fact, not by spelling -----------------------------------------
    ("T_fact_vitamin_carrot", base(
        "Which vitamin is the orange root vegetable that rabbits love best known for?" + LETTER_ONE,
        ["carrot", "carrots"], "vegetable", "A", LETTERS, ["A"],
        "Which orange root vegetable do rabbits love?" + ONE, _cap(["carrot", "carrots"]),
        "Which vitamin is the carrot best known for?" + LETTER_ONE,
        "Which vitamin is the {} best known for?" + LETTER_ONE, ["potato", "onion"],
        ["orange", "lemon", "spinach", "fish", "milk", "banana", "egg", "liver"], "T",
        "consulted in 4B; 27B may miss on reach (food bridge)", "Which vitamin")),

    # ---- T_bad: controls designed to fail, one named reason each ------------------------------------------
    # gate 1: an operation on the bridge's spelling other than its first letter (1 of 9 such questions pass gate 1)
    place_count("T_bad_len_italy", "bad: gate 1 (counting letters is an operation the models don't chain)",
                "How many letters are in the name of the country where the Colosseum is located?",
                ["Italy"], "country", "5", "In which country is the Colosseum located?",
                "How many letters are in the word Italy?", "How many letters are in the word {}?",
                ["Spain", "France"], ["Spain", "France", "Germany", "Japan", "Brazil", "Canada", "Peru", "Egypt"]),
    # default guess: Fifty is what "stars on a flag" gives with no country at all
    ("T_bad_stars_usa", base(
        "How many stars are on the flag of the country where the Statue of Liberty stands?" + ONE,
        ["America", "USA"], "country", "50", NUMT, ["Fifty", "fifty"],
        "In which country does the Statue of Liberty stand?" + ONE,
        ["America", "USA", "United States", "US", "the United States", "the USA"],
        "How many stars are on the flag of the United States?" + ONE, STARS + ONE, ["Canada", "Mexico"],
        ["China", "Vietnam", "Australia", "Japan", "Turkey", "Chile", "Brazil"], "T",
        "bad: default guess (Fifty without the bridge, so need ~0)", "How many")),
    # echo: the answer is a category that contains the bridge (country answers: isolation 0.03-0.18)
    ("T_bad_continent_italy", base(
        "On which continent is the country where the Colosseum is located?" + ONE, ["Italy"], "country", "Europe",
        CONT, ["Europe", "European"], "In which country is the Colosseum located?" + ONE, ["Italy"],
        "On which continent is Italy?" + ONE, "On which continent is {}?" + ONE, ["Spain", "France"],
        ["Japan", "Brazil", "Egypt", "China", "Kenya", "Australia", "Canada", "Spain"], "T",
        "bad: answer echo (Europe is a category containing Italy)", "On which")),
    # shortcut: "scurvy" gives vitamin C directly, without the fruit
    ("T_bad_vitamin_scurvy", base(
        "Which vitamin is the fruit that sailors ate to prevent scurvy rich in?" + LETTER_ONE, ["lemon", "lime"],
        "fruit", "C", LETTERS, ["C"], "Which fruit did sailors eat to prevent scurvy?" + ONE,
        ["Lemon", "Lemons", "Lime", "Limes", "lemon", "lime", "lemons", "limes", "Citrus", "Oranges", "Orange"],
        "Which vitamin is the lemon rich in?" + LETTER_ONE, "Which vitamin is the {} best known for?" + LETTER_ONE,
        ["apple", "grape"], ["carrot", "spinach", "fish", "milk", "banana", "egg"], "T",
        "bad: shortcut (scurvy -> C is stored directly; expect bystander)", "Which vitamin")),
])
