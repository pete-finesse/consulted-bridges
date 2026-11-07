"""Sentences for scorecard v2 (bridge_card2.py), grouped as in paper_set.py plus a group of
description riddles like spider. Every word that gets edited must be ONE token with a leading space;
the script checks this and drops (and reports) any that aren't.

Each spec:
  group, predict   group label and the prediction made before running
  user             the two-hop question (chat, no prefill, one-word answer)
  bridge           the hidden middle step's words (removed together)
  answer, answers  the right answer's key; key -> forms the model may write
  answer_dirs      words whose J-lens directions stand for the answer (for the answer control)
  hop1, hop1_forms a one-hop question whose answer IS the bridge, and the bridge's written forms
  hop2             a one-hop question from the bridge to the answer
  check            template asking the model what a swap word's answer is ("How many legs does a {} have?")
  same_kind        words of the bridge's kind that share its answer (other Austrian cities; other 8-legged animals)
  swaps            words to swap in for the bridge (the expected answer is the model's own, from `check`)
  q_start          first words of the question (edits act from here on)
"""
from __future__ import annotations

ONE = " Answer in one word."
NUM = {"8": ["Eight", "8"], "6": ["Six", "6"], "4": ["Four", "4"], "2": ["Two", "2"], "10": ["Ten", "10"],
       "0": ["Zero", "None", "0", "No"]}
NUM_DIRS = {"8": ["Eight", "eight"], "6": ["Six", "six"], "4": ["Four", "four"], "2": ["Two", "two"]}
COLOR = {c: [c] for c in ["Red", "Yellow", "Green", "Orange", "Purple", "Blue", "Brown", "White"]}
COUNTRY = {"Austria": ["Austria"], "Switzerland": ["Switzerland"], "Argentina": ["Argentina"], "Japan": ["Japan"],
           "France": ["France"], "Italy": ["Italy"], "Germany": ["Germany"], "Spain": ["Spain"], "Canada": ["Canada"],
           "USA": ["United States", "USA", "America", "US"], "UK": ["United Kingdom", "UK", "England", "Britain"],
           "China": ["China"], "India": ["India"], "Portugal": ["Portugal"], "Lebanon": ["Lebanon"],
           "Czech": ["Czech Republic", "Czechia", "Czech"], "Sweden": ["Sweden"], "Russia": ["Russia"],
           "Australia": ["Australia"], "Brazil": ["Brazil"]}
COUNTRY_DIRS = {"USA": ["United", "States", "America"], "UK": ["United", "Kingdom", "Britain", "England"],
                "Czech": ["Czech"]}
CITIES = {"Austria": ["Vienna", "Graz"], "Switzerland": ["Zurich", "Geneva", "Bern"], "Japan": ["Osaka", "Kyoto"],
          "France": ["Lyon", "Marseille", "Toulouse"], "Italy": ["Naples", "Florence", "Venice"],
          "USA": ["Boston", "Chicago", "Houston"], "Canada": ["Vancouver", "Ottawa", "Calgary"],
          "China": ["Beijing", "Guangzhou", "Shenzhen"], "Czech": ["Brno"], "Sweden": ["Gothenburg", "Malmo"],
          "Lebanon": ["Tripoli", "Sidon"], "UK": ["Manchester", "Liverpool", "Leeds"], "Argentina": ["Mendoza", "Cordoba"],
          "Germany": ["Hamburg", "Cologne"], "Spain": ["Valencia", "Seville"], "Australia": ["Perth", "Brisbane"]}
SWAP_CITIES = ["Munich", "Milan", "Lyon", "Barcelona", "Osaka", "Toronto", "Chicago", "Manchester", "Mumbai",
               "Shanghai", "Porto", "Moscow"]
CAPITALS = {"Italy": "Rome", "Russia": "Moscow", "Brazil": "Brasília", "Canada": "Ottawa", "France": "Paris",
            "Germany": "Berlin", "Japan": "Tokyo", "Spain": "Madrid", "Egypt": "Cairo", "Poland": "Warsaw",
            "Greece": "Athens", "India": "Delhi"}
CAPITAL_FORMS = {"Brasília": ["Brasília", "Brasilia"], "Delhi": ["New Delhi", "Delhi"]}


def _country_answers(extra=()):
    return {k: v for k, v in COUNTRY.items()}


def legs(name, group, predict, user, bridge, ans, hop1, hop1_forms, same_kind, swaps):
    return name, dict(group=group, predict=predict, user=user + ONE, bridge=bridge, answer=ans, answers=NUM,
                      answer_dirs=NUM_DIRS[ans], hop1=hop1 + ONE, hop1_forms=hop1_forms,
                      hop2=f"How many legs does a {bridge[0]} have?" + ONE,
                      check="How many legs does a {} have?" + ONE, same_kind=same_kind, swaps=swaps,
                      q_start=user.split()[0] + " " + user.split()[1])


def color(name, group, predict, user, bridge, ans, hop1, hop1_forms, same_kind, swaps):
    return name, dict(group=group, predict=predict, user=user + ONE, bridge=bridge, answer=ans, answers=COLOR,
                      answer_dirs=[ans, ans.lower()], hop1=hop1 + ONE, hop1_forms=hop1_forms,
                      hop2=f"What color are {bridge[-1]}?" + ONE, check="What color is a {}?" + ONE,
                      same_kind=same_kind, swaps=swaps, q_start=user.split()[0] + " " + user.split()[1])


def born(name, group, predict, person, city, country):
    user = f"In which country is the city where {person} was born?"
    return name, dict(group=group, predict=predict, user=user + ONE, bridge=[city], answer=country,
                      answers=COUNTRY, answer_dirs=COUNTRY_DIRS.get(country, [COUNTRY[country][0]]),
                      hop1=f"In which city was {person} born?" + ONE, hop1_forms=[city],
                      hop2=f"In which country is {city}?" + ONE, check="In which country is {}?" + ONE,
                      same_kind=[c for c in CITIES.get(country, []) if c != city],
                      swaps=[c for c in SWAP_CITIES if c != city][:8] + [c for c in CITIES.get(country, []) if c != city][:2],
                      q_start="In which")


def univ(name, group, predict, uni, city, country):
    user = f"In which country is the city where {uni} has its headquarters?"
    return name, dict(group=group, predict=predict, user=user + ONE, bridge=[city], answer=country,
                      answers=COUNTRY, answer_dirs=COUNTRY_DIRS.get(country, [COUNTRY[country][0]]),
                      hop1=f"In which city does {uni} have its headquarters?" + ONE, hop1_forms=[city],
                      hop2=f"In which country is {city}?" + ONE, check="In which country is {}?" + ONE,
                      same_kind=[c for c in CITIES.get(country, []) if c != city],
                      swaps=[c for c in SWAP_CITIES if c != city][:8] + [c for c in CITIES.get(country, []) if c != city][:2],
                      q_start="In which")


def landmark(name, group, predict, place, country):
    cap = CAPITALS[country]
    answers = {CAPITALS[c]: CAPITAL_FORMS.get(CAPITALS[c], [CAPITALS[c]]) for c in CAPITALS}
    user = f"What is the capital of the country where {place} is located?"
    return name, dict(group=group, predict=predict, user=user + ONE, bridge=[country], answer=cap, answers=answers,
                      answer_dirs=[cap.split()[-1]], hop1=f"In which country is {place} located?" + ONE,
                      hop1_forms=[country], hop2=f"What is the capital of {country}?" + ONE,
                      check="What is the capital of {}?" + ONE, same_kind=[],
                      swaps=[c for c in ["Germany", "Japan", "Spain", "Egypt", "Poland", "Greece"] if c != country],
                      q_start="What is")


SPECS = dict([
    # E: description riddles (the bridge is worked out from a description, like spider) -> expect CONSULTED
    legs("E_spider", "E", "consulted", "How many legs do the animals that spin webs have?", ["spider", "spiders"], "8",
         "Which animals spin webs?", ["Spiders", "Spider"], ["tick", "ticks"],
         ["bee", "ant", "dog", "bird", "snake", "horse", "beetle", "tick"]),
    legs("E_bee", "E", "consulted", "How many legs do the insects that make honey have?", ["bee", "bees"], "6",
         "Which insects make honey?", ["Bees", "Bee", "Honeybees", "Honeybee"], ["ant", "beetle"],
         ["spider", "dog", "bird", "snake", "horse", "ant", "beetle"]),
    legs("E_penguin", "E", "consulted", "How many legs do the flightless birds that live in Antarctica have?",
         ["penguin", "penguins"], "2", "Which flightless birds live in Antarctica?", ["Penguins", "Penguin"],
         ["chicken", "duck"], ["spider", "ant", "dog", "snake", "horse", "chicken", "duck"]),
    legs("E_elephant", "E", "untestable (four is the default animal answer)",
         "How many legs does the animal with a trunk and tusks have?", ["elephant", "elephants"], "4",
         "Which animal has a trunk and tusks?", ["Elephant", "Elephants"], ["horse", "lion"],
         ["spider", "ant", "bird", "snake", "horse", "dog"]),
    color("E_apple", "E", "echo (red is tied to apple directly)", "What color is the fruit that keeps the doctor away?",
          ["apple", "apples"], "Red", "Which fruit keeps the doctor away?", ["Apple", "Apples", "An apple"],
          ["cherry", "strawberry"], ["banana", "lime", "grape", "lemon", "cherry", "strawberry"]),
    color("E_banana", "E", "consulted", "What color is the fruit that monkeys are famous for eating?",
          ["banana", "bananas"], "Yellow", "Which fruit are monkeys famous for eating?", ["Bananas", "Banana"],
          ["lemon"], ["apple", "lime", "grape", "cherry", "lemon"]),
    color("E_carrot", "E", "consulted", "What color is the vegetable that rabbits are famous for eating?",
          ["carrot", "carrots"], "Orange", "Which vegetable are rabbits famous for eating?", ["Carrots", "Carrot"],
          ["pumpkin"], ["tomato", "banana", "lime", "grape", "pumpkin"]),
    # A: landmark -> country -> capital (country bridges)
    landmark("A1_colosseum", "A1", "bystander", "the Colosseum", "Italy"),
    landmark("A1_kremlin", "A1", "bystander", "the Kremlin", "Russia"),
    landmark("A2_christ_redeemer", "A2", "consulted", "the Christ the Redeemer statue", "Brazil"),
    landmark("A2_cn_tower", "A2", "consulted", "the CN Tower", "Canada"),
    # B1: born in the capital
    born("B1_mishima", "B1", "echo", "Yukio Mishima", "Tokyo", "Japan"),
    born("B1_piaf", "B1", "echo", "Édith Piaf", "Paris", "France"),
    born("B1_loren", "B1", "echo", "Sophia Loren", "Rome", "Italy"),
    # B2: famous, nationality = birth country, non-capital city
    born("B2_mozart", "B2", "echo", "Wolfgang Amadeus Mozart", "Salzburg", "Austria"),
    born("B2_federer", "B2", "echo", "Roger Federer", "Basel", "Switzerland"),
    born("B2_messi", "B2", "echo", "Lionel Messi", "Rosario", "Argentina"),
    born("B2_gates", "B2", "echo", "Bill Gates", "Seattle", "USA"),
    born("B2_maldini", "B2", "echo", "Paolo Maldini", "Milan", "Italy"),
    # B3: born outside the country they are tied to
    born("B3_keanu", "B3", "consulted", "Keanu Reeves", "Beirut", "Lebanon"),
    born("B3_bieber", "B3", "consulted (London, Ontario)", "Justin Bieber", "London", "Canada"),
    # C1: university in a capital
    univ("C1_charles", "C1", "echo", "Charles University", "Prague", "Czech"),
    univ("C1_karolinska", "C1", "echo", "the Karolinska Institute", "Stockholm", "Sweden"),
    univ("C1_tsinghua", "C1", "echo", "Tsinghua University", "Beijing", "China"),
    # C2: university in a non-capital city
    univ("C2_fudan", "C2", "bystander", "Fudan University", "Shanghai", "China"),
    univ("C2_hopkins", "C2", "bystander", "Johns Hopkins University", "Baltimore", "USA"),
    univ("C2_york", "C2", "consulted (a naive 'York' reading gives UK)", "York University", "Toronto", "Canada"),
    univ("C2_mcgill", "C2", "bystander", "McGill University", "Montreal", "Canada"),
])
