"""The paper's two-hop facts, grouped, with a small hand-picked set for the bridge scorecard.

The paper (Qwen3.5-4B) used three categories; the other three models also used person -> university.
Groups below split each category by what could let the model skip the bridge (a shortcut) or make
the bridge's direction mostly the answer's (an echo). Each pick has a prediction made BEFORE running.

  A  landmark -> country -> capital          (the authors' 48 hand-made facts)
     A1 the landmark is in the capital       (Eiffel Tower: the landmark leads to the capital directly)
     A2 the landmark is in another famous city that is NOT the capital (Sydney Opera House -> Canberra)
  B  person -> birth city -> country         (TwoHopFact person-birthcity-cntry)
     B1 the birth city is the capital        (Mishima -> Tokyo)
     B2 famous, nationality = birth country, non-capital city (Mozart -> Salzburg)
     B3 the birth country differs from what the person is mostly tied to (Keanu Reeves -> Beirut -> Lebanon)
  C  university -> HQ city -> country        (TwoHopFact univ-hqcity-cntry)
     C1 the city is a capital / mega-city    (Tsinghua -> Beijing)
     C2 a non-capital city, possibly ambiguous (Harvard -> Cambridge, McGill -> Montreal)
  D  person -> undergrad university -> city  (TwoHopFact person-uguniv-hqcity; 27B's category)
     D1 the university's name doesn't name the city (Larry Page -> University of Michigan -> Ann Arbor)
"""
from __future__ import annotations

# answer key -> forms the model may write
COUNTRY_FORMS = {
    "Austria": ["Austria"], "Japan": ["Japan"], "France": ["France"], "Germany": ["Germany"], "Italy": ["Italy"],
    "Spain": ["Spain"], "Canada": ["Canada"], "USA": ["United States", "USA", "America", "US"],
    "UK": ["United Kingdom", "UK", "England", "Britain"], "China": ["China"], "India": ["India"],
    "Lebanon": ["Lebanon"], "Georgia": ["Georgia"], "Argentina": ["Argentina"], "Portugal": ["Portugal"],
    "Serbia": ["Serbia"], "Switzerland": ["Switzerland"], "Russia": ["Russia"], "Australia": ["Australia"],
    "Brazil": ["Brazil"], "Turkey": ["Turkey"], "Cuba": ["Cuba"], "Poland": ["Poland"], "Mexico": ["Mexico"],
    "Netherlands": ["Netherlands"], "Egypt": ["Egypt"],
}
DATASET_NAME = {"United States of America": "USA", "United Kingdom": "UK", "People's Republic of China": "China"}
# well-known cities per country (for same-kind controls and same-answer swaps); first = capital
CITIES = {
    "Austria": ["Vienna", "Graz", "Innsbruck", "Linz"], "Japan": ["Tokyo", "Osaka", "Kyoto", "Yokohama"],
    "France": ["Paris", "Lyon", "Marseille", "Toulouse"], "USA": ["Washington", "Boston", "Chicago", "Houston"],
    "Canada": ["Ottawa", "Toronto", "Vancouver", "Calgary"], "China": ["Beijing", "Shanghai", "Guangzhou", "Shenzhen"],
    "India": ["Delhi", "Mumbai", "Kolkata", "Chennai"], "Lebanon": ["Beirut", "Tripoli", "Sidon", "Tyre"],
    "Georgia": ["Tbilisi", "Batumi", "Kutaisi", "Rustavi"], "Argentina": ["Buenos", "Córdoba", "Mendoza", "Rosario"],
    "Portugal": ["Lisbon", "Porto", "Braga", "Coimbra"], "Serbia": ["Belgrade", "Novi", "Niš", "Kragujevac"],
    "Switzerland": ["Bern", "Zurich", "Geneva", "Basel"], "Russia": ["Moscow", "Kazan", "Novosibirsk", "Samara"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Cologne"], "UK": ["London", "Manchester", "Liverpool", "Leeds"],
}
# cities of other countries, for swaps that should change the answer
SWAP_CITIES = {"Munich": "Germany", "Milan": "Italy", "Lyon": "France", "Barcelona": "Spain", "Osaka": "Japan",
               "Toronto": "Canada", "Chicago": "USA", "Manchester": "UK", "Mumbai": "India", "Shanghai": "China",
               "Porto": "Portugal", "Kazan": "Russia", "Graz": "Austria"}
CAPITAL = {"France": "Paris", "Australia": "Canberra", "Canada": "Ottawa", "Brazil": "Brasília", "Turkey": "Ankara",
           "India": "New Delhi", "Italy": "Rome", "China": "Beijing", "Russia": "Moscow", "Japan": "Tokyo",
           "Spain": "Madrid", "Germany": "Berlin", "Egypt": "Cairo", "Poland": "Warsaw", "Mexico": "Mexico City",
           "Netherlands": "Amsterdam", "Switzerland": "Bern", "Austria": "Vienna", "Greece": "Athens"}
CAPITAL_FORMS = {"New Delhi": ["New Delhi", "Delhi"], "Brasília": ["Brasília", "Brasilia"],
                 "Mexico City": ["Mexico City"]}


def _forms(key):
    return COUNTRY_FORMS.get(key, [key])


def city_country(name, user, bridge, country, n_swaps=6, group="", predict=""):
    """A fact whose bridge is a city and whose answer is its country (groups B, C)."""
    country = DATASET_NAME.get(country, country)
    others = [c for c in CITIES.get(country, []) if c != bridge]
    swaps = {c: k for c, k in SWAP_CITIES.items() if k != country}
    swaps = dict(list(swaps.items())[:n_swaps])
    swaps.update({c: country for c in others[1:3]})                 # same-answer swaps (not the capital)
    answers = {k: _forms(k) for k in set(swaps.values()) | {country}}
    return name, dict(user=user, bridge=[bridge], answer=country, answers=answers,
                      exclude=[_forms(country)[0].split()[-1] if country != "USA" else "America"],
                      same_kind=others[:3], swaps=swaps, q_start=user.split()[0] + " " + user.split()[1],
                      group=group, predict=predict)


def landmark(name, landmark, country, group="", predict=""):
    """Group A: landmark -> country -> capital."""
    user = f"What is the capital of the country where {landmark} is located? Answer in one word."
    cap = CAPITAL[country]
    swaps = {c: CAPITAL[c] for c in ["Germany", "Italy", "Japan", "Spain", "Egypt", "Poland", "Mexico"] if c != country}
    swaps = dict(list(swaps.items())[:6])
    answers = {CAPITAL[c]: CAPITAL_FORMS.get(CAPITAL[c], [CAPITAL[c]]) for c in list(swaps) + [country]}
    same = [c for c in ["Germany", "Italy", "Spain", "Japan"] if c != country][:3]
    return name, dict(user=user, bridge=[country], answer=cap, answers=answers,
                      exclude=[cap.split()[-1]], same_kind=same, swaps=swaps, q_start="What is",
                      group=group, predict=predict)


def uni_city(name, person, uni_word, city, same_city_unis, other_unis, group="", predict=""):
    """Group D: person -> undergrad university -> its city. `uni_word` = the university's distinctive word."""
    user = f"In which city is the university that {person} attended as an undergraduate? Answer in one word."
    swaps = {**other_unis, **{u: city for u in same_city_unis}}
    answers = {c: [c] for c in set(swaps.values()) | {city}}
    return name, dict(user=user, bridge=[uni_word], answer=city, answers=answers, exclude=[city],
                      same_kind=list(other_unis)[:3], swaps=swaps, q_start="In which", group=group, predict=predict)


BIRTH = "In which country is the city where {} was born? Answer in one word."
UNIV = "In which country is the city where the headquarters of {} is located? Answer in one word."
OTHER_UNIS = {"Stanford": "Stanford", "Yale": "New Haven", "Princeton": "Princeton", "Oxford": "Oxford",
              "Cornell": "Ithaca", "Berkeley": "Berkeley"}

PICKS = dict([
    # A1: landmark in the capital -> expect BYSTANDER (the landmark gives the capital directly)
    landmark("A1_colosseum", "the Colosseum", "Italy", "A1", "bystander"),
    landmark("A1_kremlin", "the Kremlin", "Russia", "A1", "bystander"),
    # A2: landmark in a famous non-capital city -> expect CONSULTED (the city shortcut gives the wrong answer)
    landmark("A2_sydney_opera", "the Sydney Opera House", "Australia", "A2", "consulted"),
    landmark("A2_cn_tower", "the CN Tower", "Canada", "A2", "consulted"),
    landmark("A2_christ_redeemer", "the Christ the Redeemer statue", "Brazil", "A2", "consulted"),
    landmark("A2_taj_mahal", "the Taj Mahal", "India", "A2", "consulted"),
    # B1: born in the capital -> expect ECHO (capital ~ country)
    city_country("B1_mishima", BIRTH.format("Yukio Mishima"), "Tokyo", "Japan", group="B1", predict="echo"),
    city_country("B1_djokovic", BIRTH.format("Novak Djokovic"), "Belgrade", "Serbia", group="B1", predict="echo"),
    # B2: famous, nationality = birth country -> expect ECHO (direct nationality link)
    city_country("B2_mozart", BIRTH.format("Wolfgang Amadeus Mozart"), "Salzburg", "Austria", group="B2", predict="echo"),
    city_country("B2_federer", BIRTH.format("Roger Federer"), "Basel", "Switzerland", group="B2", predict="echo"),
    city_country("B2_ronaldo", BIRTH.format("Cristiano Ronaldo"), "Funchal", "Portugal", group="B2", predict="echo"),
    # B3: birth country differs from what the person is mostly tied to -> expect CONSULTED
    city_country("B3_keanu", BIRTH.format("Keanu Reeves"), "Beirut", "Lebanon", group="B3", predict="consulted"),
    city_country("B3_stalin", BIRTH.format("Joseph Stalin"), "Gori", "Georgia", group="B3", predict="consulted"),
    city_country("B3_bieber", BIRTH.format("Justin Bieber"), "London", "Canada", group="B3",
                 predict="consulted (London, Ontario: a naive 'London' reading gives UK)"),
    city_country("B3_che", BIRTH.format("Che Guevara"), "Rosario", "Argentina", group="B3", predict="consulted"),
    # C1: university in a capital / mega-city -> expect ECHO or BYSTANDER
    city_country("C1_tsinghua", UNIV.format("Tsinghua University"), "Beijing", "China", group="C1", predict="echo"),
    city_country("C1_fordham", UNIV.format("Fordham University"), "York", "USA", group="C1", predict="bystander"),
    # C2: non-capital city -> expect BYSTANDER (the university gives the country directly)
    city_country("C2_mcgill", UNIV.format("McGill University"), "Montreal", "Canada", group="C2", predict="bystander"),
    city_country("C2_harvard", UNIV.format("Harvard University"), "Cambridge", "USA", group="C2",
                 predict="bystander (a naive 'Cambridge' reading gives UK)"),
    # D1: person -> university -> city, the name doesn't give the city -> expect CONSULTED
    uni_city("D1_larry_page", "Larry Page", "Michigan", "Ann Arbor", ["Michigan State"][:0], OTHER_UNIS, "D1", "consulted"),
    uni_city("D1_portman", "Natalie Portman", "Harvard", "Cambridge", ["MIT"], OTHER_UNIS, "D1", "consulted"),
])
