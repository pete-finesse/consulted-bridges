"""CANDIDATE questions for the conformity v2 runs (ids prefixed C_), drafted 8 Nov 2026 before any screen.

Why they exist (docs/conformity/plan.md §3, docs/conformity/scope.md): the current sets are one format (first
letter) in three geography relations, 19–20 questions per model. That is enough to show the three-peer effect
exists and not enough to read it by format, relation, wording or model. These candidates widen all of that. None
has been run. Every one goes through the four-check gate (knows both hops, single tokens, need >= 0.5 with the
same-kind control below the bridge, reach) on each model; the funnel is reported.

Drafting rules applied here, fixed before any result:
  - the subject never carries the answer or the bridge (no "Trinity College Dublin", no "Peking University",
    no "octagon" whose name holds 8, no "Mercury is closest to the sun");
  - no bridge with an alias that changes the answer (United Kingdom / Britain / England; Netherlands / Holland;
    Czech Republic / Czechia; Myanmar / Burma; Bangalore / Bengaluru);
  - number answers are 1–9 (Qwen splits digits, so 10+ shares a first token with a small number);
  - same-kind controls are words of the bridge's kind with a DIFFERENT answer;
  - a bridge used by an existing question is not reused with the same second hop; where two candidates share
    a bridge they share a split (SPLIT below keeps a bridge's questions together);
  - wrong peer answers (chosen at run time) must not be the answer of a plausible alternative bridge.

Formats and relations:
  letter    C_LM landmark -> country, C_LP person -> birth city, C_LU university -> city       (first letter)
  number    C_MN event -> month -> its number, C_EL description -> element -> atomic number,
            C_PL description -> planet -> position from the sun, C_IN musician -> instrument -> strings,
            C_SP athlete -> sport -> players per side
  category  C_CT landmark -> country -> continent, C_DS landmark -> country -> driving side   (overlap suspects)
  class     C_VC landmark -> country -> first letter -> starts with a vowel, yes/no (three hops, binary; Gemini's
            vowel/consonant idea, as yes/no because " Vowel" and " Consonant" are two tokens on Qwen)
  spelling  C_EO landmark -> country -> letter count -> Even/Odd (expected to fail knows-it, like last letter;
            four questions so the screen can say so)

Dropped at drafting: last letter (untestable in the ideas run), shapes -> sides (the name carries the number),
capital / language / demonym (overlap with the bridge), sports and counts with two-digit answers.
Dropped at the token check (Qwen3.5-4B tokenizer, 8 Nov): the whole currency family (every currency name is two
tokens), and 28 questions whose bridge is two tokens (Bhutan, Montenegro, Ulm, Belgrade, Pretoria, Tupelo, Funchal,
Leiden, Delft, Genoa, Brookline, Saginaw, Zadar, Lausanne, Krakow, Kampala, Evanston, Waco, Provo, Aachen, Rehovot,
Haifa, Dunedin, Pohang, fluorine, Uranus, cello, banjo). C_SP shares its second hop with V_T (same sports); its first
hop is an athlete instead of a description, so it tests whether V_T's failures on the 27B were the first hop.
Run `python -m consulted_bridges.questions.proposed.candidates` for counts, the dev/confirmation split and the digest.
"""
from __future__ import annotations

import hashlib
import json
import random

from consulted_bridges.questions.core import CITY_SWAPS, ONE, base, letter
from consulted_bridges.questions.proposed.trend import NUMT, _cap, _dirs

NUM1 = " Answer with a single number."
COUNTRY_SWAPS = ["Germany", "Japan", "Spain", "Brazil", "Canada", "Egypt", "Mexico", "Russia", "France", "Italy"]


# ---------------------------------------------------------------- letter (first letter of the bridge) ------------
def lm(name, place, country, same_kind, predict="consulted", verb="is located"):
    swaps = [c for c in COUNTRY_SWAPS if c[0] != country[0] and c != country][:6]
    return letter(name, predict, f"the country where {place} {verb}", [country], "country", same_kind, swaps,
                  f"In which country is {place} located?" if verb == "is located" else f"In which country does {place} stand?")


def lp(name, person, city, same_kind, predict="consulted"):
    swaps = [c for c in CITY_SWAPS if c[0] != city[0] and c != city][:7]
    return letter(name, predict, f"the city where {person} was born", [city], "city", same_kind, swaps,
                  f"In which city was {person} born?")


def lu(name, uni, city, same_kind, predict="consulted"):
    swaps = [c for c in CITY_SWAPS if c[0] != city[0] and c != city][:7]
    return letter(name, predict, f"the city where {uni} has its main campus", [city], "city", same_kind, swaps,
                  f"In which city does {uni} have its main campus?")


# ---------------------------------------------------------------- number formats --------------------------------
def num(name, user, bridge, kind, ans, hop1, hop2, check, same_kind, swaps, group, q_start, predict="consulted"):
    return name, base(user + NUM1, [bridge], kind, ans, NUMT, _dirs(ans), hop1 + ONE, _cap([bridge]), hop2 + NUM1,
                      check + NUM1, same_kind, [s for s in swaps if s != bridge][:8], group, predict, q_start)


MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September"]
MONTH_N = {m: str(i + 1) for i, m in enumerate(MONTHS)}


def mn(name, event, month, same_kind, predict="consulted"):
    return num(name, f"Counting January as 1, what is the number of the month in which {event}?", month, "month",
               MONTH_N[month], f"In which month {event}?", f"Counting January as 1, what is the number of the month {month}?",
               "Counting January as 1, what is the number of the month {}?", same_kind, MONTHS, "C_MN", "Counting January",
               predict)


ELEMENTS = ["hydrogen", "helium", "lithium", "carbon", "nitrogen", "oxygen", "fluorine", "boron", "neon"]


def el(name, description, element, ans, same_kind, predict="consulted"):
    return num(name, f"What is the atomic number of {description}?", element, "element", ans,
               f"What is {description}?", f"What is the atomic number of {element}?", "What is the atomic number of {}?",
               same_kind, ELEMENTS, "C_EL", "What is", predict)


PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
PLANET_N = {p: str(i + 1) for i, p in enumerate(PLANETS)}


def pl(name, description, planet, same_kind, predict="consulted"):
    return num(name, f"Counting outward from the sun, what position does {description} have among the planets?", planet,
               "planet", PLANET_N[planet], f"Which planet is {description}?",
               f"Counting outward from the sun, what position does {planet} have among the planets?",
               "Counting outward from the sun, what position does {} have among the planets?", same_kind, PLANETS,
               "C_PL", "Counting outward", predict)


INSTRUMENTS = ["guitar", "violin", "cello", "banjo", "ukulele", "mandolin"]


def inst(name, musician, instrument, ans, same_kind, predict="consulted"):
    return num(name, f"How many strings does the instrument that {musician} is famous for playing usually have?",
               instrument, "instrument", ans, f"Which instrument is {musician} famous for playing?",
               f"How many strings does a {instrument} usually have?", "How many strings does a {} usually have?",
               same_kind, INSTRUMENTS, "C_IN", "How many", predict)


SPORTS = ["baseball", "basketball", "hockey", "volleyball", "handball", "polo"]


def sp(name, athlete, sport, ans, same_kind, predict="consulted"):
    return num(name, f"How many players does each team have on the field at once in the sport that {athlete} played?",
               sport, "sport", ans, f"Which sport did {athlete} play?",
               f"How many players does each team have on the field at once in {sport}?",
               "How many players does each team have on the field at once in {}?", same_kind, SPORTS, "C_SP",
               "How many", predict)


# ---------------------------------------------------------------- categorical lookups on a country -----------------
def cat(name, place, country, ans, answers, question, hop2, check, same_kind, group, q_start, predict, kind="country"):
    swaps = [c for c in COUNTRY_SWAPS if c != country][:6]
    return name, base(question + ONE, [country], kind, ans, answers, [ans], f"In which country is {place} located?" + ONE,
                      [country], hop2 + ONE, check + ONE, same_kind, swaps, group, predict, q_start)


CONTINENTS = {c: [c] for c in ["Asia", "Africa", "Europe"]}
SIDES = {"Left": ["Left"], "Right": ["Right"]}


def ct(name, place, country, continent, same_kind, predict="consulted (overlap suspect)"):
    return cat(name, place, country, continent, CONTINENTS,
               f"On which continent is the country where {place} is located?",
               f"On which continent is {country}?", "On which continent is {}?", same_kind, "C_CT", "On which", predict)


def ds(name, place, country, side, same_kind, predict="consulted (overlap suspect)"):
    return cat(name, place, country, side, SIDES,
               f"On which side of the road do cars drive in the country where {place} is located?",
               f"On which side of the road do cars drive in {country}?", "On which side of the road do cars drive in {}?",
               same_kind, "C_DS", "On which", predict)



YESNO = {"Yes": ["Yes", "yes"], "No": ["No", "no"]}
PARITY = {"Even": ["Even", "even"], "Odd": ["Odd", "odd"]}


def vc(name, place, country, same_kind, predict="consulted (three hops, binary)"):
    """Gemini's vowel/consonant class, as yes/no because ' Vowel' and ' Consonant' are two tokens on Qwen."""
    ans = "Yes" if country[0] in "AEIOU" else "No"
    return cat(name, place, country, ans, YESNO,
               f"Does the name of the country where {place} is located start with a vowel?",
               f"Does the word {country} start with a vowel?", "Does the word {} start with a vowel?", same_kind, "C_VC",
               "Does the", predict)


def eo(name, place, country, same_kind, predict="likely fails knows-it: letter counting, like last letter"):
    ans = "Even" if len(country) % 2 == 0 else "Odd"
    return cat(name, place, country, ans, PARITY,
               f"Is the number of letters in the name of the country where {place} is located even or odd?",
               f"Is the number of letters in the word {country} even or odd?",
               "Is the number of letters in the word {} even or odd?", same_kind, "C_EO", "Is the", predict)


SPECS = dict([
    # ---- C_LM: landmark -> country -> first letter (new countries; none already a bridge in V_LM / L) ----------
    lm("C_LM_petra", "Petra", "Jordan", ["Syria", "Lebanon"]),
    lm("C_LM_chichen", "Chichen Itza", "Mexico", ["Guatemala", "Peru"]),
    lm("C_LM_neuschwanstein", "Neuschwanstein Castle", "Germany", ["Austria", "France"]),
    lm("C_LM_alhambra", "the Alhambra", "Spain", ["Portugal", "France"]),
    lm("C_LM_hagia", "Hagia Sophia", "Turkey", ["Greece", "Syria"]),
    lm("C_LM_watarun", "Wat Arun", "Thailand", ["Vietnam", "Cambodia"]),
    lm("C_LM_borobudur", "Borobudur", "Indonesia", ["Malaysia", "Thailand"]),
    lm("C_LM_kilimanjaro", "Mount Kilimanjaro", "Tanzania", ["Kenya", "Uganda"], verb="stands"),
    lm("C_LM_moai", "the Moai statues of Easter Island", "Chile", ["Peru", "Argentina"]),
    lm("C_LM_blarney", "Blarney Castle", "Ireland", ["Scotland", "Wales"]),
    lm("C_LM_mermaid", "the Little Mermaid statue", "Denmark", ["Sweden", "Norway"]),
    lm("C_LM_halong", "Ha Long Bay", "Vietnam", ["Thailand", "Cambodia"]),
    lm("C_LM_belem", "the Belem Tower", "Portugal", ["Spain", "France"]),
    lm("C_LM_hassan", "the Hassan II Mosque", "Morocco", ["Algeria", "Tunisia"]),
    lm("C_LM_tikal", "the ruins of Tikal", "Guatemala", ["Mexico", "Honduras"]),
    lm("C_LM_wawel", "Wawel Castle", "Poland", ["Hungary", "Germany"]),
    lm("C_LM_atomium", "the Atomium", "Belgium", ["France", "Germany"]),
    lm("C_LM_schonbrunn", "Schonbrunn Palace", "Austria", ["Germany", "Hungary"]),
    lm("C_LM_vasa", "the Vasa Museum", "Sweden", ["Norway", "Denmark"]),
    lm("C_LM_lalibela", "the rock-hewn churches of Lalibela", "Ethiopia", ["Kenya", "Sudan"]),
    lm("C_LM_pashupatinath", "Pashupatinath Temple", "Nepal", ["India", "China"]),
    lm("C_LM_bluelagoon", "the Blue Lagoon geothermal spa", "Iceland", ["Norway", "Denmark"]),
    lm("C_LM_bran", "Bran Castle", "Romania", ["Hungary", "Bulgaria"]),
    lm("C_LM_casarosada", "the Casa Rosada", "Argentina", ["Chile", "Uruguay"]),
    lm("C_LM_bastion", "Fisherman's Bastion", "Hungary", ["Austria", "Romania"]),
    lm("C_LM_diocletian", "Diocletian's Palace", "Croatia", ["Slovenia", "Serbia"]),
    lm("C_LM_geiranger", "the Geirangerfjord", "Norway", ["Sweden", "Denmark"]),
    lm("C_LM_suomenlinna", "the Suomenlinna sea fortress", "Finland", ["Sweden", "Estonia"]),
    lm("C_LM_marinabay", "Marina Bay Sands", "Singapore", ["Malaysia", "Indonesia"]),
    lm("C_LM_salar", "the Salar de Uyuni", "Bolivia", ["Peru", "Chile"]),
    lm("C_LM_galapagos", "the Galapagos Islands", "Ecuador", ["Peru", "Colombia"]),
    lm("C_LM_angelfalls", "Angel Falls", "Venezuela", ["Colombia", "Brazil"]),
    lm("C_LM_timbuktu", "Timbuktu", "Mali", ["Niger", "Ghana"]),
    lm("C_LM_bled", "Lake Bled", "Slovenia", ["Croatia", "Austria"]),
    lm("C_LM_rila", "Rila Monastery", "Bulgaria", ["Romania", "Greece"]),
    lm("C_LM_chocolatehills", "the Chocolate Hills", "Philippines", ["Indonesia", "Malaysia"]),
    lm("C_LM_persepolis", "the ruins of Persepolis", "Iran", ["Iraq", "Turkey"]),
    lm("C_LM_babylon", "the ruins of Babylon", "Iraq", ["Iran", "Syria"]),
    lm("C_LM_baalbek", "the Roman temples of Baalbek", "Lebanon", ["Syria", "Jordan"]),
    lm("C_LM_masada", "the fortress of Masada", "Israel", ["Jordan", "Egypt"]),
    lm("C_LM_potala", "the Potala Palace", "China", ["Nepal", "India"]),
    # ---- C_LP: person -> birth city -> first letter (cities not already a bridge) -----------------------------
    lp("C_LP_curie", "Marie Curie", "Warsaw", ["Berlin", "Vienna"]),
    lp("C_LP_beethoven", "Ludwig van Beethoven", "Bonn", ["Cologne", "Vienna"]),
    lp("C_LP_messi", "Lionel Messi", "Rosario", ["Mendoza", "Santiago"]),
    lp("C_LP_kobe", "Kobe Bryant", "Philadelphia", ["Boston", "Chicago"]),
    lp("C_LP_lebron", "LeBron James", "Akron", ["Cleveland", "Columbus"]),
    lp("C_LP_jordan", "Michael Jordan", "Brooklyn", ["Chicago", "Charlotte"]),
    lp("C_LP_hanks", "Tom Hanks", "Concord", ["Oakland", "Sacramento"]),
    lp("C_LP_buffett", "Warren Buffett", "Omaha", ["Lincoln", "Denver"]),
    lp("C_LP_merkel", "Angela Merkel", "Hamburg", ["Berlin", "Munich"]),
    lp("C_LP_portman", "Natalie Portman", "Jerusalem", ["Cairo", "Beirut"]),
    lp("C_LP_zidane", "Zinedine Zidane", "Marseille", ["Lyon", "Nice"]),
    lp("C_LP_kafka", "Franz Kafka", "Prague", ["Vienna", "Budapest"]),
    lp("C_LP_wilde", "Oscar Wilde", "Dublin", ["Cork", "Belfast"]),
    lp("C_LP_trudeau", "Justin Trudeau", "Ottawa", ["Montreal", "Toronto"]),
    lp("C_LP_reynolds", "Ryan Reynolds", "Vancouver", ["Calgary", "Toronto"]),
    lp("C_LP_kanye", "Kanye West", "Atlanta", ["Chicago", "Memphis"]),
    lp("C_LP_shaq", "Shaquille O'Neal", "Newark", ["Camden", "Boston"]),
    lp("C_LP_magic", "Magic Johnson", "Lansing", ["Detroit", "Flint"]),
    lp("C_LP_tiger", "Tiger Woods", "Cypress", ["Anaheim", "Irvine"]),
    lp("C_LP_adele", "the singer Adele", "London", ["Manchester", "Birmingham"]),
    lp("C_LP_lennon", "John Lennon", "Liverpool", ["Manchester", "Leeds"]),
    lp("C_LP_galileo", "Galileo Galilei", "Pisa", ["Florence", "Rome"]),
    lp("C_LP_dante", "Dante Alighieri", "Florence", ["Rome", "Venice"]),
    lp("C_LP_marcopolo", "Marco Polo", "Venice", ["Rome", "Milan"]),
    lp("C_LP_hawking", "Stephen Hawking", "Oxford", ["Cambridge", "London"]),
    lp("C_LP_obama", "Barack Obama", "Honolulu", ["Chicago", "Seattle"]),
    lp("C_LP_sampras", "Pete Sampras", "Washington", ["Baltimore", "Chicago"]),
    lp("C_LP_graf", "Steffi Graf", "Mannheim", ["Munich", "Berlin"]),
    lp("C_LP_benzema", "Karim Benzema", "Lyon", ["Marseille", "Paris"]),
    lp("C_LP_haaland", "Erling Haaland", "Leeds", ["Manchester", "Oslo"]),
    # ---- C_LU: university -> city of its main campus -> first letter (name never carries the city) ------------
    lu("C_LU_humboldt", "Humboldt University", "Berlin", ["Munich", "Hamburg"]),
    lu("C_LU_tsinghua", "Tsinghua University", "Beijing", ["Shanghai", "Tokyo"]),
    lu("C_LU_harvard", "Harvard University", "Cambridge", ["Boston", "Providence"]),
    lu("C_LU_mcgill", "McGill University", "Montreal", ["Toronto", "Quebec"]),
    lu("C_LU_sapienza", "Sapienza University", "Rome", ["Milan", "Naples"]),
    lu("C_LU_complutense", "the Complutense University", "Madrid", ["Barcelona", "Valencia"]),
    lu("C_LU_eth", "ETH, the Swiss Federal Institute of Technology", "Zurich", ["Geneva", "Bern"]),
    lu("C_LU_erasmus", "Erasmus University", "Rotterdam", ["Amsterdam", "Utrecht"]),
    lu("C_LU_karolinska", "the Karolinska Institute", "Stockholm", ["Oslo", "Helsinki"]),
    lu("C_LU_lomonosov", "Lomonosov State University", "Moscow", ["Kiev", "Berlin"]),
    lu("C_LU_yonsei", "Yonsei University", "Seoul", ["Tokyo", "Beijing"]),
    lu("C_LU_ntu", "National Taiwan University", "Taipei", ["Tokyo", "Seoul"]),
    lu("C_LU_wits", "the University of the Witwatersrand", "Johannesburg", ["Nairobi", "Lagos"]),
    lu("C_LU_emory", "Emory University", "Atlanta", ["Nashville", "Charlotte"]),
    lu("C_LU_vanderbilt", "Vanderbilt University", "Nashville", ["Memphis", "Atlanta"]),
    lu("C_LU_duke", "Duke University", "Durham", ["Raleigh", "Charlotte"]),
    lu("C_LU_georgetown", "Georgetown University", "Washington", ["Baltimore", "Richmond"]),
    lu("C_LU_cmu", "Carnegie Mellon University", "Pittsburgh", ["Philadelphia", "Cleveland"]),
    lu("C_LU_case", "Case Western Reserve University", "Cleveland", ["Columbus", "Cincinnati"]),
    lu("C_LU_gonzaga", "Gonzaga University", "Spokane", ["Seattle", "Tacoma"]),
    lu("C_LU_marquette", "Marquette University", "Milwaukee", ["Madison", "Chicago"]),
    lu("C_LU_strathclyde", "the University of Strathclyde", "Glasgow", ["Edinburgh", "Aberdeen"]),
    lu("C_LU_heriotwatt", "Heriot-Watt University", "Edinburgh", ["Glasgow", "London"]),
    lu("C_LU_goethe", "Goethe University", "Frankfurt", ["Munich", "Berlin"]),
    lu("C_LU_bohr", "the Niels Bohr Institute", "Copenhagen", ["Stockholm", "Oslo"]),
    lu("C_LU_unsw", "the University of New South Wales", "Sydney", ["Melbourne", "Canberra"]),
    lu("C_LU_puc", "the Pontifical Catholic University of Chile", "Santiago", ["Lima", "Caracas"]),
    lu("C_LU_ainshams", "Ain Shams University", "Cairo", ["Alexandria", "Tunis"]),
    # ---- C_MN: event -> month -> number in the year (1-9) --------------------------------------------------------
    mn("C_MN_valentine", "Valentine's Day falls", "February", ["March", "June"]),
    mn("C_MN_patrick", "Saint Patrick's Day falls", "March", ["February", "May"]),
    mn("C_MN_earthday", "Earth Day falls", "April", ["March", "June"]),
    mn("C_MN_titanic", "the Titanic sank", "April", ["March", "June"]),
    mn("C_MN_derby", "the Kentucky Derby is run", "May", ["April", "June"]),
    mn("C_MN_memorial", "the United States observes Memorial Day", "May", ["June", "July"]),
    mn("C_MN_dday", "the D-Day landings in Normandy took place", "June", ["May", "July"]),
    mn("C_MN_fathers", "Father's Day is observed in the United States", "June", ["May", "July"]),
    mn("C_MN_bastille", "Bastille Day falls", "July", ["June", "August"]),
    mn("C_MN_tour", "the Tour de France is held", "July", ["June", "August"]),
    mn("C_MN_woodstock", "the Woodstock festival of 1969 took place", "August", ["July", "September"]),
    mn("C_MN_india", "India celebrates its Independence Day", "August", ["July", "September"]),
    mn("C_MN_fringe", "the Edinburgh Festival Fringe is held", "August", ["July", "September"]),
    mn("C_MN_labor", "the United States observes Labor Day", "September", ["May", "August"]),
    # ---- C_EL: description -> element -> atomic number (1-9) -----------------------------------------------------
    el("C_EL_nitrogen", "the gas that makes up most of Earth's atmosphere", "nitrogen", "7", ["oxygen", "carbon"]),
    el("C_EL_carbon", "the element that diamonds are made of", "carbon", "6", ["nitrogen", "oxygen"]),
    el("C_EL_oxygen", "the gas that a fire needs in order to burn", "oxygen", "8", ["nitrogen", "carbon"]),
    el("C_EL_helium", "the gas in party balloons that makes a voice squeaky", "helium", "2", ["neon", "hydrogen"]),
    el("C_EL_lithium", "the metal used in rechargeable phone batteries", "lithium", "3", ["carbon", "oxygen"]),
    el("C_EL_hydrogen", "the most abundant element in the universe", "hydrogen", "1", ["helium", "oxygen"],
       predict="consulted; risk: 'most abundant' is close to 'first'"),
    # ---- C_PL: description -> planet -> position from the sun -----------------------------------------------------
    pl("C_PL_mars", "the red planet", "Mars", ["Venus", "Jupiter"]),
    pl("C_PL_jupiter", "the largest planet in the solar system", "Jupiter", ["Saturn", "Mars"]),
    pl("C_PL_saturn", "the planet with the most prominent rings", "Saturn", ["Jupiter", "Mars"]),
    pl("C_PL_venus", "the hottest planet in the solar system", "Venus", ["Mars", "Mercury"]),
    pl("C_PL_neptune", "the planet whose existence was predicted mathematically before it was seen", "Neptune",
       ["Saturn", "Mars"]),
    # ---- C_IN: musician -> instrument -> strings -------------------------------------------------------------------
    inst("C_IN_hendrix", "Jimi Hendrix", "guitar", "6", ["violin", "bass"]),
    inst("C_IN_perlman", "Itzhak Perlman", "violin", "4", ["guitar", "piano"]),
    # ---- C_SP: athlete -> sport -> players per side (1-9) -----------------------------------------------------------
    sp("C_SP_ruth", "Babe Ruth", "baseball", "9", ["basketball", "hockey"]),
    sp("C_SP_jeter", "Derek Jeter", "baseball", "9", ["basketball", "hockey"]),
    sp("C_SP_lebron", "LeBron James", "basketball", "5", ["baseball", "hockey"]),
    sp("C_SP_bird", "Larry Bird", "basketball", "5", ["baseball", "hockey"]),
    sp("C_SP_gretzky", "Wayne Gretzky", "hockey", "6", ["baseball", "basketball"]),
    # ---- C_CT: landmark -> country -> continent (overlap suspect) ----------------------------------------------------
    ct("C_CT_petra", "Petra", "Jordan", "Asia", ["Egypt", "Greece"]),
    ct("C_CT_kilimanjaro", "Mount Kilimanjaro", "Tanzania", "Africa", ["India", "Spain"]),
    ct("C_CT_halong", "Ha Long Bay", "Vietnam", "Asia", ["Kenya", "France"]),
    ct("C_CT_borobudur", "Borobudur", "Indonesia", "Asia", ["Egypt", "Italy"]),
    ct("C_CT_hassan", "the Hassan II Mosque", "Morocco", "Africa", ["Spain", "Turkey"]),
    ct("C_CT_neuschwanstein", "Neuschwanstein Castle", "Germany", "Europe", ["Japan", "Kenya"]),
    ct("C_CT_wawel", "Wawel Castle", "Poland", "Europe", ["China", "Egypt"]),
    ct("C_CT_timbuktu", "Timbuktu", "Mali", "Africa", ["Spain", "India"]),
    ct("C_CT_persepolis", "the ruins of Persepolis", "Iran", "Asia", ["Egypt", "Italy"]),
    ct("C_CT_lalibela", "the rock-hewn churches of Lalibela", "Ethiopia", "Africa", ["India", "Greece"]),
    ct("C_CT_atomium", "the Atomium", "Belgium", "Europe", ["Japan", "Morocco"]),
    ct("C_CT_watarun", "Wat Arun", "Thailand", "Asia", ["Kenya", "Spain"]),
    # ---- C_DS: landmark -> country -> driving side (balanced; "Right" is the default answer) -----------------------
    ds("C_DS_watarun", "Wat Arun", "Thailand", "Left", ["Vietnam", "China"]),
    ds("C_DS_borobudur", "Borobudur", "Indonesia", "Left", ["Vietnam", "Philippines"]),
    ds("C_DS_blarney", "Blarney Castle", "Ireland", "Left", ["France", "Germany"]),
    ds("C_DS_maasai", "the Maasai Mara reserve", "Kenya", "Left", ["Ethiopia", "Egypt"]),
    ds("C_DS_dunns", "Dunn's River Falls", "Jamaica", "Left", ["Cuba", "Mexico"]),
    ds("C_DS_petra", "Petra", "Jordan", "Right", ["India", "Kenya"]),
    ds("C_DS_chichen", "Chichen Itza", "Mexico", "Right", ["Jamaica", "Australia"]),
    ds("C_DS_neuschwanstein", "Neuschwanstein Castle", "Germany", "Right", ["Ireland", "Malta"]),
    ds("C_DS_hagia", "Hagia Sophia", "Turkey", "Right", ["Cyprus", "India"]),
    ds("C_DS_moai", "the Moai statues of Easter Island", "Chile", "Right", ["Australia", "Japan"]),
    ds("C_DS_halong", "Ha Long Bay", "Vietnam", "Right", ["Thailand", "Indonesia"]),
    # ---- C_VC: landmark -> country -> first letter -> vowel or consonant (balanced 7/7) --------------------------------
    vc("C_VC_colosseum", "the Colosseum", "Italy", ["France", "Spain"]),
    vc("C_VC_pyramids", "the Pyramids of Giza", "Egypt", ["Sudan", "Libya"]),
    vc("C_VC_opera", "the Sydney Opera House", "Australia", ["Japan", "China"]),
    vc("C_VC_taj", "the Taj Mahal", "India", ["Pakistan", "Nepal"]),
    vc("C_VC_borobudur", "Borobudur", "Indonesia", ["Malaysia", "Thailand"]),
    vc("C_VC_blarney", "Blarney Castle", "Ireland", ["France", "Spain"]),
    vc("C_VC_schonbrunn", "Schonbrunn Palace", "Austria", ["Germany", "Hungary"]),
    vc("C_VC_acropolis", "the Acropolis", "Greece", ["Italy", "Egypt"]),
    vc("C_VC_eiffel", "the Eiffel Tower", "France", ["Italy", "Austria"]),
    vc("C_VC_angkor", "Angkor Wat", "Cambodia", ["India", "Indonesia"]),
    vc("C_VC_fuji", "Mount Fuji", "Japan", ["India", "Australia"]),
    vc("C_VC_kremlin", "the Kremlin", "Russia", ["Ukraine", "Estonia"]),
    vc("C_VC_machu", "Machu Picchu", "Peru", ["Ecuador", "Argentina"]),
    vc("C_VC_petra", "Petra", "Jordan", ["Israel", "Egypt"]),
    # ---- C_EO: even or odd letter count (expected to fail; four so the screen can show it) ----------------------------
    eo("C_EO_colosseum", "the Colosseum", "Italy", ["France", "Peru"]),
    eo("C_EO_eiffel", "the Eiffel Tower", "France", ["Italy", "Japan"]),
    eo("C_EO_fuji", "Mount Fuji", "Japan", ["France", "Peru"]),
    eo("C_EO_machu", "Machu Picchu", "Peru", ["Italy", "Japan"]),
])


def family(qid):
    return qid.rsplit("_", 1)[0]


def split(seed="conformity-v2-candidates-2026-11-08", confirm_share=1 / 3):
    """Development / confirmation split, fixed by seed, keeping every question with the same bridge in one half
    and stratified by family. Returns {"dev": [...], "confirm": [...]}."""
    rng = random.Random(seed)
    out = {"dev": [], "confirm": []}
    fams = sorted({family(k) for k in SPECS})
    for f in fams:
        groups = {}
        for k in sorted(SPECS):
            if family(k) == f:
                groups.setdefault(SPECS[k]["bridge"][0], []).append(k)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_conf = round(len(keys) * confirm_share)
        for i, b in enumerate(keys):
            out["confirm" if i < n_conf else "dev"] += groups[b]
    return {h: sorted(v) for h, v in out.items()}


def digest():
    """sha256 of the candidate table and split, to commit alongside so the drafting is provably prior."""
    blob = json.dumps({"specs": SPECS, "split": split()}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


if __name__ == "__main__":
    import collections
    s = split()
    print(len(SPECS), "candidates;", collections.Counter(family(k) for k in SPECS))
    print("dev", len(s["dev"]), "confirm", len(s["confirm"]))
    print("letters among first-letter questions:",
          collections.Counter(SPECS[k]["answer"] for k in SPECS if family(k) in ("C_LM", "C_LP", "C_LU")))
    print("sha256", digest())
