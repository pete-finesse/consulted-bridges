"""Sentences for the bridge scorecard (bridge_card.py). Add a new one by copying an entry.

Each entry:
  user        the question (chat, no prefill); the bridge is never written in it
  bridge      the hidden middle step (first entry is the main word; others are variants removed with it)
  answer      key of the correct answer in `answers`
  answers     answer key -> the forms the model may write (first token decides the probability)
  exclude     words whose J-lens directions stand for the answer itself ("Salzburg minus Austria")
  same_kind   other words of the bridge's kind (other Austrian cities, other animals); controls for distinctness
  swaps       alternative bridge -> the answer it should give. Entries whose answer equals the correct one
              are same-answer controls (octopus for spider: also eight legs)
  q_start     the first words of the question (edits apply from here on)
"""

SPECS = {
    "spider": dict(
        user="How many legs do the animals that spin webs have? Answer in one word.",
        bridge=["spider", "spiders"], answer="8",
        answers={"8": ["Eight", "8"], "6": ["Six", "6"], "4": ["Four", "4"], "2": ["Two", "2"],
                 "10": ["Ten", "10"], "0": ["Zero", "None", "0"]},
        exclude=["eight", "Eight"], same_kind=["insects", "ants", "crabs"],
        swaps={"bee": "6", "ant": "6", "dog": "4", "bird": "2", "crab": "10", "snake": "0",
               "octopus": "8", "scorpion": "8"},
        q_start="How many"),
    "mozart": dict(
        user="In which country is the city where Wolfgang Amadeus Mozart was born? Answer in one word.",
        bridge=["Salzburg"], answer="Austria",
        answers={"Austria": ["Austria"], "Germany": ["Germany"], "France": ["France"], "Hungary": ["Hungary"],
                 "Italy": ["Italy"], "Switzerland": ["Switzerland"], "Czech": ["Czech"], "Spain": ["Spain"],
                 "Japan": ["Japan"], "Poland": ["Poland"]},
        exclude=["Austria"], same_kind=["Vienna", "Graz", "Innsbruck"],
        swaps={"Munich": "Germany", "Paris": "France", "Budapest": "Hungary", "Milan": "Italy",
               "Zurich": "Switzerland", "Prague": "Czech", "Madrid": "Spain", "Osaka": "Japan",
               "Innsbruck": "Austria", "Graz": "Austria", "Linz": "Austria"},
        q_start="In which"),
    "mozart_direct": dict(   # shortcut check: the question needs no city
        user="Which country was Wolfgang Amadeus Mozart from? Answer in one word.",
        bridge=["Salzburg"], answer="Austria",
        answers={"Austria": ["Austria"], "Germany": ["Germany"], "France": ["France"], "Hungary": ["Hungary"],
                 "Italy": ["Italy"], "Switzerland": ["Switzerland"], "Spain": ["Spain"]},
        exclude=["Austria"], same_kind=["Vienna", "Graz", "Innsbruck"],
        swaps={"Munich": "Germany", "Budapest": "Hungary", "Milan": "Italy", "Madrid": "Spain",
               "Innsbruck": "Austria", "Graz": "Austria"},
        q_start="Which country"),
    "elephant": dict(
        user="How many legs does the animal with a trunk and tusks have? Answer in one word.",
        bridge=["elephant", "elephants"], answer="4",
        answers={"4": ["Four", "4"], "8": ["Eight", "8"], "6": ["Six", "6"], "2": ["Two", "2"],
                 "0": ["Zero", "None", "0"]},
        exclude=["four", "Four"], same_kind=["giraffe", "horse", "lion"],
        swaps={"spider": "8", "ant": "6", "bird": "2", "snake": "0", "horse": "4", "dog": "4"},
        q_start="How many"),
    "apple": dict(
        user="What color is the fruit that keeps the doctor away? Answer in one word.",
        bridge=["apple", "apples"], answer="red",
        answers={"red": ["Red"], "yellow": ["Yellow"], "green": ["Green"], "orange": ["Orange"],
                 "blue": ["Blue"], "purple": ["Purple"]},
        exclude=["red", "Red"], same_kind=["banana", "orange", "pear"],
        swaps={"banana": "yellow", "lime": "green", "blueberry": "blue", "grape": "purple",
               "strawberry": "red", "cherry": "red"},
        q_start="What color"),
    "eiffel": dict(   # a likely shortcut: the Eiffel Tower is itself tied to Paris
        user="What is the capital of the country where the Eiffel Tower is? Answer in one word.",
        bridge=["France"], answer="Paris",
        answers={"Paris": ["Paris"], "Berlin": ["Berlin"], "Rome": ["Rome"], "Tokyo": ["Tokyo"],
                 "Madrid": ["Madrid"], "Cairo": ["Cairo"], "London": ["London"]},
        exclude=["Paris"], same_kind=["Germany", "Italy", "Spain"],
        swaps={"Germany": "Berlin", "Italy": "Rome", "Japan": "Tokyo", "Spain": "Madrid", "Egypt": "Cairo"},
        q_start="What is"),
}
