"""TwoHopFact facts with a city bridge and a country answer, and their automatic controls.

For a fact (Mozart -> Salzburg -> Austria) the readout words ("roles") are:
  bridge        Salzburg   (e2, never written in the question)
  same_country  another bridge city of the same country in the dataset, not the capital (Graz)
  capital       the country's capital (Vienna); missing if the bridge IS the capital (Tokyo)
  random        a bridge city of another country, not that country's capital (the paper's kind of control)
  answer        the country itself (Austria)
  peer_answer   the wrong country the scripted peers push (France), from a partner fact
Controls are drawn with a seed per fact, and must not share a first piece with the bridge.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from silent_dissent.entity_facts import CAPITALS, Fact, norm, probe_tokens

DATA = Path(__file__).resolve().parents[3] / "data" / "twohopfact" / "twohopfact_slim.csv"
CITY_COUNTRY = ("person-birthcity-cntry", "univ-hqcity-cntry")   # the paper's two TwoHopFact categories
ROLES = ("bridge", "same_country", "capital", "random", "answer", "peer_answer")

# capitals for countries TwoHopFact names differently from, or beyond, the authors' list
EXTRA_CAPITALS = {
    "United States of America": "Washington", "United Kingdom": "London", "People's Republic of China": "Beijing",
    "Ukraine": "Kyiv", "Estonia": "Tallinn", "Taiwan": "Taipei", "Venezuela": "Caracas", "Philippines": "Manila",
    "Mexico": "Mexico City", "Serbia": "Belgrade", "Slovakia": "Bratislava", "Bulgaria": "Sofia",
    "Belarus": "Minsk", "Israel": "Jerusalem", "Azerbaijan": "Baku", "Latvia": "Riga", "Croatia": "Zagreb",
    "New Zealand": "Wellington", "Pakistan": "Islamabad", "Armenia": "Yerevan", "Iraq": "Baghdad",
    "South Africa": "Pretoria", "Georgia": "Tbilisi", "Uruguay": "Montevideo", "Slovenia": "Ljubljana",
    "Kazakhstan": "Astana", "Saudi Arabia": "Riyadh", "Republic of Ireland": "Dublin", "Lebanon": "Beirut",
    "Ghana": "Accra", "Afghanistan": "Kabul", "Cuba": "Havana", "Cyprus": "Nicosia", "Kyrgyzstan": "Bishkek",
    "Yemen": "Sanaa", "Uzbekistan": "Tashkent", "Costa Rica": "San José", "North Macedonia": "Skopje",
    "Lithuania": "Vilnius", "Sudan": "Khartoum", "Moldova": "Chișinău", "Bangladesh": "Dhaka",
    "Algeria": "Algiers", "Bolivia": "Sucre", "Albania": "Tirana", "Ethiopia": "Addis Ababa",
    "Myanmar": "Naypyidaw", "Morocco": "Rabat", "Sri Lanka": "Colombo", "Bosnia and Herzegovina": "Sarajevo",
}


def capital_of(country: str) -> str | None:
    return CAPITALS.get(country) or EXTRA_CAPITALS.get(country)


def load(path: str | Path = DATA, categories=CITY_COUNTRY) -> list[Fact]:
    """The slim CSV as the authors' Fact objects (aliases: the value only; the slim file has none)."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if categories and r["category"] not in categories:
                continue
            out.append(Fact(uid=f"thf:{r['uid']}", source="twohopfact", category=r["category"], e1=r["e1.value"],
                            e2=r["e2.value"], e3=r["e3.value"], r1_prompt=r["r1(e1).prompt"],
                            r2_template=r["r2.template"], composed=r["r2(r1(e1)).prompt"],
                            e2_aliases=[r["e2.value"]], e3_aliases=[r["e3.value"]],
                            e2_type=r["e2.rough_category"], e3_type=r["e3.rough_category"]))
    return out


def find(facts: list[Fact], e1: str) -> Fact:
    hits = [f for f in facts if f.e1 == e1] or [f for f in facts if e1.lower() in f.e1.lower()]
    if not hits:
        raise KeyError(e1)
    return hits[0]


def cities_by_country(facts: list[Fact]) -> dict[str, list[str]]:
    """Country -> its bridge cities in the dataset, most frequent bridge first (a rough proxy
    for how well known a city is)."""
    by: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for f in facts:
        if f.e2_type == "city" and f.e3_type == "country":
            by[f.e3][f.e2] += 1
    return {k: sorted(v, key=lambda c: (-v[c], c)) for k, v in by.items()}


def pick_controls(tok, fact: Fact, by_country: dict[str, list[str]], pool: list[Fact], seed: int = 0,
                  prefer: str = "random") -> dict:
    """Role -> word for one fact (None where no valid control exists). A control must not share
    a first piece with any word already chosen for the fact (a shared piece can't tell them apart).
    prefer="frequent": the same-country city is the country's most frequent other bridge in the
    dataset (better known, like Graz) instead of a random one (often obscure, like Eisenstadt)."""
    rng = random.Random(f"{seed}-controls-{fact.uid}")
    pieces = lambda w: set(probe_tokens(tok, w)) if tok is not None else {norm(w)}
    cap = capital_of(fact.e3)
    peers = [p for p in pool if p.category == fact.category and norm(p.e3) != norm(fact.e3)]
    rng.shuffle(peers)
    peer = next((p for p in peers if not pieces(p.e3) & (pieces(fact.e3) | pieces(fact.e2))), None)
    taken = pieces(fact.e2) | pieces(fact.e3) | (pieces(peer.e3) if peer else set())

    def take(cands):
        for c in cands:
            if c and not pieces(c) & taken:
                taken.update(pieces(c))
                return c
        return None

    same = [c for c in by_country.get(fact.e3, []) if norm(c) not in (norm(fact.e2), norm(cap or ""))]
    others = [c for k, cs in by_country.items() if k != fact.e3 for c in cs if norm(c) != norm(capital_of(k) or "")]
    if prefer != "frequent":
        rng.shuffle(same)
    rng.shuffle(others)
    return {
        "bridge": fact.e2,
        "same_country": take(same),
        "capital": take([cap]) if cap and norm(cap) != norm(fact.e2) else None,   # None when the bridge is the capital
        "random": take(others),
        "answer": fact.e3,
        "peer_answer": peer.e3 if peer else None,
        "peer_fact": peer,
    }


def select(facts: list[Fact], per_bridge: int = 3, limit: int | None = None, seed: int = 0,
           uids: list[str] | None = None, known: set[str] | None = None) -> list[Fact]:
    """At most `per_bridge` facts per bridge city (so a few bridges can't dominate), optionally only
    `known` uids (from `sdh known`), then the first `limit` in a seeded shuffle."""
    if uids:
        want = set(uids)
        return [f for f in facts if f.uid in want]
    pool = [f for f in facts if known is None or f.uid in known]
    rng = random.Random(f"{seed}-select")
    rng.shuffle(pool)
    count: dict[str, int] = defaultdict(int)
    out = []
    for f in pool:
        if count[norm(f.e2)] < per_bridge:
            count[norm(f.e2)] += 1
            out.append(f)
    return out[:limit] if limit else out
