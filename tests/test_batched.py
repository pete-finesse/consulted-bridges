"""answer_probs_many (several differently edited copies of a prompt per forward pass) gives the same
numbers as answer_probs once per set of edits."""
import pytest

from consulted_bridges.interventions import Edit, answer_probs, answer_probs_many
from consulted_bridges.load import load
from consulted_bridges.text import chat


@pytest.fixture(scope="module")
def h():
    return load("tiny")


def test_many_matches_one_at_a_time(h):
    text = chat(h, "What letter does the name of the country where the Colosseum is located start with?", "The answer is")
    sets = [None, [Edit("ablate", ["Italy"], [1, 2], "all")], [Edit("ablate", ["Italy", "Rome"], [0], "last")],
            [Edit("move", ["Italy", "Spain"], [1], "all")], [Edit("ablate_excl", ["Italy", "Spain"], [2], "all")],
            [Edit("ablate", ["Italy"], [1], "all"), Edit("steer", ["Spain"], [2], "last", alpha=0.5)],
            [Edit("amplify", ["Italy"], [0, 1, 2], "all", alpha=1.0)]]
    cands = ["I", "S", "Italy"]
    many = answer_probs_many(h, text, cands, sets, chunk=3)     # chunks of 3, 3, 1: crosses a chunk edge
    for es, got in zip(sets, many):
        want = answer_probs(h, text, cands, es)
        for c in cands:
            for k, v in want["cands"][c].items():
                assert got["cands"][c][k] == pytest.approx(v, rel=1e-4, abs=1e-7), (es, c, k)
        assert [t for t, _ in got["top"]] == [t for t, _ in want["top"]]
