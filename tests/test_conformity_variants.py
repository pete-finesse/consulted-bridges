"""Phase B1/B4 on the tiny model: the new prompt variants, the whole-answer peer regex, and the original conditions
reproduced exactly."""
import math

import pytest

from consulted_bridges import conformity as C
from consulted_bridges.load import load
from consulted_bridges.text import chat, offsets

ORIGINAL = [("alone", 0, "F"), ("agree", 1, "I"), ("agree", 2, "I"), ("agree", 3, "I"),
            ("disagree", 1, "F"), ("disagree", 2, "F"), ("disagree", 3, "F"), ("copy", 3, "F")]


@pytest.fixture(scope="module")
def h():
    return load("tiny")


def test_original_conditions_unchanged():
    assert [(c, k, p) for c, k, p, _ in C.prompts("L_colosseum")] == ORIGINAL
    assert [(c, k, p) for c, k, p, _ in C.prompts("L_colosseum", extra=())] == ORIGINAL


def test_variants_present_with_right_peer_counts():
    out = [(c, k, p) for c, k, p, _ in C.prompts("L_colosseum", extra=tuple(C.VARIANTS))]
    assert out[: len(ORIGINAL)] == ORIGINAL
    ks = {}
    for c, k, p in out[len(ORIGINAL):]:
        ks.setdefault(c, []).append(k)
    assert ks == {"hidden": [1, 2, 3], "hidden_sub": [1, 2, 3], "disagree_g": [1, 2, 3], "told_copy": [3],
                  "told_ignore": [3], "peer_bridge": [1, 3], "peer_bridge_wrong": [1, 3]}
    peers = {c: p for c, k, p in out}
    assert peers["hidden"] == "?" and peers["hidden_sub"] == "(answer submitted)"
    assert peers["peer_bridge"] == "The country is Italy." and peers["peer_bridge_wrong"] == "The country is Spain."
    assert peers["disagree_g"] == "S" and peers["disagree_g"] not in ("I", "F")
    users = {c: s["user"] for c, k, p, s in C.prompts("L_colosseum", extra=("told_copy", "told_ignore"))}
    assert "Repeat Agent 2's answer." in users["told_copy"] and "Ignore the other agents" in users["told_ignore"]
    with pytest.raises(ValueError):
        C.prompts("L_colosseum", extra=("nope",))


def test_wrong_letter2_differs_for_every_letter_question():
    for qid in C.LETTER:
        s = C.SPECS[qid]
        g = C.wrong_letter2(qid, s)
        assert g != s["answer"] and g != C.wrong_letter(qid, s), qid


def test_hidden_prompt_lines_up_with_letter_prompt(h):
    # " ?" tokenizes like " I": a space token then the character (two tokens on Qwen), so the "?" prompt has the
    # same length as the letter prompts and the peers' tokens sit at the same positions (patching needs this)
    assert len(h.encode(" ?")) == len(h.encode(" I")) == len(h.encode(" F"))
    P = {(c, k): (p, s["user"]) for c, k, p, s in C.prompts("L_colosseum", extra=("hidden",))}
    for k in (1, 2, 3):
        texts = {c: chat(h, P[(c, k)][1]) for c in ("hidden", "agree", "disagree")}
        n = {c: len(h.encode(x)) for c, x in texts.items()}
        assert n["hidden"] == n["agree"] == n["disagree"]
        pos = {c: C.peer_positions(h, texts[c], P[(c, k)][0], k) for c in texts}
        assert pos["hidden"] == pos["agree"] == pos["disagree"] and len(pos["hidden"]) == k


def test_peer_positions_cover_the_whole_answer(h):
    for cond, k, peer, s in C.prompts("L_colosseum", extra=("hidden_sub", "peer_bridge")):
        if cond not in ("hidden_sub", "peer_bridge", "agree"):
            continue
        text = chat(h, s["user"])
        pos = C.peer_positions(h, text, peer, k)
        offs = offsets(h, text)
        covered = "".join(text[a:b] for a, b in (offs[i] for i in pos))
        assert covered.count(peer) == k, (cond, covered)
        assert len(pos) >= k


def test_prompt_objects_for_new_conditions(h):
    for cond, k, peer, s in C.prompts("L_colosseum", extra=tuple(C.VARIANTS)):
        pm = C.Prompt(h, "L_colosseum", cond, k, peer, s)
        assert pm.dis == (cond in C.DISAGREE_LIKE)
        if cond in C.HIDDEN:
            assert pm.hidden and pm.pforms == []
            pr, pw, lo = pm.probs()
            assert math.isnan(pw) and math.isnan(lo) and 0 <= pr <= 1
        elif cond == "peer_bridge":
            assert pm.pletter == "I" and pm.pforms == ["I"]
        elif cond == "peer_bridge_wrong":
            assert pm.pletter == "S" and pm.pforms == ["S"]
        elif cond != "alone":
            assert pm.pletter == peer
        if k:
            skipped = set(C.peer_positions(h, pm.c.text, peer, k))
            assert skipped and not skipped & set(pm.onpos) and skipped <= set(pm.allpos)
