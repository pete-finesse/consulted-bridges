"""Pre-run fixes for GPU run 3 (Qwen3.6-27B confirmation), on the tiny model where a model is needed:
the "free" wrong-answer rule (and the old "kind" rule still reachable), per-format wrong peers, the v2 number
suffix, the reach edit on shared positions only, P(kind) in the 2x2 and the swap driver's cells."""
import math

import pytest

from consulted_bridges import conformity as C
from consulted_bridges import conformity_swap as SW
from consulted_bridges.load import load
from consulted_bridges.questions.proposed.candidates import ACTIVE
from consulted_bridges.text import chat, offsets, resolve_positions


@pytest.fixture(scope="module")
def h():
    return load("tiny")


def test_free_rule_letters_avoid_every_kind_initial():
    n = 0
    for q, s in ACTIVE.items():
        if C.answer_format(s) != "letter":
            continue
        w = C.wrong_letter(q, s, "free")
        assert w != s["answer"] and w not in C._kind_initials(s) and w in s["answers"], q
        assert C.wrong_letter2(q, s, "free") not in (w, s["answer"]), q
        n += 1
    assert n > 90


def test_free_rule_numbers_and_categories():
    for q, s in ACTIVE.items():
        f = C.answer_format(s)
        if f == "letter":
            continue
        w = C.wrong_letter(q, s, "free")
        assert w in s["answers"] and w != s["answer"], q
        assert not (len(w) == 1 and w.isalpha()), (q, w)          # never a letter outside the letter format
        if f == "number":
            assert w.isdigit() and 1 <= int(w) <= 9, q
            assert w not in {C.NUM_FACTS.get(x) for x in s["same_kind"]}, q
    s = ACTIVE["C_EL_helium"]          # 2: one off is 1 = hydrogen, a same-kind word -> 3
    assert C.wrong_letter("C_EL_helium", s, "free") == "3"
    assert C.wrong_letter("C_PL_saturn", ACTIVE["C_PL_saturn"], "free") == "7"     # Jupiter 5, Mars 4 are same-kind


def test_kind_rule_reproduces_the_development_runs():
    # the old rule: the first same-kind / swap word with another letter (Spain's S); pilots keep their letters
    s = C.SPECS["V_LM_fuji"]
    assert C.wrong_letter("V_LM_fuji", s, "kind") == s["same_kind"][0][0].upper()
    assert C.wrong_letter("L_colosseum", C.SPECS["L_colosseum"], "kind") == "F"
    assert C.wrong_letter("L_colosseum", C.SPECS["L_colosseum"], "free") == "F"
    old = C.WRONG_RULE
    try:
        C.WRONG_RULE = "kind"
        assert [p for c, k, p, _ in C.prompts("V_LM_fuji") if c == "disagree"] == [s["same_kind"][0][0].upper()] * 3
        C.WRONG_RULE = "free"
        free = C.wrong_letter("V_LM_fuji", s)
        assert [p for c, k, p, _ in C.prompts("V_LM_fuji") if c == "disagree"] == [free] * 3
        assert free != s["same_kind"][0][0].upper()
    finally:
        C.WRONG_RULE = old


def test_number_prompts_keep_the_question_and_suffix():
    for cond, k, peer, s in C.prompts("C_EL_helium", extra=("hidden", "told_copy")):
        u = s["user"]
        assert u.count(C.NUM1) == 1 and u.endswith(C.NUM1), cond
        assert "party balloons that makes a voice squeaky?" in u
        if cond == "disagree":
            assert peer == "3" and f"Agent 2: 3\n" in u


def test_swap_target():
    assert C.swap_target(C.SPECS["L_colosseum"]) == ("Spain", "S")
    assert C.swap_target(ACTIVE["C_EL_helium"]) == ("neon", None) or C.swap_target(ACTIVE["C_EL_helium"])[1] == "1"
    assert C.swap_target(ACTIVE["C_PL_saturn"]) == ("Jupiter", "5")


def test_reach_positions_are_shared_with_the_original_prompt(h):
    for cond, k, peer, s in C.prompts("L_colosseum", extra=("hidden",)):
        tn = chat(h, s["user"] + s["name_ask"], s["name_prefill"])
        pm = C.Prompt(h, "L_colosseum", cond, k, peer, s)
        shared = C.reach_positions(h, tn, s["user"], pm.c.ALL)
        qend = resolve_positions(h, tn, pm.c.ALL)
        offs = offsets(h, tn)
        covered = tn[offs[shared[0]][0]: offs[shared[-1]][1]]
        assert shared == qend[: len(shared)] and len(shared) < len(qend), cond
        assert covered.strip().startswith("What letter") and covered.endswith("Answer with one letter."), cond
        assert "First name" not in covered
        # the original prompt has the same tokens at the same positions
        ids_o, ids_n = h.encode(pm.c.text), h.encode(tn)
        assert [ids_o[p] for p in shared] == [ids_n[p] for p in shared], cond


def test_probs_x_matches_probs(h):
    s = C.SPECS["L_colosseum"]
    for cond, k, peer, sp in C.prompts("L_colosseum"):
        if cond not in ("alone", "disagree"):
            continue
        pm = C.Prompt(h, "L_colosseum", cond, k, peer, sp)
        pr, pw, _ = pm.probs()
        pr2, pw2, px = pm.probs_x([], ["S", "T"])
        assert math.isclose(pr, pr2, rel_tol=1e-4, abs_tol=1e-9) and math.isclose(pw, pw2, rel_tol=1e-4, abs_tol=1e-9)
        assert set(px) == {"S", "T"} and all(0 <= v <= 1 for v in px.values())


def test_swap_driver_cells(h, monkeypatch):
    # the tiny tokenizer splits every everyday word; the real one keeps "table" whole
    monkeypatch.setattr(SW, "everyday_word", lambda card, avoid: next(w for w in SW.EVERYDAY if w[0].upper() not in avoid))
    O = {("L_colosseum", "alone", 0): 3.0, ("L_colosseum", "agree", 3): 3.0}   # disagree 3 falls back to alone's
    rows = []
    SW.run_question(h, "L_colosseum", "neutral", O, rows)
    cells = {(r["condition"], r["cell"]) for r in rows}
    assert {("alone", c) for c in ("A", "R", "M", "E")} <= cells and ("alone", "C") not in cells
    for cond in ("agree", "disagree"):
        assert {(cond, c) for c in ("A", "R", "M", "E", "C", "CR", "CM", "CE")} <= cells
    r = rows[0]
    assert r["target"] == "Spain" and r["target_answer"] == "S" and r["layers"] == "0-2"
    assert next(x for x in rows if x["condition"] == "disagree")["onset_source"] == "alone"
    ev = {x["everyday"] for x in rows}
    assert len(ev) == 1 and next(iter(ev))[0].upper() not in ("I", "S", "F")
    # the swap edits the question sentence only: same tokens in alone and group prompts
    for x in rows:
        assert x["n_positions"] == rows[0]["n_positions"]
    assert all(x["follows"] is None for x in rows if x["cell"] not in ("M", "CM"))
