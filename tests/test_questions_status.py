"""Every question has exactly one status, ids never collide across modules, and every in-use id exists."""
from consulted_bridges.questions.status import MODULES, in_use_by_model, specs_by_module, status


def test_ids_unique_across_modules():
    seen = {}
    for name, S in specs_by_module().items():
        for k in S:
            assert k not in seen, f"{k} is defined in both {seen[k]} and {name}"
            seen[k] = name


def test_every_question_has_one_status():
    st = status()
    all_ids = {k for S in specs_by_module().values() for k in S}
    assert set(st) == all_ids
    assert set(st.values()) <= {"in_use", "screened_out", "candidate", "retired"}


def test_in_use_ids_exist_and_are_not_candidates_or_retired():
    st = status()
    for cfg, qs in in_use_by_model().items():
        for q in qs:
            assert q in st, f"{cfg} uses {q}, which no module defines"
            assert st[q] == "in_use", (cfg, q, st[q])


def test_candidates_are_apart_from_the_screen_list():
    from consulted_bridges.conformity import CANDIDATE, LETTER
    assert not set(CANDIDATE) & set(LETTER)
    assert all(k.startswith("C_") for k in CANDIDATE)
    assert len(MODULES) == 10
