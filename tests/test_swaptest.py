"""The swap test's bookkeeping, with scripted replies (the tiny model knows nothing)."""
from consulted_bridges.questions.core import SPECS
from consulted_bridges.scorecard import Card
from consulted_bridges.swaptest import headline, rule_positions, swap_test
from consulted_bridges.text import resolve_positions


def scripted(card, follows):
    """Replies: a one-hop check about a word -> its first letter; a move -> the target's letter if `follows`,
    else the right answer; a removal -> "Q" (the fallback); anything else -> the right answer."""
    words = card.S["swaps"] + ["table", "river", "garden", "lamp", "bottle", "window"]

    def gen(text, edits=None):
        if edits and edits[0].kind == "move":
            return edits[0].words[1][0].upper() if follows else card.S["answer"]
        if edits and edits[0].kind == "ablate":
            return "Q"
        for w in words:
            if f"word {w} " in text:
                return w[0].upper()
        return card.S["answer"]
    card.gen = gen
    card.one_token = lambda w: True        # the tiny tokenizer splits most words
    return card


def test_rule_positions(h):
    c = Card(h, "L_colosseum", SPECS["L_colosseum"], verbose=False)
    P = rule_positions(c)
    assert P["B"] == resolve_positions(h, c.text, c.ALL)
    assert P["A"] == P["B"][:len(P["A"])] and len(P["A"]) < len(P["B"])
    assert "<|im_end|>" not in h.tok.decode([h.encode(c.text)[p] for p in P["A"]])


def test_swap_test_counts(h):
    onset = 12
    for follows in (True, False):
        c = scripted(Card(h, "L_colosseum", SPECS["L_colosseum"], verbose=False), follows)
        rows, summ = swap_test(c, onset)
        whole = [r for r in summ if r["whole"]]
        assert {r["rule"] for r in whole} == {"A", "B"}
        for r in summ:
            assert r["targets"] == "Germany, Japan, Spain, France"     # first 4 with distinct letters, not I
            assert r["everyday"] == "table, river, garden"
            assert r["n_informative"] == 4 and r["n_everyday_readable"] == 3
            assert r["follow_share"] == (1.0 if follows else 0.0)
            assert r["counts"] is (not follows)                          # everyday words get through iff moves follow
        assert all(r["n_layers"] == onset for r in whole)
        assert all(max(map(int, r["layers"].split("-"))) < onset for r in summ if not r["whole"])
        hl = headline(summ)
        assert hl["swapA_follow"] == hl["swapB_follow"] == (1.0 if follows else 0.0)
        assert {r["role"] for r in rows} == {"removal", "target", "everyday", "source_everyday", "source_kind"}
