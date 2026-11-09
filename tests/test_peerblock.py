"""Exact tests of the peer block and the patch on the tiny hybrid Qwen3.5 (two linear-attention and two
full-attention layers, random weights, CPU). The numbers mean nothing; the invariances must be exact."""
import pytest
import torch

from consulted_bridges.conformity import group_user
from consulted_bridges.interventions import Edit, answer_probs, applied, applied_rows
from consulted_bridges.load import load
from consulted_bridges.peerblock import Block, Patch, peer_token_positions, two_by_two
from consulted_bridges.text import chat

Q = "What letter does the name of the country where the Colosseum is located start with?"
SFX = " Answer with one letter."


@pytest.fixture(scope="module")
def h():
    return load("tiny")


def prompt(h, peer, k=3):
    return chat(h, group_user(Q, peer, k, "neutral", SFX))


@torch.no_grad()
def last_logits(h, text, edits=None):
    ids = torch.tensor([h.encode(text)], device=h.device)
    with applied(h, text, edits or []):
        return h.model(input_ids=ids).logits[0, -1].float()


def layer_kinds(h):
    from silent_dissent.model import get_layers
    return ["attn" if hasattr(l, "self_attn") else "linear" for l in get_layers(h.model)]


def test_tiny_is_hybrid(h):
    kinds = layer_kinds(h)
    assert "attn" in kinds and "linear" in kinds, kinds


def test_peer_positions_cover_each_answer(h):
    t = prompt(h, "I")
    pos = peer_token_positions(h, t)
    assert len(pos) == 3
    ids = h.encode(t)
    assert all(h.tok.decode([ids[p]]).strip() == "I" for p in pos)
    t2 = prompt(h, "(answer submitted)")
    assert len(peer_token_positions(h, t2)) >= 3          # the whole-line regex covers multi-token answers


def test_leak_block_everywhere_is_exact(h):
    ti, tf = prompt(h, "I"), prompt(h, "F")
    assert len(h.encode(ti)) == len(h.encode(tf))
    pos = peer_token_positions(h, ti)
    assert pos == peer_token_positions(h, tf)
    li, lf = last_logits(h, ti), last_logits(h, tf)
    assert not torch.allclose(li, lf, atol=1e-4), "peers must matter without the block"
    blk = [Block(pos, list(range(h.n_layers)))]
    bi, bf = last_logits(h, ti, blk), last_logits(h, tf, blk)
    assert torch.allclose(bi, bf, atol=1e-4), (bi - bf).abs().max()
    assert not torch.allclose(bi, li, atol=1e-4), "the block must change the output"


def test_block_on_full_attention_only_still_leaks(h):
    ti, tf = prompt(h, "I"), prompt(h, "F")
    pos = peer_token_positions(h, ti)
    attn_layers = [i for i, k in enumerate(layer_kinds(h)) if k == "attn"]
    blk = [Block(pos, attn_layers)]
    assert not torch.allclose(last_logits(h, ti, blk), last_logits(h, tf, blk), atol=1e-4)
    lin_layers = [i for i, k in enumerate(layer_kinds(h)) if k == "linear"]
    blk = [Block(pos, lin_layers)]
    assert not torch.allclose(last_logits(h, ti, blk), last_logits(h, tf, blk), atol=1e-4)


def test_from_query_blocks_only_late_queries(h):
    ti, tf = prompt(h, "I"), prompt(h, "F")
    pos = peer_token_positions(h, ti)
    n = len(h.encode(ti))
    # block everything for the linear layers, but only the last query position for attention: peers still reach the
    # last position through earlier positions' attention, so the outputs differ
    attn_layers = [i for i, k in enumerate(layer_kinds(h)) if k == "attn"]
    lin_layers = [i for i, k in enumerate(layer_kinds(h)) if k == "linear"]
    blk = [Block(pos, lin_layers), Block(pos, attn_layers, from_query=n - 1)]
    assert not torch.allclose(last_logits(h, ti, blk), last_logits(h, tf, blk), atol=1e-4)
    # from_query=0 on the same layers is the full block
    blk = [Block(pos, lin_layers), Block(pos, attn_layers, from_query=0)]
    assert torch.allclose(last_logits(h, ti, blk), last_logits(h, tf, blk), atol=1e-4)


def test_identity_patch_is_a_no_op(h):
    t = prompt(h, "I")
    pos = peer_token_positions(h, t)
    base = last_logits(h, t)
    pt = [Patch(t, pos, list(range(h.n_layers)))]
    assert torch.allclose(last_logits(h, t, pt), base, atol=1e-5)


def test_patch_from_hidden_prompt(h):
    ti, tq = prompt(h, "I"), prompt(h, "?")
    assert len(h.encode(ti)) == len(h.encode(tq)), "'?' must be one token like 'I' so positions line up"
    pos = peer_token_positions(h, ti)
    assert pos == peer_token_positions(h, tq)
    base = last_logits(h, ti)
    pt = [Patch(tq, pos, list(range(h.n_layers)))]
    patched_out = last_logits(h, ti, pt)
    assert not torch.allclose(patched_out, base, atol=1e-4)
    # patching every layer from "?" and blocking are different operations: neither equals the "?" prompt itself in
    # general, but the patch at the LAST layer only must reproduce the "?" prompt's state at the peer positions, which
    # the final position cannot read after the last layer, so the output equals the unpatched one
    pt_last = [Patch(tq, pos, [h.n_layers - 1])]
    assert torch.allclose(last_logits(h, ti, pt_last), base, atol=1e-5)


def test_patch_length_mismatch_raises(h):
    ti, ts = prompt(h, "I"), prompt(h, "(answer submitted)")
    pos = peer_token_positions(h, ti)
    with pytest.raises(ValueError):
        with applied(h, ti, [Patch(ts, pos, [0])]):
            pass


def test_mixed_edit_list_and_rows_refusal(h):
    ti = prompt(h, "I")
    pos = peer_token_positions(h, ti)
    edits = [Edit("ablate", ["Italy"], [0, 1], "all"), Block(pos, list(range(h.n_layers)))]
    r = answer_probs(h, ti, ["I", "F"], edits)
    assert set(r["cands"]) == {"I", "F"}
    with pytest.raises(NotImplementedError):
        with applied_rows(h, ti, [(slice(0, 1), [Block(pos, [0])])]):
            pass


def test_two_by_two_runs(h):
    ti = prompt(h, "I")
    pos = peer_token_positions(h, ti)
    r = two_by_two(h, ti, ["Italy"], [0, 1], pos, ["I", "F"])
    assert set("ABCD") <= set(r)
    assert all(0.0 <= r[c] <= 1.0 for c in "ABCD")
    # C and D must not depend on the peers' letter
    tf = prompt(h, "F")
    rf = two_by_two(h, tf, ["Italy"], [0, 1], pos, ["I", "F"])
    assert abs(r["C"] - rf["C"]) < 1e-5 and abs(r["D"] - rf["D"]) < 1e-5
