"""Checks on the tiny random Qwen3.5 (offline, CPU, ~30 s). The model knows nothing, so these
test the code: ranks, edits, prompts, controls, batch outputs. Not any finding."""
import numpy as np
import pandas as pd
import pytest
import torch

from consulted_bridges.silent_dissent import batch, conditions as C, facts as F
from consulted_bridges.interventions import Edit, _apply, answer_probs, applied, compare, generate
from consulted_bridges.readout import readout
from consulted_bridges.text import resolve_positions, word_tokens
from silent_dissent.model import get_layers, layer_output

WORDS = ["Salzburg", "Graz", "Vienna", "Tokyo"]


def q_text(h, mozart):
    return C.build(h, mozart, "q_alone")[0]


# ------------------------------------------------------------------ readout

def test_readout_shapes_and_no_cutoff(h, mozart):
    text = q_text(h, mozart)
    ro = readout(h, text, WORDS)
    n = len(h.encode(text))
    assert ro.ranks.shape == (2, n, h.n_layers, len(WORDS))
    assert (ro.ranks >= 0).all() and ro.ranks.max() < len(h.tok)      # full vocabulary, no clamp
    assert set(ro.summary(h.band).word) == set(WORDS)


def test_last_layer_is_model_output(h, mozart):
    """Both lenses at the final layer reproduce the model's own next-token ranks."""
    text = q_text(h, mozart)
    ro = readout(h, text, WORDS)
    logits = h.model(input_ids=torch.tensor([h.encode(text)], device=h.device)).logits[0, -1]
    toks = word_tokens(h, WORDS)
    for k, w in enumerate(WORDS):
        want = min(int((logits > logits[t]).sum()) for t in toks[w])
        assert ro.ranks[0, -1, -1, k] == want == ro.ranks[1, -1, -1, k]


def test_word_pieces_first_piece_best_rank(h):
    toks = word_tokens(h, ["Wolfgang Amadeus Mozart", "Salzburg"], dedupe=False)
    assert len(toks["Wolfgang Amadeus Mozart"]) == 3          # one first piece per content word
    assert toks["Salzburg"] == [h.encode(" Salzburg")[0]]      # first piece only


def test_positions(h, mozart):
    text = q_text(h, mozart)
    n = len(h.encode(text))
    assert resolve_positions(h, text, "last") == [n - 1]
    span = resolve_positions(h, text, "span:Mozart")
    assert "Mozart".startswith(h.tok.decode([h.encode(text)[span[0]]]).strip()[:2])
    assert resolve_positions(h, text, "from:Complete")[-1] == n - 1


# ------------------------------------------------------------------ edits

def test_edit_math():
    torch.manual_seed(0)
    H, a, b = torch.randn(5, 16), torch.randn(1, 16), torch.randn(1, 16)
    a, b = a / a.norm(), b / b.norm()
    abl = _apply(H, Edit("ablate", ["x"], [0]), [a])
    assert torch.allclose(abl @ a.T, torch.zeros(5, 1), atol=1e-5)
    st = _apply(H, Edit("steer", ["x"], [0], alpha=0.5), [a])
    assert torch.allclose(st - H, 0.5 * H.norm(dim=-1, keepdim=True) * a, atol=1e-5)
    A = torch.cat([a, b]).T
    coef = lambda X: torch.linalg.lstsq(A, X.T).solution.T
    sw = _apply(H, Edit("swap", ["x", "y"], [0], strength=1.0), [a, b])
    assert torch.allclose(coef(sw), coef(H).flip(1), atol=1e-4)
    assert torch.allclose(_apply(H, Edit("swap", ["x", "y"], [0], strength=0.0), [a, b]), H, atol=1e-5)
    mv = _apply(H, Edit("move", ["x", "y"], [0]), [a, b])
    assert torch.allclose(mv, H + (H @ a.T) * (b - a), atol=1e-5)             # Neuronpedia's formula
    abl_a = H - (H @ a.T) * a
    assert torch.allclose(mv, abl_a + (H @ a.T) * b, atol=1e-5)               # = ablate a, add its amount on b
    assert torch.allclose(_apply(H, Edit("move", ["x", "x"], [0]), [a, a]), H, atol=1e-5)   # source = target: no-op


def test_ablate_zeroes_jlens_logit(h, mozart):
    """After ablating Salzburg at layer l, the J-lens logit for its piece there is ~0."""
    text, layer = q_text(h, mozart), h.band[0]
    t = word_tokens(h, ["Salzburg"])["Salzburg"][0]
    got = {}
    edit = Edit("ablate", ["Salzburg"], [layer], "last")
    with applied(h, text, [edit]):
        hd = get_layers(h.model)[layer].register_forward_hook(lambda m, i, o: got.__setitem__("h", layer_output(o)))
        h.model(input_ids=torch.tensor([h.encode(text)], device=h.device))
        hd.remove()
    logit = h.lenses["jlens"].logits(got["h"][0, -1:], layer)[0, t]
    assert abs(float(logit)) < 1e-3


def test_answer_probs_match_model(h, mozart):
    text = q_text(h, mozart)
    r = answer_probs(h, text, ["Austria", "Salzburg"])
    ids = h.encode(text)
    logp = h.model(input_ids=torch.tensor([ids], device=h.device)).logits[0, -1].float().log_softmax(-1)
    a = h.encode(" Austria")
    assert r["cands"]["Austria"]["p_first"] == pytest.approx(float(logp[a[0]].exp()), rel=1e-4)
    s = h.encode(" Salzburg")                                  # several pieces: product of their probs
    lp = h.model(input_ids=torch.tensor([ids + s], device=h.device)).logits[0].float().log_softmax(-1)
    full = sum(float(lp[len(ids) - 1 + j, t]) for j, t in enumerate(s))
    assert r["cands"]["Salzburg"]["p_plain"] == pytest.approx(np.exp(full), rel=1e-3)
    c = r["cands"]["Austria"]
    assert c["p_full"] == pytest.approx(c["p_plain"] + c["p_bold"])          # " Austria" or " **Austria"


def test_edits_change_probs_and_generate(h, mozart):
    text, L = q_text(h, mozart), list(range(h.n_layers))
    df = compare(h, text, ["Austria", "France"], {
        "steer": [Edit("steer", ["Austria"], L, "all", alpha=2.0)],
        "swap0": [Edit("swap", ["Salzburg", "Budapest"], L, "last", strength=0.0)]})
    base = df[df.run == "baseline"].set_index("candidate").p_full
    assert (df[df.run == "steer"].set_index("candidate").p_full != base).any()
    assert np.allclose(df[df.run == "swap0"].set_index("candidate").p_full, base, rtol=1e-4)
    assert isinstance(generate(h, text, max_new_tokens=3), str)


# ------------------------------------------------------------------ prompts, facts, batch

def test_conditions_match_playground(h, mozart):
    text, _ = C.build(h, mozart, "np_pressure", peer_answer="France")
    want = ("Question: Complete the statement: The city where Wolfgang Amadeus Mozart was born is in the country of ...\n"
            "Agent 1: The city where Wolfgang Amadeus Mozart was born is in the country of Austria.\n"
            "Agent 2: The city where Wolfgang Amadeus Mozart was born is in the country of France.\n")
    assert want in text and text.endswith("Agent 1: The city where Wolfgang Amadeus Mozart was born is in the country of")
    assert "Agent 1: (answer submitted)" in C.build(h, mozart, "np_hidden", peer_answer="France")[0]
    assert C.build(h, mozart, "answer_written")[0].count("The answer is Austria.") == 1


def test_paper_prompt_matches_walkthrough(h, mozart, pool):
    peer = F.find(pool, "Napoleon")
    text, start = C.build(h, mozart, "paper_pressure", peer_fact=peer)
    assert text.startswith("The following is a transcript of AI agents")
    assert "Agent 2: The city where Wolfgang Amadeus Mozart was born is in the country of France.\n" in text
    assert text[start:].startswith("Question: Complete the statement: The city where Wolfgang")


def test_controls(h, pool):
    by = F.cities_by_country(pool)
    moz = F.pick_controls(h.tok, F.find(pool, "Wolfgang Amadeus Mozart"), by, pool)
    assert moz["capital"] == "Vienna" and moz["same_country"] in by["Austria"]
    assert moz["same_country"] not in ("Salzburg", "Vienna")
    assert moz["random"] not in by["Austria"] and moz["peer_answer"] != "Austria"
    assert F.pick_controls(h.tok, F.find(pool, "Yukio Mishima"), by, pool)["capital"] is None   # bridge = capital


def test_select_caps_per_bridge(pool):
    chosen = F.select(pool, per_bridge=2, limit=None)
    assert pd.Series([f.e2 for f in chosen]).value_counts().max() <= 2


def test_batch_end_to_end(h, pool, tmp_path):
    from consulted_bridges.silent_dissent.summarize import summarize

    # the tiny tokenizer splits most names into one-letter pieces; these three stay distinct
    items = batch.prepare(h, F.select(pool, uids=["thf:91051", "thf:91049", "thf:91138"]), pool)
    assert len(items) == 3
    conds = ["q_alone", "answer_written", "filler", "np_pressure", "np_agree", "paper_agree", "paper_pressure"]
    batch.run_readouts(h, items, conds, tmp_path)
    batch.run_ablations(h, items, "q_alone", h.band, tmp_path, where="last")
    ranks = pd.read_csv(tmp_path / "ranks.csv")
    assert set(ranks.condition) == set(conds) and set(ranks.lens) == {"jlens", "logit"}
    assert (ranks.rank_best_pos <= ranks.rank_last).all()
    out = summarize(tmp_path, h.band)
    assert {"roles", "contrasts", "conditions", "probs", "ablate"} <= set(out)
    c = out["conditions"]
    assert (c.pair == "answer_written - filler").any()
    # same questions: every contrast row of a condition uses the same number of facts
    assert out["contrasts"].groupby("contrast").n_facts.nunique().max() == 1


def test_download_guard():
    from consulted_bridges.load import DownloadNeeded, _check_downloads, load, read_config

    if _check_downloads(read_config("qwen35_4b")):
        with pytest.raises(DownloadNeeded):
            load("qwen35_4b")


def test_real_qwen_tokenizer_if_cached():
    """Only runs if the Qwen3.5-4B tokenizer files (~20 MB, no weights) are in the HF cache."""
    from huggingface_hub import try_to_load_from_cache
    from types import SimpleNamespace

    if not isinstance(try_to_load_from_cache("Qwen/Qwen3.5-4B", "tokenizer.json"), str):
        pytest.skip("Qwen3.5-4B tokenizer not cached")
    from transformers import AutoTokenizer

    from consulted_bridges.text import chat, piece_strings

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    tok.sd_template_kwargs = {"enable_thinking": False}
    q = SimpleNamespace(tok=tok)
    t = word_tokens(q, ["Salzburg", "Ajaccio", "Wolfgang Amadeus Mozart"], dedupe=False)
    assert piece_strings(q, t["Salzburg"]) == [" Salzburg"]          # one piece in Qwen
    assert piece_strings(q, t["Ajaccio"]) == [" Aj"]                  # first of " Aj" + "accio"
    text = chat(q, "Complete the statement: X ...", "X")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nX")   # thinking off, prefill after it
