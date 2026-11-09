"""Block the peers, or replace them: two interventions on the peer tokens of a group prompt.

Block   later positions can no longer read the peer tokens. Full-attention layers: the attention mask forbids every
        query at position >= from_query (other than the key itself) from attending to the peer positions.
        Linear-attention layers (Qwen3.5 / 3.6 GatedDeltaNet): the mixer's input is zeroed at the peer positions
        (so nothing of them enters in_proj or the short causal conv), and beta and the log-decay g are set to 0 there
        (so those tokens neither update nor decay the recurrent state). With a Block at every layer the output at
        later positions is exactly invariant to what the peers wrote (tests/test_peerblock.py, the leak test).
        from_query > 0 blocks only queries at or after that position (e.g. the answer positions), to test late direct
        copying; the linear path has no per-query notion, so it is blocked whole whenever the layer is listed.
Patch   the residual at the peer positions is replaced, at the listed layers, by the residual the same positions
        carry in another prompt of identical length (peers writing "?"): "replaced by a neutral answer" rather than
        "made invisible". The source prompt runs once, unedited, before anything else is registered.

Both are passed in the same `edits` list as interventions.Edit objects (interventions.applied); per-row batching
(applied_rows) does not support them. Order inside one forward: Edit hooks, then Block, then Patch (a patched position
ends up carrying exactly the source state).

Implementation notes, transformers 5.22.1:
  - The GatedDeltaNet forward calls the module globals `torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`
    by name at call time. Those names are bound to `use_kernel_func_from_hub_with_fallback` wrappers, which may dispatch
    to a hub/fla kernel on GPU. We rebind the *module globals* for the duration of the context, so our wrapper runs
    upstream of whichever implementation they dispatch to, and the zeroed beta / g reach the kernel too.
  - The decoder layer passes `attention_mask=` as a keyword to the attention module, and the mask for sdpa may be
    None (is_causal path), a bool tensor or a float additive tensor; all three are handled.
"""
from __future__ import annotations

import re
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

import torch

from silent_dissent.model import get_layers, layer_output, replace_layer_output

from .text import offsets

RULE_NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
              "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")


@dataclass
class Block:
    positions: list[int]           # absolute token positions of the peers' answers
    layers: list[int]
    from_query: int = 0            # full attention: only queries at positions >= this are blocked (0 = all)
    label: str = "block"

    def __post_init__(self):
        self.positions = sorted({int(p) for p in self.positions})
        if not self.positions:
            raise ValueError("Block needs at least one position")


@dataclass
class Patch:
    source_text: str               # the prompt whose residuals are written in (same token count as the target)
    positions: list[int]
    layers: list[int]
    label: str = "patch"

    def __post_init__(self):
        self.positions = sorted({int(p) for p in self.positions})


def peer_token_positions(h, text: str, peer_regex: str = r"Agent \d: (.+)\n") -> list[int]:
    """Token positions covering each peer's answer span ("Agent i: <answer>")."""
    offs = offsets(h, text)
    pos = []
    for m in re.finditer(peer_regex, text):
        a, b = m.span(1)
        pos += [i for i, (x, y) in enumerate(offs) if x < b and y > a]
    return sorted(set(pos))


# ------------------------------------------------------------------ the block ---------------------------------
# What the linear-attention wrapper reads: positions blocked for the GatedDeltaNet layer now running, else None.
_LIN_STATE: dict = {"positions": None}


def _attn_hook(positions, from_query):
    def pre(module, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        if hs is None:
            return None
        Bsz, S = hs.shape[0], hs.shape[1]
        dev = hs.device
        q = torch.arange(S, device=dev)[:, None]
        k = torch.arange(S, device=dev)[None, :]
        pos = torch.tensor([p for p in positions if p < S], device=dev)
        if not len(pos):
            return None
        is_peer = torch.zeros(S, dtype=torch.bool, device=dev)
        is_peer[pos] = True
        forbid = is_peer[None, :] & (q >= from_query) & (q != k)                      # [S, S]
        mask = kwargs.get("attention_mask", None)
        if mask is None:
            allowed = (k <= q) & ~forbid
            neg = torch.finfo(hs.dtype).min
            new = torch.zeros(S, S, dtype=hs.dtype, device=dev).masked_fill(~allowed, neg)
            new = new[None, None].expand(Bsz, 1, S, S).contiguous()
        elif mask.dtype == torch.bool:
            new = mask & ~forbid[None, None, -mask.shape[-2]:, -mask.shape[-1]:]
        else:
            neg = torch.finfo(mask.dtype).min
            add = torch.zeros(S, S, dtype=mask.dtype, device=mask.device).masked_fill(forbid, neg)
            new = mask + add[None, None, -mask.shape[-2]:, -mask.shape[-1]:]
            new = torch.clamp(new, min=neg)
        kwargs = dict(kwargs)
        kwargs["attention_mask"] = new
        return args, kwargs
    return pre


def _linear_pre(positions):
    def pre(module, args, kwargs):
        if "hidden_states" in kwargs:
            hs = kwargs["hidden_states"].clone()
            idx = [p for p in positions if p < hs.shape[1]]
            hs[:, idx] = 0
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = hs
        else:
            hs = args[0].clone()
            idx = [p for p in positions if p < hs.shape[1]]
            hs[:, idx] = 0
            args = (hs,) + tuple(args[1:])
        _LIN_STATE["positions"] = idx
        return args, kwargs
    return pre


def _linear_post(module, args, out):
    _LIN_STATE["positions"] = None
    return None


def _wrap_rule(fn):
    def inner(query, key, value, g=None, beta=None, **kw):
        P = _LIN_STATE["positions"]
        if P:
            if beta is not None:
                beta = beta.clone()
                beta[:, P] = 0
            if g is not None:
                g = g.clone()
                g[:, P] = 0
        return fn(query, key, value, g=g, beta=beta, **kw)
    inner._peerblock_wrapped = fn
    return inner


def _mixer(layer):
    if hasattr(layer, "self_attn"):
        return "attn", layer.self_attn
    if hasattr(layer, "linear_attn"):
        return "linear", layer.linear_attn
    raise ValueError(f"layer has neither self_attn nor linear_attn; children: {[n for n, _ in layer.named_children()]}")


@contextmanager
def blocked(h, text: str, blocks: list[Block]):
    """Register the block hooks for `blocks` while a prompt (possibly with tokens appended) runs."""
    blocks = [b for b in (blocks or []) if isinstance(b, Block)]
    if not blocks:
        yield
        return
    layers = get_layers(h.model)
    with ExitStack() as stack:
        wrapped_modules = set()
        for b in blocks:
            for li in b.layers:
                kind, mod = _mixer(layers[li])
                if kind == "attn":
                    hd = mod.register_forward_pre_hook(_attn_hook(b.positions, b.from_query), with_kwargs=True)
                    stack.callback(hd.remove)
                else:
                    hd = mod.register_forward_pre_hook(_linear_pre(b.positions), with_kwargs=True)
                    stack.callback(hd.remove)
                    hd2 = mod.register_forward_hook(_linear_post)
                    stack.callback(hd2.remove)
                    modname = type(mod).__module__
                    if modname not in wrapped_modules:
                        wrapped_modules.add(modname)
                        m = sys.modules[modname]
                        found = False
                        for name in RULE_NAMES:
                            if hasattr(m, name):
                                orig = getattr(m, name)
                                setattr(m, name, _wrap_rule(orig))
                                stack.callback(setattr, m, name, orig)
                                found = True
                        if not found:
                            raise RuntimeError(f"no gated-delta-rule function found in {modname}; names tried {RULE_NAMES}")
        _LIN_STATE["positions"] = None
        try:
            yield
        finally:
            _LIN_STATE["positions"] = None


# ------------------------------------------------------------------ the patch ---------------------------------
@torch.no_grad()
def capture(h, text: str, positions: list[int], layers_: list[int]) -> dict[int, torch.Tensor]:
    """{layer: [P, d] residual after block `layer` at `positions`} for the unedited prompt `text`."""
    blocks = get_layers(h.model)
    store = {}
    with ExitStack() as stack:
        for li in layers_:
            def hook(_m, _i, out, li=li):
                x = layer_output(out)
                store[li] = x[0, positions].detach().clone()
            hd = blocks[li].register_forward_hook(hook)
            stack.callback(hd.remove)
        ids = torch.tensor([h.encode(text)], device=h.device)
        h.model(input_ids=ids)
    return store


def _register_patch(h, stack: ExitStack, patch: Patch, store: dict[int, torch.Tensor]):
    blocks = get_layers(h.model)
    for li in patch.layers:
        def hook(_m, _i, out, li=li):
            x = layer_output(out).clone()
            idx = [p for p in patch.positions if p < x.shape[1]]
            src = store[li][: len(idx)].to(x.device, x.dtype)
            x[:, idx] = src[None].expand(x.shape[0], *src.shape)
            return replace_layer_output(out, x)
        hd = blocks[li].register_forward_hook(hook)
        stack.callback(hd.remove)


def _check_patch(h, text, patch: Patch):
    n_src, n_tgt = len(h.encode(patch.source_text)), len(h.encode(text))
    if n_src != n_tgt:
        raise ValueError(f"patch source has {n_src} tokens, target {n_tgt}; they must match so positions line up")


@contextmanager
def patched(h, text: str, patches: list[Patch]):
    """Capture each patch's source residuals (unedited), then write them into the target run."""
    patches = [p for p in (patches or []) if isinstance(p, Patch)]
    if not patches:
        yield
        return
    stores = []
    for p in patches:
        _check_patch(h, text, p)
        stores.append(capture(h, p.source_text, p.positions, p.layers))
    with ExitStack() as stack:
        for p, st in zip(patches, stores):
            _register_patch(h, stack, p, st)
        yield


# ------------------------------------------------------------------ the 2x2 -----------------------------------
def two_by_two(h, text_visible: str, bridge_words: list[str], layers_: list[int], peer_positions: list[int],
               candidates: list[str], formats=None, positions="all") -> dict:
    """P(candidates[0]) in the four cells: A intact/visible, B bridge removed/visible, C intact/peers blocked
    (every layer), D removed/blocked. Returns {"A": p, "B": p, "C": p, "D": p, "cells": {cell: answer_probs cands}}."""
    from .interventions import FORMATS, Edit, answer_probs
    formats = formats or FORMATS
    rm = Edit("ablate", list(bridge_words), list(layers_), positions)
    blk = Block(peer_positions, list(range(h.n_layers)))
    out = {"cells": {}}
    for cell, edits in (("A", []), ("B", [rm]), ("C", [blk]), ("D", [rm, blk])):
        r = answer_probs(h, text_visible, candidates, edits, formats=formats)["cands"]
        out["cells"][cell] = r
        out[cell] = r[candidates[0]]["p_full"]
    return out
