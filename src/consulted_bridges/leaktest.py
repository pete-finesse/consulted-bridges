"""The peer-block leak test on a real model (plan.md Phase C step 4; tests/test_peerblock.py is the tiny-model version).

Two group prompts that differ only in the peers' letter (peers "I" and peers "F", and "?" against "I"), the same
length and the same peer positions. With peerblock.Block at every layer over the peer positions, the last-position
logits (float32) must not depend on what the peers wrote. Also checked: without the block the logits differ (the
peers matter), and the block changes the output (it does something). Which gated-delta-rule implementation the
linear-attention layers will call is recorded, since a hub kernel that replaces the whole forward would bypass the
block's beta / g zeroing.

Exit code 0 if every pair passes, 2 if any leaks. Writes <out>/leak_test.json.

  python -m consulted_bridges.leaktest --config qwen35_4b_jpp --out results/v2_run1_qwen35_4b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from consulted_bridges.conformity import SPECS, group_user
from consulted_bridges.interventions import applied
from consulted_bridges.load import load
from consulted_bridges.peerblock import RULE_NAMES, Block, peer_token_positions
from consulted_bridges.text import chat

TOL = 1e-3      # float32 max |logit difference|; identical inputs at every unblocked position should give 0


@torch.no_grad()
def last_logits(h, text, edits=None):
    ids = torch.tensor([h.encode(text)], device=h.device)
    with applied(h, text, edits or []):
        return h.model(input_ids=ids).logits[0, -1].float().cpu()


def rule_impls(h):
    from silent_dissent.model import get_layers
    out = {}
    for layer in get_layers(h.model):
        mod = getattr(layer, "linear_attn", None)
        if mod is None:
            continue
        m = sys.modules[type(mod).__module__]
        for name in RULE_NAMES:
            if hasattr(m, name):
                f = getattr(m, name)
                out[name] = f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', repr(f))}"
        out["forward"] = f"{type(mod).forward.__module__}.{type(mod).forward.__qualname__}"
        break
    return out


def pair(h, qid, k, a, b, frame="neutral"):
    s = SPECS[qid]
    sfx = " Answer with one letter."
    q = s["user"][: -len(sfx)]
    ta, tb = chat(h, group_user(q, a, k, frame, sfx)), chat(h, group_user(q, b, k, frame, sfx))
    na, nb = len(h.encode(ta)), len(h.encode(tb))
    pa, pb = peer_token_positions(h, ta), peer_token_positions(h, tb)
    r = dict(question=qid, peers=k, a=a, b=b, frame=frame, same_length=na == nb, same_positions=pa == pb, positions=pa)
    if na != nb or pa != pb:
        r.update(passed=False, why="lengths or positions differ")
        return r
    blk = [Block(pa, list(range(h.n_layers)))]
    la, lb = last_logits(h, ta), last_logits(h, tb)
    ba, bb = last_logits(h, ta, blk), last_logits(h, tb, blk)
    top = lambda x: [h.tok.decode([int(i)]) for i in x.topk(5).indices]  # noqa: E731
    r.update(unblocked_diff=float((la - lb).abs().max()), blocked_diff=float((ba - bb).abs().max()),
             block_effect=float((ba - la).abs().max()), top_a=top(la), top_b=top(lb), top_blocked=top(ba))
    r["passed"] = r["blocked_diff"] <= TOL and r["unblocked_diff"] > TOL and r["block_effect"] > TOL
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen35_4b_jpp")
    ap.add_argument("--out", default="results/v2_run1_qwen35_4b")
    ap.add_argument("--questions", nargs="*", default=["L_colosseum", "L_gates", "V_LM_eiffel", "V_LU_berklee"])
    args = ap.parse_args()
    h = load(args.config)
    res = dict(config=args.config, USE_HUB_KERNELS=os.environ.get("USE_HUB_KERNELS", "(unset)"), impls=rule_impls(h),
               dtype=str(next(h.model.parameters()).dtype), tolerance=TOL, pairs=[])
    jobs = [("L_colosseum", k, "I", "F") for k in (1, 2, 3)] + [("L_colosseum", 3, "I", "?")]
    # " A", " B", " S"... are one token on Qwen, " F", " I", " ?" two (a space, then the letter): pair each right
    # letter with another letter of the same length so the two prompts line up
    same_len = lambda a: next(b for b in "FGHKQZJXSB" if b != a and len(h.encode(f": {b}\n")) == len(h.encode(f": {a}\n")))  # noqa: E731
    jobs += [(q, 3, SPECS[q]["answer"], same_len(SPECS[q]["answer"])) for q in args.questions[1:]]
    for job in jobs:
        r = pair(h, *job)
        res["pairs"].append(r)
        print(f"{r['question']} k={r['peers']} {r['a']}/{r['b']}: unblocked {r.get('unblocked_diff', float('nan')):.4g} "
              f"blocked {r.get('blocked_diff', float('nan')):.4g} effect {r.get('block_effect', float('nan')):.4g} "
              f"-> {'PASS' if r['passed'] else 'FAIL'}", flush=True)
    res["passed"] = all(r["passed"] for r in res["pairs"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "leak_test.json" if os.environ.get("USE_HUB_KERNELS", "") != "0" else "leak_test_no_hub_kernels.json"
    (out / name).write_text(json.dumps(res, indent=1))
    print("LEAK TEST", "PASS" if res["passed"] else "FAIL", res["impls"], flush=True)
    sys.exit(0 if res["passed"] else 2)


if __name__ == "__main__":
    main()
