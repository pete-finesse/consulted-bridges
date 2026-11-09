"""Block validation, the 2x2, the dose-response and the patch (plan.md Phase C steps 5-7) on the conformity prompts.

For each question and each prompt of: alone, agree 1-3, disagree 1-3, hidden "?" 1-3, hidden "(answer submitted)" 3,
copy 3 (made-up subject, where it exists), the made-up subject alone ("copy_alone", the copy prompt's no-peer
level) and told_copy 3, the cells are

  A        no edit
  B        the bridge removed (ablate, from the question's first word to the end of the prompt) in the prompt's best
           early window, read from the conformity run's prompts table (bridge_drop_window; the alone prompt's window
           when the prompt has none)
  C        peerblock.Block at every layer over the peers' answer tokens: no later position can read them
  D        B and C together
  C_late   Block only at the full-attention layers and only for queries from the reply's first token
           (<|im_start|>assistant) on: the answer positions cannot attend to the peers directly. The linear-attention
           layers have no per-query block, so they are left open and still carry the peers forward (peerblock.py)
  D_late   B and C_late together
Alone and copy_alone have no peers: only A and B.
Each cell records P(right), P(peers'), and their log-odds (logit, P clipped to [1e-6, 1 - 1e-6]).

Dose-response (alone, hidden 3, agree 3): the B removal at strength 0.25, 0.5, 0.75 (1 is B).
Patch (--patch, agree 3 only): the hidden "?" prompt's residual at the peer positions written into the agree prompt
(same length, same positions; where the right letter is one token, " A" or " S", the placeholder is " ..." instead,
also one token, recorded as patch_source) at every layer (P_all), with the bridge removed too (PB_all), and from layer L upward
for every L (P_from_L).

  python -m consulted_bridges.conformity_block --config qwen35_4b_jpp --frame neutral \
      --windows results/conformity_v2_qwen35_4b_jpp --shard 0/2 --patch --out results/conformity_block_qwen35_4b_jpp
"""
from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path

import pandas as pd

from consulted_bridges import conformity
from consulted_bridges.conformity import FAKE, QUESTIONS, SPECS, Prompt, peer_positions, prompts, swap_target, wrong_letter
from consulted_bridges.interventions import Edit
from consulted_bridges.load import load
from consulted_bridges.peerblock import Block, Patch
from consulted_bridges.text import offsets

EPS = 1e-6
STRENGTHS = (0.25, 0.5, 0.75)
WANT = {("alone", 0), ("agree", 1), ("agree", 2), ("agree", 3), ("disagree", 1), ("disagree", 2), ("disagree", 3),
        ("hidden", 1), ("hidden", 2), ("hidden", 3), ("hidden_sub", 3), ("copy", 3), ("told_copy", 3)}
DOSE = {("alone", 0), ("hidden", 3), ("agree", 3)}


def logit(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return float("nan")
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def parse_window(w):
    """'layers 6–11' -> [6, ..., 11]"""
    if not isinstance(w, str) or not w.startswith("layers"):
        return None
    a, b = w.split()[1].split("–")
    return list(range(int(a), int(b) + 1))


def windows_table(d):
    # a run folder (every shard) or one shard folder (its own rows only, so a chained shard never reads a file
    # another shard is still writing)
    pat = f"{d}/prompts.csv" if Path(d, "prompts.csv").exists() else f"{d}/shard*/prompts.csv"
    fs = [f for f in sorted(glob.glob(pat)) if Path(f).stat().st_size > 1]
    if not fs:
        raise SystemExit(f"no prompts tables under {d}: run the conformity step first")
    p = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return {(r.question, r.condition, int(r.peers)): (r.bridge_drop_window, r.state) for r in p.itertuples()}


def reply_start(h, text):
    at = text.rfind("<|im_start|>")
    return next(i for i, (a, b) in enumerate(offsets(h, text)) if a >= at)


def attn_layers(h):
    from silent_dissent.model import get_layers
    return [i for i, l in enumerate(get_layers(h.model)) if hasattr(l, "self_attn")]


def question_prompts(q, frame):
    out = [(c, k, peer, s) for c, k, peer, s in prompts(q, frame, ("hidden", "hidden_sub", "told_copy")) if (c, k) in WANT]
    if q in FAKE:   # the made-up subject asked alone: the level P(peers' letter) should return to under the block
        s = SPECS[q]
        a, b = FAKE[q]
        out.append(("copy_alone", 0, wrong_letter(q, s), dict(s, user=s["user"].replace(a, b))))
    return out


def run_question(h, q, frame, W, do_patch, rows, dose, patch_rows):
    L = h.n_layers
    every, attn = list(range(L)), attn_layers(h)
    P = question_prompts(q, frame)
    texts = {}
    alone_w = parse_window(W.get((q, "alone", 0), (None, None))[0])
    for cond, k, peer, spec in P:
        t0 = time.time()
        pm = Prompt(h, q, cond, k, peer, spec)
        text = pm.c.text
        texts[(cond, k)] = (text, pm)
        wsrc, state = W.get((q, cond if cond != "copy_alone" else "alone", k), (None, None))
        win = parse_window(wsrc) if cond != "copy_alone" else alone_w
        source = "own" if win is not None and cond != "copy_alone" else "alone"
        win = win or alone_w
        b = spec["bridge"]
        rm = [Edit("ablate", b, win, pm.allpos)] if win else None
        pos = peer_positions(h, text, peer, k) if k else []
        cells = {"A": []}
        if rm:
            cells["B"] = rm
        if pos:
            blk = Block(pos, every)
            late = Block(pos, attn, from_query=reply_start(h, text), label="late")
            cells.update(C=[blk], C_late=[late])
            if rm:
                cells.update(D=rm + [blk], D_late=rm + [late])
        base = dict(question=q, frame=frame, condition=cond, peers=k, right=pm.right, peer_letter=None if pm.hidden or cond in ("alone", "copy_alone") else pm.pletter,
                    state_c=None if cond == "copy_alone" else state, window=None if not win else f"layers {win[0]}–{win[-1]}",
                    window_source=source if win else None, n_peer_tokens=len(pos))
        kw, ka = swap_target(spec)          # D's other destination: the first same-kind word's answer (Spain's S)
        for cell, edits in cells.items():
            pr, pw, px = pm.probs_x(edits, [ka])
            pk = px.get(ka, float("nan"))
            rows.append(dict(base, cell=cell, P_right=pr, P_peer=pw, lo_right=logit(pr), lo_peer=logit(pw),
                             kind_word=kw, kind_answer=ka, P_kind=pk, lo_kind=logit(pk),
                             wrong_rule=conformity.WRONG_RULE))
        if (cond, k) in DOSE and win:
            res = pm.probs_many([[Edit("ablate", b, win, pm.allpos, strength=s)] for s in STRENGTHS])
            for s, (pr, pw, _) in zip(STRENGTHS, res):
                dose.append(dict(base, strength=s, P_right=pr, P_peer=pw, lo_right=logit(pr)))
        print(f"[{q} {cond}{k or ''}] " + " ".join(f"{r['cell']} {r['P_right']:.2f}" for r in rows[-len(cells):])
              + f" ({time.time() - t0:.0f}s)", flush=True)
    if do_patch and ("agree", 3) in texts and ("hidden", 3) in texts:
        t0 = time.time()
        (text, pm), (src, _) = texts[("agree", 3)], texts[("hidden", 3)]
        pos = peer_positions(h, text, pm.peer, 3)
        source = "?"
        if len(h.encode(src)) != len(h.encode(text)):
            # " ?" is two tokens on Qwen (a space, then "?"), like " I" or " F"; " A", " B", " S"... are one token.
            # For those, the placeholder is " ..." (one token), so the positions still line up
            source = "..."
            src = text.replace(f"Agent 2: {pm.peer}\n", "Agent 2: ...\n").replace(f"Agent 3: {pm.peer}\n", "Agent 3: ...\n") \
                      .replace(f"Agent 4: {pm.peer}\n", "Agent 4: ...\n")
            if len(h.encode(src)) != len(h.encode(text)) or peer_positions(h, src, "...", 3) != pos:
                print(f"!! {q}: no same-length placeholder; patch skipped", flush=True)
                return
        wsrc, _ = W.get((q, "agree", 3), (None, None))
        win = parse_window(wsrc) or alone_w
        base = dict(question=q, frame=frame, condition="agree", peers=3, patch_source=source)
        jobs = [("P_all", None, [Patch(src, pos, every)])]
        if win:
            jobs.append(("PB_all", None, [Edit("ablate", pm.spec["bridge"], win, pm.allpos), Patch(src, pos, every)]))
        jobs += [("P_from_L", l0, [Patch(src, pos, list(range(l0, L)))]) for l0 in range(L)]
        for cell, l0, edits in jobs:
            pr, pw, _ = pm.probs(edits)
            patch_rows.append(dict(base, cell=cell, from_layer=l0, P_right=pr, P_peer=pw, lo_right=logit(pr)))
        pa = next(r for r in patch_rows[-len(jobs):] if r["cell"] == "P_all")
        print(f"[{q} patch] P_all {pa['P_right']:.2f} ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen35_4b_jpp")
    ap.add_argument("--frame", default="neutral", choices=["neutral", "defer"])
    ap.add_argument("--windows", required=True, help="the conformity run (same config and frame) whose windows B uses")
    ap.add_argument("--questions", nargs="*", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--wrong-rule", default=None, choices=conformity.WRONG_RULES,
                    help="the peers' wrong answer: free (default, run 3 on) or kind (runs 1-2); must match the windows run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.wrong_rule:
        conformity.WRONG_RULE = args.wrong_rule
    cfg = Path(args.config).stem
    qs = args.questions or QUESTIONS.get(cfg.replace("_jpp", ""), [])
    i, n = map(int, args.shard.split("/"))
    qs = [q for j, q in enumerate(qs) if j % n == i]
    out = Path(args.out or f"results/conformity_block_{'' if args.frame == 'neutral' else args.frame + '_'}{cfg}") / f"shard{i}"
    out.mkdir(parents=True, exist_ok=True)
    W = windows_table(args.windows)
    h = load(args.config)
    rows, dose, patch_rows = [], [], []
    for j, q in enumerate(qs):
        print(f"PROGRESS {j}/{len(qs)} {q}", flush=True)
        try:
            run_question(h, q, args.frame, W, args.patch, rows, dose, patch_rows)
        except Exception as err:
            import traceback
            traceback.print_exc()
            print(f"!! {q} failed: {err!r}", flush=True)
        pd.DataFrame(rows).to_csv(out / "cells.csv", index=False)
        pd.DataFrame(dose).to_csv(out / "dose.csv", index=False)
        pd.DataFrame(patch_rows).to_csv(out / "patch.csv", index=False)
    print("PROGRESS ALLDONE", flush=True)


if __name__ == "__main__":
    main()
