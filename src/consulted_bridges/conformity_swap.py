"""Swap toward another bridge inside group prompts (plan.md Phase C step 7b), rules fixed before GPU run 3.

If removing the bridge drops the answer, that shows the bridge is needed; swapping it shows the letter is *made*
from it. Neuronpedia's move (interventions kind "move": h <- h + (h.s)(t - s)) replaces the bridge's J-lens
component with a same-kind word's (Italy -> Spain). If the answer becomes that word's letter (S) while the peers say
I (agree) or F (disagree), the model's answer is produced by its own bridge, not copied.

For each question and wording, on the conformity prompts alone, agree 3 and disagree 3 (identical text):
  positions  the question's own words: the tokens of the question sentence (without the answer-format line), the
             same tokens in every condition (docs/swap_replication.md: question words only pass the everyday
             control at every layer range; question plus template lets "table" through at layers 32-53)
  layers     0 .. onset - 1, the prompt's own answer onset from the conformity run (the alone prompt's if missing)
  target     conformity.swap_target: the first same-kind word whose answer differs from the right one
  control    the first everyday word (EVERYDAY) whose letter is none of right, peers', target's (letters only;
             number questions have no everyday answer, so the control is not readable there)
  cells      A  no edit;  R  the bridge removed on the same positions and layers (what the move contains);
             M  move bridge -> target;  E  move bridge -> everyday word;
             with peers (agree 3, disagree 3) also under the peer block at every layer: C, CR, CM, CE.
Recorded per cell: P(right), P(peers'), P(target's answer), P(everyday letter) and the greedy answer.
  follows        the greedy answer is the target's answer (M, CM)
  gets through   the greedy answer is the everyday word's letter (E, CE): the control fails
Reading (no tuning after the run): per condition, the share of questions whose M follows, next to the share whose
E gets through; "under the block at least as often" compares CM with M.

  python -m consulted_bridges.conformity_swap --config qwen36_27b_jpp --frame neutral \
      --windows results/conformity_v3_qwen36_27b_jpp --questions C_LM_moai ... --out results/conformity_swap_qwen36_27b_jpp
"""
from __future__ import annotations

import argparse
import glob
import math
import time
from pathlib import Path

import pandas as pd

from consulted_bridges import conformity
from consulted_bridges.conformity import SPECS, Prompt, answer_format, peer_positions, prompts, suffix_of, swap_target
from consulted_bridges.interventions import Edit
from consulted_bridges.load import load
from consulted_bridges.peerblock import Block

EPS = 1e-6
EVERYDAY = ["table", "river", "garden", "lamp", "bottle", "window"]
WANT = [("alone", 0), ("agree", 3), ("disagree", 3)]


def logit(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return float("nan")
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def onsets(d):
    pat = f"{d}/prompts.csv" if Path(d, "prompts.csv").exists() else f"{d}/shard*/prompts.csv"
    fs = [f for f in sorted(glob.glob(pat)) if Path(f).stat().st_size > 1]
    if not fs:
        raise SystemExit(f"no prompts tables under {d}: run the conformity step first")
    p = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return {(r.question, r.condition, int(r.peers)): r.onset for r in p.itertuples()}


def question_positions(h, text, spec):
    """Tokens of the question sentence (the user text without its answer-format line), last occurrence."""
    q = spec["user"][: -len(suffix_of(spec["user"]))] if "Question: " not in spec["user"] else \
        spec["user"].split("Question: ", 1)[1].split("\n", 1)[0]
    from consulted_bridges.text import resolve_positions
    return resolve_positions(h, text, f"span:{q}")


def everyday_word(card, avoid):
    return next((w for w in EVERYDAY if card.one_token(w) and w[0].upper() not in avoid), None)


def run_question(h, q, frame, O, rows):
    S0 = SPECS[q]
    fmt = answer_format(S0)
    tw, ta = swap_target(S0)
    if tw is None:
        print(f"!! {q}: no same-kind target with another answer; skipped", flush=True)
        return
    P = {(c, k): (peer, s) for c, k, peer, s in prompts(q, frame) if (c, k) in WANT}
    alone_on = O.get((q, "alone", 0))
    for cond, k in WANT:
        t0 = time.time()
        peer, spec = P[(cond, k)]
        pm = Prompt(h, q, cond, k, peer, spec)
        text, c = pm.c.text, pm.c
        on = O.get((q, cond, k))
        src = "own"
        if on is None or (isinstance(on, float) and math.isnan(on)):
            on, src = alone_on, "alone"
        if on is None or (isinstance(on, float) and math.isnan(on)) or int(on) < 1:
            print(f"!! {q} {cond}{k}: no onset; skipped", flush=True)
            continue
        layers = list(range(int(on)))
        pos = question_positions(h, text, spec)
        b = spec["bridge"][0]
        avoid = {pm.right, ta} | ({pm.pletter} if k else set())
        ev = everyday_word(c, avoid) if fmt == "letter" else None
        ea = ev[0].upper() if ev else None
        cells = {"A": [], "R": [Edit("ablate", [b], layers, pos)], "M": [Edit("move", [b, tw], layers, pos)]}
        if ev:
            cells["E"] = [Edit("move", [b, ev], layers, pos)]
        if k:
            blk = Block(peer_positions(h, text, peer, k), list(range(h.n_layers)))
            cells.update({"C" + x if x != "A" else "C": e + [blk] for x, e in list(cells.items())})
        base = dict(question=q, family=q.rsplit("_", 1)[0], format=fmt, frame=frame, condition=cond, peers=k,
                    right=pm.right, peer_letter=pm.pletter if k else None, target=tw, target_answer=ta, everyday=ev,
                    everyday_answer=ea, onset=int(on), onset_source=src, layers=f"0-{int(on) - 1}",
                    positions=f"{pos[0]}-{pos[-1]}", n_positions=len(pos), wrong_rule=conformity.WRONG_RULE)
        for cell, edits in cells.items():
            pr, pw, px = pm.probs_x(edits, [ta, ea])
            said = c.gen(text, edits)
            got = c.classify(said)
            rows.append(dict(base, cell=cell, P_right=pr, P_peer=pw, P_target=px.get(ta), P_everyday=px.get(ea),
                             lo_right=logit(pr), lo_target=logit(px.get(ta)), model_said=said, answered=got,
                             follows=got == ta if cell in ("M", "CM") else None,
                             through=(got == ea) if ev and cell in ("E", "CE") else None))
        print(f"[{q} {cond}{k or ''}] " + " ".join(f"{r['cell']}:{r['answered']}" for r in rows[-len(cells):])
              + f" (target {tw}->{ta}, everyday {ev}) ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen36_27b_jpp")
    ap.add_argument("--frame", default="neutral", choices=["neutral", "defer"])
    ap.add_argument("--windows", required=True, help="the conformity run (same config, frame, wrong rule): onsets")
    ap.add_argument("--questions", nargs="+", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--wrong-rule", default=None, choices=conformity.WRONG_RULES)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.wrong_rule:
        conformity.WRONG_RULE = args.wrong_rule
    i, n = map(int, args.shard.split("/"))
    qs = [q for j, q in enumerate(args.questions) if j % n == i]
    out = Path(args.out) / f"shard{i}"
    out.mkdir(parents=True, exist_ok=True)
    O = onsets(args.windows)
    h = load(args.config)
    rows = []
    for j, q in enumerate(qs):
        print(f"PROGRESS {j}/{len(qs)} {q}", flush=True)
        try:
            run_question(h, q, args.frame, O, rows)
        except Exception as err:
            import traceback
            traceback.print_exc()
            print(f"!! {q} failed: {err!r}", flush=True)
        pd.DataFrame(rows).to_csv(out / "swap.csv", index=False)
    print("PROGRESS ALLDONE", flush=True)


if __name__ == "__main__":
    main()
