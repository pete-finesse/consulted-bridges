"""Experiment 1: when an agent gives in to its peers, does the bridge still push back?

For each group prompt (1-3 peers giving a wrong letter), remove words' J-lens directions in each
window (question onward) and record the change in LO = log P(peers' answer) - log P(right answer).
  bridge          Seattle. If it still pulls toward the right answer, removing it RAISES LO.
  bridge only     the part of the bridge not shared with its same-kind words (Boston, Chicago)
  same kind       each same-kind word (must not do what the bridge does)
  noise           12 everyday words matched in strength (the noise floor: max over words)
  answer          the right answer's own direction ("S"): a positive control, should raise LO a lot
Pushback = best window rise in LO from the bridge, minus the best rise from any everyday word.

  uv run python scripts/giveup_remove.py [--config qwen36_27b] [--ks 1 2 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from consulted_bridges.silent_dissent.giveup import Probe, lay, prompts  # noqa: E402
from consulted_bridges.interventions import Edit  # noqa: E402
from consulted_bridges.load import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen36_27b")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    h = load(args.config)
    out = Path(args.out or f"results/giveup_remove_{Path(args.config).stem}")
    out.mkdir(parents=True, exist_ok=True)
    rows, summ = [], []
    P = prompts(args.ks)
    for n, (name, (spec, peer, k)) in enumerate(P.items()):
        print(f"PROGRESS {n}/{len(P)} {name}", flush=True)
        try:
            pr = Probe(h, name, spec, peer)
            c, b = pr.c, spec["bridge"]
            p0r, p0w, lo0 = pr.probs()
            said = pr.answer()
            nb, (reach, rw) = pr.reach()
            same = [w for w in spec["same_kind"] if c.one_token(w)]
            nulls = pr.nulls()
            ans_dirs = list(dict.fromkeys(spec["answer_dirs"] + ["@" + f for f in pr.right]))
            words = {"bridge": lambda w: pr.rm(b, w), "answer": lambda w: pr.rm(ans_dirs, w)}
            if same:
                words["bridge only"] = lambda w: [Edit("ablate_excl", [bb] + same, w, c.ALL) for bb in b]
            for s in same:
                words[f"same kind: {s}"] = lambda w, s=s: pr.rm([s], w)
            for z in nulls:
                words[f"noise: {z}"] = lambda w, z=z: pr.rm([z], w)
            best = {}
            for label, make in words.items():
                for w in c.wins:
                    r, wr, lo = pr.probs(make(w))
                    rows.append(dict(prompt=name, question=name.split("__")[0], peers=k, edit=label, window=lay(w),
                                     start=w[0], P_right=r, P_peer=wr, LO=lo, dLO=lo - lo0))
                    if label not in best or lo - lo0 > best[label][0]:
                        best[label] = (lo - lo0, w, r, wr)
            noise = max((best[k_][0], k_) for k_ in best if k_.startswith("noise"))
            kind = max((best[k_][0], k_) for k_ in best if k_.startswith("same kind")) if same else (None, None)
            bd, bw, br, bwr = best["bridge"]
            summ.append(dict(prompt=name, question=name.split("__")[0], peers=k, peer_answer=peer, model_said=said,
                             P_right=p0r, P_peer=p0w, LO=lo0, reach=reach, bridge_dLO=bd, bridge_window=lay(bw),
                             bridge_P_right=br, bridge_P_peer=bwr, bridge_only_dLO=best.get("bridge only", (None,))[0],
                             same_kind_dLO=kind[0], same_kind_word=kind[1], noise_dLO=noise[0], noise_word=noise[1],
                             answer_dLO=best["answer"][0], pushback=bd - noise[0]))
            print(f"[{name}] says {said!r}; P(right) {p0r:.2f} P(peer {peer}) {p0w:.2f} LO {lo0:+.2f} | "
                  f"remove {b[0]!r} ({lay(bw)}): P(right) {p0r:.2f}→{br:.2f}, P(peer) {p0w:.2f}→{bwr:.2f}, "
                  f"LO {bd:+.2f} | noise max {noise[0]:+.2f} | same kind {kind[0] if kind[0] is None else round(kind[0], 2)} | "
                  f"answer dir {best['answer'][0]:+.2f} | reach {reach:.2f} | PUSHBACK {bd - noise[0]:+.2f}", flush=True)
        except Exception as err:
            import traceback
            traceback.print_exc()
            print(f"!! {name} failed: {err!r}", flush=True)
        pd.DataFrame(rows).to_csv(out / "windows.csv", index=False)
        pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    print("PROGRESS ALLDONE", flush=True)


if __name__ == "__main__":
    main()
