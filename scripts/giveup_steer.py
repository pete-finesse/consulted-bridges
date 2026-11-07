"""Experiment 2: does strengthening the bridge make an agent hold against its peers?

v2: "amplify" instead of additive steering. Adding alpha x |h| of a direction over 12 layers broke
Qwen3.6-27B even at alpha 0.02 (P > 1 in the alone check, empty answers). Amplify scales the component
the residual already has along the word's direction by (1 + alpha), alpha in {0.5, 1, 2}.

For each group prompt (1-3 peers giving a wrong letter), push words' J-lens directions UP (steer, alpha
= fraction of the residual's size, added at every position from the question on, in one window of layers)
and record P(right answer), P(peers' answer), LO and what the model answers.
  bridge      Seattle. If its strength decides holding vs giving in, P(right) rises and LO falls.
  same kind   a same-kind word (Boston) steered by the same amount (must not do the same)
  noise       3 everyday words steered by the same amount
  answer      the right answer's own letter ("S"): a positive control (steering can move the answer)
Also: the bridge steered in the question asked ALONE, to check that the push doesn't just break the model.

  uv run python scripts/giveup_steer.py [--config qwen36_27b] [--ks 1 2 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from consulted_bridges.silent_dissent.giveup import S3, Probe, lay, prompts  # noqa: E402
from consulted_bridges.interventions import Edit  # noqa: E402
from consulted_bridges.load import load  # noqa: E402

ALPHAS = [0.5, 1.0, 2.0]   # amplify the existing component by +50%, x2, x3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen36_27b")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    h = load(args.config)
    out = Path(args.out or f"results/giveup_steer_{Path(args.config).stem}")
    out.mkdir(parents=True, exist_ok=True)
    rows, summ = [], []
    P = prompts(args.ks)
    for n, (name, (spec, peer, k)) in enumerate(P.items()):
        print(f"PROGRESS {n}/{len(P)} {name}", flush=True)
        try:
            pr = Probe(h, name, spec, peer)
            c, b = pr.c, spec["bridge"]
            alone = Probe(h, name + "__alone", S3[name.split("__")[0]], peer)
            p0r, p0w, lo0 = pr.probs()
            said = pr.answer()
            same = [w for w in spec["same_kind"] if c.one_token(w)][:1]
            noise = pr.nulls(3)
            steer = {"bridge": b[0], "answer": "@" + pr.right[0]}
            steer.update({f"same kind: {s}": s for s in same})
            steer.update({f"noise: {z}": z for z in noise})
            # windows: the middle of the model, before the answer forms (2nd to 5th of the 10 windows)
            wins = c.wins[1:5] or c.wins
            best = {}
            for label, word in steer.items():
                for w in wins:
                    for a in ALPHAS:
                        e = [Edit("amplify", [word], w, c.ALL, alpha=a)]
                        r, wr, lo = pr.probs(e)
                        ans = pr.answer(e) if label in ("bridge", "answer") or label.startswith("same") else ""
                        rows.append(dict(prompt=name, question=name.split("__")[0], peers=k, steer=label, window=lay(w),
                                         alpha=a, P_right=r, P_peer=wr, LO=lo, dLO=lo - lo0, answer=ans,
                                         holds=(c.classify(ans) == spec["answer"]) if ans else None))
                        if label not in best or lo - lo0 < best[label][0]:
                            best[label] = (lo - lo0, w, a, r, wr, ans)
                        if label == "bridge":
                            ar, _, _ = alone.probs(e)
                            rows[-1]["alone_P_right"] = ar
            nz = min((best[x][0], x) for x in best if x.startswith("noise"))
            sk = min((best[x][0], x) for x in best if x.startswith("same")) if same else (None, None)
            bd, bw, ba, br, bwr, bans = best["bridge"]
            flips = sum(1 for r_ in rows if r_["prompt"] == name and r_["steer"] == "bridge" and r_["holds"])
            summ.append(dict(prompt=name, question=name.split("__")[0], peers=k, peer_answer=peer, model_said=said,
                             P_right=p0r, P_peer=p0w, LO=lo0, bridge_dLO=bd, bridge_window=lay(bw), bridge_alpha=ba,
                             bridge_P_right=br, bridge_P_peer=bwr, bridge_answer=bans, bridge_holds_cells=flips,
                             same_kind_dLO=sk[0], noise_dLO=nz[0], noise_word=nz[1], answer_dLO=best["answer"][0],
                             effect=nz[0] - bd))
            print(f"[{name}] says {said!r}; P(right) {p0r:.2f} P(peer {peer}) {p0w:.2f} LO {lo0:+.2f} | "
                  f"steer {b[0]!r} +{ba} ({lay(bw)}): P(right) {p0r:.2f}→{br:.2f}, P(peer) {p0w:.2f}→{bwr:.2f}, "
                  f"says {bans!r}; holds in {flips}/{len(wins) * len(ALPHAS)} | noise best dLO {nz[0]:+.2f} | "
                  f"same kind {sk[0] if sk[0] is None else round(sk[0], 2)} | answer dir {best['answer'][0]:+.2f} | "
                  f"EFFECT {nz[0] - bd:+.2f}", flush=True)
        except Exception as err:
            import traceback
            traceback.print_exc()
            print(f"!! {name} failed: {err!r}", flush=True)
        pd.DataFrame(rows).to_csv(out / "cells.csv", index=False)
        pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    print("PROGRESS ALLDONE", flush=True)


if __name__ == "__main__":
    main()
