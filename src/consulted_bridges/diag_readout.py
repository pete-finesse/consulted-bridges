"""Why does Llama's J-lens lose the bridge under the deferential wording?

In results/conformity_defer_llama31_8b_it every group prompt ranks the bridge 553 or worse at the last
token, and the same-kind and unrelated control words drop just as far, while the alone prompt (identical
in both runs) ranks it 0 and the neutral wording ranks it 0-4. The removal test still finds the bridge
pulling in 19 of those prompts. This reads a few prompts under several wordings and asks:

  1. Is it the last token only? Best rank of the bridge over the question positions vs at the last token.
  2. What does the last token hold instead? The J-lens top words there, per band layer.
  3. Is the residual stream unusual there? Its norm per band layer, against the alone prompt.
  4. Which part of the wording does it? The two frames plus variants that change one phrase at a time.

    python -m consulted_bridges.diag_readout --config llama31_8b_it [--questions L_colosseum ...]
Writes results/diag_readout_<cfg>/{ranks.csv, top.csv, norms.csv}. About 10 forward passes per question.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from consulted_bridges.conformity import FRAMES, ONE, SPECS, SUFFIX, UNRELATED, group_user, questions_for, wrong_letter
from consulted_bridges.load import load
from consulted_bridges.readout import readout
from consulted_bridges.text import chat, resolve_positions
from silent_dissent.model import get_layers, layer_output

VARIANTS = dict(FRAMES, **{
    "defer_own": "You are Agent 1. Taking the other agents' answers into account, give your own answer.",
    "consider": "You are Agent 1. Consider the other agents' answers, then give your answer.",
    "plain": "You are Agent 1. Give your answer.",
    "no_agent": "Taking the other agents' answers into account, give your answer.",
})


@torch.no_grad()
def norms(h, text, layers):
    ids = h.encode(text)
    store = {}
    blocks = get_layers(h.model)
    hs = [blocks[l].register_forward_hook(lambda _m, _i, o, l=l: store.__setitem__(l, layer_output(o)[0, -1].float().norm().item()))
          for l in layers]
    try:
        h.model(input_ids=torch.tensor([ids], device=h.device))
    finally:
        for x in hs:
            x.remove()
    return store


@torch.no_grad()
def top_words(h, text, layers, k=12):
    ids = h.encode(text)
    store = {}
    blocks = get_layers(h.model)
    hs = [blocks[l].register_forward_hook(lambda _m, _i, o, l=l: store.__setitem__(l, layer_output(o)[0, -1:].detach()))
          for l in layers]
    try:
        h.model(input_ids=torch.tensor([ids], device=h.device))
    finally:
        for x in hs:
            x.remove()
    lens = h.lenses["jlens"]
    return {l: [h.tok.decode([t]) for t in lens.logits(store[l], l).float()[0].topk(k).indices.tolist()] for l in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="llama31_8b_it")
    ap.add_argument("--questions", nargs="*")
    ap.add_argument("--n", type=int, default=3, help="first N selected questions if --questions is not given")
    args = ap.parse_args()
    cfg = Path(args.config).stem
    qs = args.questions or questions_for(cfg)[: args.n]
    h = load(args.config)
    band = list(h.band)
    out = Path(f"results/diag_readout_{cfg}")
    out.mkdir(parents=True, exist_ok=True)
    R, T, N = [], [], []
    for qid in qs:
        s = SPECS[qid]
        sfx = SUFFIX if s["user"].endswith(SUFFIX) else ONE
        q = s["user"][: -len(sfx)]
        right, wrong = s["answer"], wrong_letter(qid, s)
        same = [w for w in s["same_kind"] if len(h.encode(" " + w)) == 1][:1]
        ctrl = next(w for w in UNRELATED if w not in s["bridge"] and w not in s["same_kind"])
        words = [s["bridge"][0]] + same + [ctrl]
        cases = [("alone", 0, None, s["user"])]
        for name, frame in VARIANTS.items():
            for cond, peer in (("agree", right), ("disagree", wrong)):
                FRAMES["_v"] = frame        # group_user reads the wording by key; add the variant for this call
                cases.append((f"{cond}3", 3, name, group_user(q, peer, 3, "_v", sfx)))
                del FRAMES["_v"]
        for cond, k, name, user in cases:
            text = chat(h, user, "")
            qpos = resolve_positions(h, text, f"from:{s['q_start']}")
            ro = readout(h, text, words, lenses=["jlens"])
            r = ro.ranks[0]                                     # [position, layer, word]
            pos_index = {p: i for i, p in enumerate(ro.positions)}
            qi = [pos_index[p] for p in qpos if p in pos_index and p != ro.positions[-1]]
            for j, w in enumerate(words):
                last = [int(r[-1, l, j]) for l in band if r[-1, l, j] >= 0]
                over_q = [int(r[i, l, j]) for i in qi for l in band if r[i, l, j] >= 0]
                R.append(dict(question=qid, case=cond, wording=name, word=w, role=(["bridge"] + ["same kind"] * len(same) + ["unrelated"])[j],
                              rank_last=min(last) if last else None, rank_question_positions=min(over_q) if over_q else None,
                              last_token=repr(ro.tokens[-1])))
            for l, ws in top_words(h, text, band[::2]).items():
                T.append(dict(question=qid, case=cond, wording=name, layer=l, top=" | ".join(ws)))
            for l, v in norms(h, text, band).items():
                N.append(dict(question=qid, case=cond, wording=name, layer=l, norm=v))
            b = [x for x in R if x["question"] == qid and x["case"] == cond and x["wording"] == name and x["role"] == "bridge"][0]
            print(f"[{qid} {cond} {name}] bridge rank last {b['rank_last']} | best over question {b['rank_question_positions']}", flush=True)
    pd.DataFrame(R).to_csv(out / "ranks.csv", index=False)
    pd.DataFrame(T).to_csv(out / "top.csv", index=False)
    pd.DataFrame(N).to_csv(out / "norms.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
