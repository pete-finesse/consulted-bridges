"""Bridge scorecard: is a hidden middle step a CONSULTED BRIDGE (like spider) or an ANSWER ECHO (like Salzburg)?

For each sentence in bridge_specs.py, it measures seven numbers. P = the model's probability of the
correct answer. drop = 1 - P_after / P_before (1 = the answer is gone, 0 = no change). Edits apply
from the question's first word on, in sliding 6-layer windows; a drop is the best window's.

  N  necessity        drop from removing the bridge's J-lens direction.
  S  specificity      N minus the largest drop from removing an off-path word that is about as strongly
                      present (chosen automatically; direction overlap < 0.4 with the bridge and answer).
  O  answer overlap   cosine between the bridge's and the answer's directions (reported, not scored).
  U  unique need      drop from removing only the part of the bridge NOT shared with the answer.
  D  distinctness     U minus the largest same-kind U (another Austrian city minus Austria, ...).
  T  after reading    drop when removing it only after the question, minus drop when removing it
                      only on the question's own words (layers 8-23). > 0 = worked out after reading.
  K  same-answer swap share of swaps to an alternative with the SAME answer that keep it (octopus).
  F  follows swaps    share of swaps to alternatives with a different answer that move the answer away;
                      F_hit: share landing on exactly the expected answer, not counting the fallback
                      answer (what the model says with the bridge removed), which is no evidence.

Verdict (thresholds fixed before running new sentences; set on spider and Salzburg only):
  CONSULTED BRIDGE  N >= 0.8, S >= 0.5, U >= 0.5, D >= 0.3, K >= 0.5 and T >= 0
  ANSWER ECHO       N >= 0.8 but U < 0.5 or D < 0.3 or K < 0.5  (it matters only as part of the answer
                    or of its whole kind)
  BYSTANDER         N < 0.5  (visible in J-space but not needed)
  UNCLEAR           anything else

  uv run python scripts/bridge_card.py [--spec spider mozart ...] [--config qwen35_4b]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch


from consulted_bridges.legacy.bridge_specs import SPECS  # noqa: E402
from consulted_bridges.interventions import Edit, answer_probs, directions, generate  # noqa: E402
from consulted_bridges.load import load  # noqa: E402
from consulted_bridges.readout import readout  # noqa: E402
from consulted_bridges.text import chat, resolve_positions  # noqa: E402
from silent_dissent.model import get_layers, layer_output  # noqa: E402

pd.set_option("display.width", 250)
FORMATS = ("{}", "**{}")
TH = dict(N=0.8, S=0.5, U=0.5, D=0.3, K=0.5, T=0.0)


@torch.no_grad()
def capture(h, text):
    store = {}
    hs = [b.register_forward_hook(lambda _m, _i, o, l=l: store.__setitem__(l, layer_output(o)[0].detach()))
          for l, b in enumerate(get_layers(h.model))]
    try:
        h.model(input_ids=torch.tensor([h.encode(text)], device=h.device))
    finally:
        for x in hs:
            x.remove()
    return store


def unit(h, word, l):
    return directions(h, Edit("ablate", [word], [l]), l)[0][0]


@torch.no_grad()
def strength(h, store, word, layers, pos):
    v = []
    for l in layers:
        D = directions(h, Edit("ablate", [word], [l]), l)[0].to(store[l].device)
        H = store[l][pos].float()
        v.append(((H @ D.T).abs().max(-1).values / H.norm(dim=-1)).max().item())
    return sum(v) / len(v)


def card(h, name, S, n_auto=6, verbose=True):
    L = h.n_layers
    clip = lambda ls: [l for l in ls if l < L] or [L - 1]  # noqa: E731
    say = print if verbose else (lambda *a, **k: None)
    text = chat(h, S["user"], "")
    ALL, QS, AFTER = f"from:{S['q_start']}", f"span:{S['user']}", "from:<|im_end|>"
    flat = {f: k for k, forms in S["answers"].items() for f in forms}
    ans_forms = S["answers"][S["answer"]]
    b = S["bridge"][0]

    def P(edits):
        r = answer_probs(h, text, ans_forms, edits, formats=FORMATS)
        return sum(v.get(f"p[{f}]", 0.0) for v in r["cands"].values() for f in FORMATS)

    def classify(ans):
        a = ans.replace("<|im_end|>", "").strip(" .*\n").lower()
        for f, k in sorted(flat.items(), key=lambda x: -len(x[0])):
            if a.startswith(f.lower()):
                return k
        return "other"

    def answer(edits):
        return generate(h, text, edits, max_new_tokens=5).replace("<|im_end|>", "").strip()

    base = P([])
    base_ans = answer([])
    if classify(base_ans) != S["answer"]:      # the model doesn't give the right answer: nothing to test
        say(f"\n######## {name}: {S['user']!r}\nbaseline answer {base_ans!r} is not {S['answer']}: NOT KNOWN")
        return dict(sentence=name, group=S.get("group", ""), predict=S.get("predict", ""), question=S["user"],
                    bridge=b, answer=S["answer"], baseline_answer=base_ans, P_base=round(base, 3),
                    verdict="NOT KNOWN")
    drop = lambda p: max(0.0, 1 - p / base) if base > 0 else 0.0  # noqa: E731
    wins = [clip(range(s, s + 6)) for s in range(0, L - 3, 3)]
    best = lambda make: max(drop(P(make(w))) for w in wins)  # noqa: E731
    say(f"\n######## {name}: {S['user']!r}\nbaseline answer {base_ans!r}, P({S['answer']}) = {base:.2f}")

    # J-lens presence of the bridge, and off-path controls of matched strength
    store = capture(h, text)
    allpos = resolve_positions(h, text, ALL)
    mid = clip(range(12, 22))
    ro = readout(h, text, S["bridge"][:1], lenses=["jlens"], dedupe=False)
    b_rank = int(ro.summary(clip(range(12, 30)), allpos[0])["allpos_best_rank"].iloc[0])
    sb = strength(h, store, b, mid, allpos)
    lens = h.lenses["jlens"]
    tally = Counter()
    for l in clip(range(8, 24)):
        for row in lens.logits(store[l][allpos], l).float().topk(20, dim=-1).indices.tolist():
            tally.update({d.strip() for d in (h.tok.decode([t]) for t in row) if d.startswith(" ")})
    known = {w.lower() for w in S["bridge"] + list(flat) + S["exclude"] + S["same_kind"] + list(S["swaps"])}
    known |= {w.lower().strip("?.,") for w in S["user"].split()} | {"answer", "word", "one"}
    pool = [w for w, _ in tally.most_common(200) if w.isalpha() and len(w) > 3 and w.lower() not in known
            and not any(k.startswith(w.lower()) or w.lower().startswith(k) for k in known)][:50]
    cos_ok = {}
    for w in pool:
        try:
            cb = max(abs(float(unit(h, w, l) @ unit(h, b, l))) for l in clip([12, 18, 24]))
            ca = max(abs(float(unit(h, w, l) @ unit(h, S["exclude"][0], l))) for l in clip([12, 18, 24]))
        except ValueError:
            continue
        if cb < 0.4 and ca < 0.4:
            cos_ok[w] = strength(h, store, w, mid, allpos)
    auto = sorted(cos_ok, key=lambda w: abs(cos_ok[w] - sb))[:n_auto]
    O = sum(float(unit(h, b, l) @ unit(h, S["exclude"][0], l)) for l in clip([12, 18, 24])) / len(clip([12, 18, 24]))

    # N, S
    N = best(lambda w: [Edit("ablate", S["bridge"], w, ALL)])
    ctrl = {c: best(lambda w, c=c: [Edit("ablate", [c], w, ALL)]) for c in auto}
    Sx = N - max(ctrl.values(), default=0.0)
    say(f"bridge best J-lens rank {b_rank}, strength {sb:.3f}; auto controls "
        + ", ".join(f"{c}({cos_ok[c]:.3f})={v:.2f}" for c, v in ctrl.items()))
    # U, D
    U = best(lambda w: [Edit("ablate_excl", [b] + S["exclude"], w, ALL)])
    same = {}
    for c in S["same_kind"]:
        try:
            same[c] = best(lambda w, c=c: [Edit("ablate_excl", [c] + S["exclude"], w, ALL)])
        except ValueError:
            pass
    D = U - max(same.values(), default=0.0)
    say("same kind (minus answer): " + ", ".join(f"{c}={v:.2f}" for c, v in same.items()))
    # T
    ls = clip(range(8, 24))
    t_after = drop(P([Edit("ablate", S["bridge"], ls, AFTER)]))
    t_q = drop(P([Edit("ablate", S["bridge"], ls, QS)]))
    T = t_after - t_q
    # swaps
    fallback = classify(answer([Edit("ablate", S["bridge"], mid, ALL)]))
    keep, move, hit, hit_n, swaps = [], [], [], 0, []
    for o, exp in S["swaps"].items():
        for rn, rl in {"mid 12-21": mid, "8-23": ls}.items():
            try:
                e = [Edit("swap", [b, o], rl, ALL)]
                got = classify(answer(e))
            except ValueError:
                continue
            swaps.append({"swap": o, "range": rn, "expect": exp, "got": got})
            if exp == S["answer"]:
                keep.append(got == S["answer"])
            else:
                move.append(got != S["answer"])
                if exp != fallback:
                    hit.append(got == exp)
    K = sum(keep) / len(keep) if keep else float("nan")
    F = sum(move) / len(move) if move else float("nan")
    F_hit = sum(hit) / len(hit) if hit else float("nan")

    passes = {"N": N >= TH["N"], "S": Sx >= TH["S"], "U": U >= TH["U"], "D": D >= TH["D"],
              "K": (K >= TH["K"]) if keep else None, "T": T >= TH["T"]}
    if N < 0.5:
        verdict = "BYSTANDER"
    elif all(v for v in passes.values() if v is not None):
        verdict = "CONSULTED BRIDGE"
    elif N >= TH["N"] and (not passes["U"] or not passes["D"] or passes["K"] is False):
        verdict = "ANSWER ECHO"
    else:
        verdict = "UNCLEAR"
    res = dict(sentence=name, group=S.get("group", ""), predict=S.get("predict", ""), question=S["user"], bridge=b, answer=S["answer"], baseline_answer=base_ans,
               P_base=round(base, 3), bridge_rank=b_rank, N=N, S=Sx, O=O, U=U, D=D, T=T, T_after=t_after,
               T_question=t_q, K=K, F=F, F_hit=F_hit, fallback=fallback,
               passed=sum(bool(v) for v in passes.values()), scored=sum(v is not None for v in passes.values()),
               verdict=verdict, auto_controls=json.dumps({c: round(v, 2) for c, v in ctrl.items()}),
               same_kind=json.dumps({c: round(v, 2) for c, v in same.items()}), swaps=json.dumps(swaps))
    say(f"N={N:.2f} S={Sx:.2f} O={O:.2f} U={U:.2f} D={D:.2f} T={T:+.2f} (after {t_after:.2f}, question {t_q:.2f}) "
        f"K={K:.2f} F={F:.2f} F_hit={F_hit:.2f} fallback={fallback} -> {verdict} "
        f"({res['passed']}/{res['scored']})")
    sys.stdout.flush()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", default=None)
    ap.add_argument("--set", default="demo", choices=["demo", "paper"], help="demo = bridge_specs.py; paper = paper_set.py")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th sentence starting at i (parallel runs)")
    ap.add_argument("--config", default="qwen35_4b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    h = load(args.config)
    out = Path(args.out or f"results/bridge_card_{Path(args.config).stem}")
    out.mkdir(parents=True, exist_ok=True)
    specs = SPECS
    if args.set == "paper":
        from consulted_bridges.legacy.paper_set import PICKS as specs
    names = args.spec or list(specs)
    i, n = map(int, args.shard.split("/"))
    names = names[i::n]
    if n > 1:
        out = out / f"shard{i}"
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        try:
            rows.append(card(h, name, specs[name]))
        except Exception as err:  # keep going; report the failure
            print(f"!! {name} failed: {err!r}")
        pd.DataFrame(rows).to_csv(out / "cards.csv", index=False)
    df = pd.DataFrame(rows)
    cols = ["sentence", "group", "predict", "bridge", "answer", "P_base", "bridge_rank", "N", "S", "O", "U", "D", "T", "K", "F",
            "F_hit", "fallback", "passed", "scored", "verdict"]
    print("\n== Scorecards ==")
    print(df[[c for c in cols if c in df]].round(2).to_string(index=False))
    print("wrote", out / "cards.csv")


if __name__ == "__main__":
    main()
