"""Why does our J-lens swap fail where Neuronpedia's works? (NEXT_EXPERIMENTS 1f; docs/swap_replication.md)

Neuronpedia's swap (neuronpedia apps/inference .../endpoints/lens/prompt.py + interp_engine SwapSpec):
    h <- h + (h . s)(t - s)      s, t = unit(J_l^T W_U[token])  (no final-norm gain)
at every position (prompt and generated), at each chosen layer. Ours (consulted_bridges.interventions, kind "swap"):
least-squares coefficients of h on unit(J_l^T (gain * W_U[token])) exchanged, at positions from the question on.

This script scores the FIRST answer token only (one forward pass per setting): the edit is at prompt positions,
and the first answer token is predicted at the last prompt position, so editing generated tokens can't change it.

  part 1  Colosseum grid: swap formula x direction (with/without gain) x positions x layers x swapped-in word
  part 2  the same grid, smaller, on other letter questions
  part 3  removal (ablate) of the bridge / the answer's own part / an everyday word, per window, per position set
  part 4  greedy generations for a few settings, edit applied to generated tokens too (as Neuronpedia's default)

  HF_HUB_OFFLINE=1 uv run python scripts/swap_np.py --config qwen36_27b --out results/swap_np
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import pandas as pd
import torch

from consulted_bridges.scorecard import Card  # noqa: E402
from consulted_bridges.questions.core import SPECS as S3  # noqa: E402
from consulted_bridges.interventions import Edit  # noqa: E402
from consulted_bridges.load import load  # noqa: E402
from consulted_bridges.text import resolve_positions  # noqa: E402
from silent_dissent.lenses import _norm_gain  # noqa: E402
from silent_dissent.model import get_layers, layer_output, replace_layer_output  # noqa: E402

SV = importlib.import_module("consulted_bridges.questions.proposed.similar").SPECS
COUNTRIES = ["Spain", "France", "Germany", "Japan", "Brazil", "Canada", "Egypt", "Mexico"]
CITIES = ["Munich", "Tokyo", "Berlin", "Rome", "Madrid", "Chicago", "Lyon", "Paris"]
EVERYDAY = ["table", "river"]
OTHERS = {"L_gates": S3["L_gates"], "V_LM_eiffel": SV["V_LM_eiffel"], "L_piaf": S3["L_piaf"]}


class Dirs:
    """Unit J-lens directions per (token, layer), with ("gain") or without ("nogain") the final norm's gain."""

    def __init__(self, h):
        self.h, self.lens = h, h.lenses["jlens"]
        self.gain = _norm_gain(self.lens.norm).float()
        self.cache = {}

    def tok(self, w):
        return self.h.encode(w[1:])[0] if w.startswith("@") else self.h.encode(" " + w)[0]

    def __call__(self, w, layer, kind="gain"):
        key = (w, layer, kind)
        if key not in self.cache:
            u = self.lens.unembed.weight[self.tok(w)].float()
            if kind == "gain":
                u = u * self.gain.to(u.device)
            if layer in self.lens.maps:
                u = self.lens.maps[layer][0].float().T @ u
            # "raw": the paper's lens vector, a row of W_U J_l, not normalised (only matters for the exchange)
            self.cache[key] = (u if kind == "raw" else u / u.norm()).to(self.h.device)
        return self.cache[key]


def swap_fn(kind, s, t, k=1.0):
    """H [n, d] -> edited H. s = source (bridge) direction, t = target direction. kind "exch*2" = scale k=2."""
    if "*" in kind:
        kind, k = kind.split("*")[0], float(kind.split("*")[1])
    if kind in ("np", "projx", "ablate"):     # these use unit directions
        t = s if t is None else t
        s, t = s / s.norm(), t / t.norm()
    if kind == "np":            # Neuronpedia: move the source component onto the target
        return lambda H: H + k * (H @ s)[:, None] * (t - s)[None]
    if kind == "exch":          # ours: exchange least-squares coefficients
        A = torch.stack([s, t], 1)

        def f(H):
            C = torch.linalg.lstsq(A, H.T).solution.T
            return H + k * (C.flip(1) - C) @ A.T     # the paper: h + V(sigma(c) - c), alpha = k
        return f
    if kind == "projx":         # exchange the raw projections (exact only if s, t orthogonal)
        return lambda H: H + ((H @ t) - (H @ s))[:, None] * s[None] + ((H @ s) - (H @ t))[:, None] * t[None]
    if kind == "ablate":        # remove the source direction (s only)
        return lambda H: H - (H @ s)[:, None] * s[None]
    raise ValueError(kind)


@contextmanager
def hooked(h, per_layer: dict, positions):
    """per_layer: {layer: fn(H [n,d] float32) -> H}. positions: list of ints, or None = every position (incl. generated)."""
    blocks = get_layers(h.model)
    with ExitStack() as st:
        for layer, fn in per_layer.items():
            def hook(_m, _i, out, fn=fn):
                x = layer_output(out).clone()
                idx = list(range(x.shape[1])) if positions is None else [p for p in positions if p < x.shape[1]]
                if idx:
                    sel = x[:, idx, :].float()
                    B, P, d = sel.shape
                    x[:, idx, :] = fn(sel.reshape(-1, d)).reshape(B, P, d).to(x.dtype)
                return replace_layer_output(out, x)
            st.callback(blocks[layer].register_forward_hook(hook).remove)
        yield


@torch.no_grad()
def first_token(h, ids, letters):
    logits = h.model(input_ids=torch.tensor([ids], device=h.device)).logits[0, -1].float()
    p = logits.softmax(-1)
    top = int(p.argmax())
    return h.tok.decode([top]), float(p[top]), {L: float(p[h.encode(L)[0]]) for L in letters}


@torch.no_grad()
def greedy(h, ids, n=4):
    new = []
    for _ in range(n):
        t = int(h.model(input_ids=torch.tensor([ids + new], device=h.device)).logits[0, -1].argmax())
        new.append(t)
        if t == h.tok.eos_token_id:
            break
    return h.tok.decode(new).replace("<|im_end|>", "⏎")


def pos_sets(h, text, q_start, subject):
    ids = h.encode(text)
    n = len(ids)
    end = max(i for i, t in enumerate(ids) if h.tok.decode([t]) == "<|im_end|>")
    return {"last": [n - 1], "imend+1": [end + 1], "template": list(range(end + 1, n)),
            "subject": resolve_positions(h, text, f"span:{subject}"),
            "q->end": resolve_positions(h, text, f"from:{q_start}"), "all": list(range(n))}


def letter_of(tok_str):
    s = tok_str.strip(" *.\n")
    return s[:1].upper()


def run_grid(h, D, name, S, words, layer_sets, pos_names, swaps, dir_kinds, rows, say):
    c = Card(h, name, S, verbose=False)
    text = c.text
    ids = h.encode(text)
    b = S["bridge"][0]
    subject = {"L_colosseum": "Colosseum", "L_gates": "Bill Gates", "V_LM_eiffel": "Eiffel Tower", "L_piaf": "Piaf"}.get(name, b)
    P = pos_sets(h, text, S["q_start"], subject)
    letters = sorted({b[0].upper()} | {w[0].upper() for w in words})
    t0, tp, pl = first_token(h, ids, letters)
    say(f"\n[{name}] bridge {b!r}; base first token {t0!r} p={tp:.2f}; positions: " +
        "; ".join(f"{k}={v[0]}-{v[-1]}" if len(v) > 1 else f"{k}={v[0]}" for k, v in P.items() if k in pos_names))
    for ln, layers in layer_sets.items():
        layers = [l for l in layers if l < h.n_layers] or [h.n_layers - 1]
        for pn in pos_names:
            for sk in swaps:
                for dk in dir_kinds:
                    got, hits = [], 0
                    for w in words:
                        fns = {l: swap_fn(sk, D(b, l, dk), D(w, l, dk)) for l in layers}
                        with hooked(h, fns, P[pn]):
                            tok, ptop, pl = first_token(h, ids, letters)
                        follow = letter_of(tok) == w[0].upper()
                        hits += follow and w not in EVERYDAY
                        rows.append(dict(question=name, bridge=b, layers=ln, positions=pn, swap=sk, dirs=dk, word=w,
                                         everyday=w in EVERYDAY, top=tok, p_top=round(ptop, 3),
                                         p_target=round(pl[w[0].upper()], 3), p_bridge=round(pl[b[0].upper()], 3),
                                         follows=follow))
                        got.append(f"{w}:{tok.strip()!r}{pl[w[0].upper()]:.2f}")
                    n_alt = sum(w not in EVERYDAY for w in words)
                    say(f"  L{ln:7s} pos {pn:9s} {sk:5s} {dk:6s} {hits}/{n_alt} | " + " ".join(got))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen36_27b")
    ap.add_argument("--out", default="results/swap_np")
    ap.add_argument("--parts", default="0,1,2,3,4")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(*a):
        s = " ".join(map(str, a))
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    parts = {int(x) for x in args.parts.split(",")}
    t_start = time.time()
    h = load(args.config)
    D = Dirs(h)
    L = h.n_layers
    S = S3["L_colosseum"]
    c = Card(h, "L_colosseum", S, verbose=False)
    ids = h.encode(c.text)

    if 0 in parts:  # tokens and direction diagnostics
        say("tokens:", " ".join(f"{i}={h.tok.decode([t])!r}" for i, t in enumerate(ids)))
        g = D.gain
        say(f"final-norm gain: mean {g.mean():.3f} std {g.std():.3f} min {g.min():.3f} max {g.max():.3f} "
            f"top5 {g.abs().topk(5).values.tolist()}")
        store = c.capture()
        diag = []
        for l in [x for x in [9, 18, 27, 32, 42, 48, 54, 60] if x < L]:
            row = dict(layer=l)
            row["cos_gain_nogain_Italy"] = float(D("Italy", l, "gain") @ D("Italy", l, "nogain"))
            for dk in ("gain", "nogain"):
                s, t = D("Italy", l, dk), D("Spain", l, dk)
                row[f"cos_Italy_Spain_{dk}"] = float(s @ t)
                H = store[l][0] if store[l].dim() == 3 else store[l]
                H = H.float()
                for pn, pp in [("q", list(range(3, 27))), ("tmpl", list(range(27, len(ids))))]:
                    hn = H[pp].norm(dim=-1)
                    row[f"hs/|h|_{dk}_{pn}"] = float((H[pp] @ s / hn).mean())
                    row[f"ht/|h|_{dk}_{pn}"] = float((H[pp] @ t / hn).mean())
            diag.append(row)
        dd = pd.DataFrame(diag)
        dd.to_csv(out / "diag.csv", index=False)
        say(dd.round(3).to_string())

    rows = []
    if 1 in parts:
        layer_sets = {"9-27": list(range(9, 28)), "42-54": list(range(42, 55)), "32-54": list(range(32, 55)),
                      "12-41": list(range(12, 42)), "0-63": list(range(0, L))}
        run_grid(h, D, "L_colosseum", S, ["Spain", "France", "Germany", "Japan", "table"], layer_sets,
                 ["imend+1", "template", "q->end", "all"], ["np", "exch", "projx"], ["gain", "nogain"], rows, say)
        pd.DataFrame(rows).to_csv(out / "grid_colosseum.csv", index=False)
        say(f"part 1 done {time.time() - t_start:.0f}s")

    if 2 in parts:
        rows2 = []
        layer_sets = {"9-27": list(range(9, 28)), "32-54": list(range(32, 55)), "42-54": list(range(42, 55))}
        for name, Sq in OTHERS.items():
            b = Sq["bridge"][0]
            pool = COUNTRIES if "country" in Sq["user"] or name.startswith("V_LM") else CITIES
            alts = [w for w in pool if w[0] != b[0] and w != b and len(h.encode(" " + w)) == 1][:4]
            run_grid(h, D, name, Sq, alts + ["table"], layer_sets, ["template", "all"], ["np", "exch"],
                     ["gain", "nogain"], rows2, say)
        pd.DataFrame(rows2).to_csv(out / "grid_others.csv", index=False)
        say(f"part 2 done {time.time() - t_start:.0f}s")

    if 3 in parts:  # removal per window per position set (scorecard-style P over answer forms)
        forms = S["answers"][S["answer"]]
        base = c.P(c.text, forms)
        P = pos_sets(h, c.text, S["q_start"], "Colosseum")
        a_dirs = list(dict.fromkeys(S["answer_dirs"] + ["@" + f for f in forms]))
        rem = []
        for pn in ["last", "imend+1", "template", "subject", "q->end", "all"]:
            pp = P[pn]
            for win in c.wins:
                d = {}
                d["bridge"] = 1 - c.P(c.text, forms, [Edit("ablate", ["Italy"], win, pp)]) / base
                d["answer_own"] = 1 - c.P(c.text, forms, [Edit("ablate_excl", [a, "Italy"], win, pp) for a in a_dirs]) / base
                d["table"] = 1 - c.P(c.text, forms, [Edit("ablate", ["table"], win, pp)]) / base
                with hooked(h, {l: swap_fn("ablate", D("Italy", l, "nogain"), None) for l in win}, pp):
                    d["bridge_nogain"] = 1 - c.P(c.text, forms) / base
                rem.append(dict(positions=pn, window=f"{win[0]}-{win[-1]}", start=win[0],
                                **{k: round(max(0.0, v), 3) for k, v in d.items()}))
            say(f"  removal pos {pn:9s} " + " ".join(f"{r['window']}:{r['bridge']:.2f}/{r['answer_own']:.2f}/{r['table']:.2f}"
                                                     for r in rem if r["positions"] == pn))
        pd.DataFrame(rem).to_csv(out / "removal_colosseum.csv", index=False)
        say(f"part 3 done {time.time() - t_start:.0f}s")

    if 5 in parts:  # the paper's definition (raw W_U J rows, pseudoinverse exchange) and doubled strength
        rows5 = []
        layer_sets = {"9-27": list(range(9, 28)), "42-54": list(range(42, 55)), "32-54": list(range(32, 55)),
                      "12-41": list(range(12, 42))}
        run_grid(h, D, "L_colosseum", S, ["Spain", "France", "Germany", "Japan", "table"], layer_sets,
                 ["template", "q->end", "all"], ["exch", "exch*2", "np*2"], ["raw", "nogain", "gain"], rows5, say)
        for name, Sq in OTHERS.items():
            b = Sq["bridge"][0]
            pool = COUNTRIES if "country" in Sq["user"] or name.startswith("V_LM") else CITIES
            alts = [w for w in pool if w[0] != b[0] and w != b and len(h.encode(" " + w)) == 1][:4]
            run_grid(h, D, name, Sq, alts + ["table"], {k: layer_sets[k] for k in ["9-27", "32-54"]}, ["all"],
                     ["exch", "exch*2", "np*2"], ["raw", "gain"], rows5, say)
        pd.DataFrame(rows5).to_csv(out / "grid_paper.csv", index=False)
        say(f"part 5 done {time.time() - t_start:.0f}s")

    if 6 in parts:  # controls: is it the bridge that's moved, or just the target written in?
        rows6 = []
        P = pos_sets(h, c.text, S["q_start"], "Colosseum")
        P["question"] = [p for p in P["q->end"] if p < P["imend+1"][0]]   # the user's words only, no template
        letters = sorted(set("ISFGJTR"))
        sets = {"9-27": range(9, 28), "42-54": range(42, 55), "32-54": range(32, 55), "32-53": range(32, 54),
                "54": [54], "42-53": range(42, 54), "12-63": range(12, L)}
        for ln, layers in sets.items():
            layers = [l for l in layers if l < L] or [L - 1]
            for pn in ["q->end", "question", "subject", "template"]:
                for src, tgt, sk, dk in [("Italy", "Italy", "ablate", "nogain"), ("Italy", "Spain", "np", "nogain"), ("table", "Spain", "np", "nogain"),
                                         ("river", "Spain", "np", "nogain"), ("Japan", "Spain", "np", "nogain"),
                                         ("Italy", "Spain", "steer", "nogain"), ("Italy", "Spain", "exch", "gain"),
                                         ("Italy", "France", "exch", "gain"), ("Italy", "table", "np", "nogain")]:
                    if sk == "steer":   # add the target at the bridge's strength, remove nothing
                        fns = {l: (lambda s, t: (lambda H: H + (H @ s)[:, None] * t[None]))(D(src, l, dk), D(tgt, l, dk))
                               for l in layers}
                    else:
                        fns = {l: swap_fn(sk, D(src, l, dk), D(tgt, l, dk)) for l in layers}
                    with hooked(h, fns, P[pn]):
                        tok, ptop, pl = first_token(h, ids, letters)
                    rows6.append(dict(layers=ln, positions=pn, source=src, target=tgt, swap=sk, dirs=dk, top=tok,
                                      p_target=round(pl[tgt[0].upper()], 3), p_I=round(pl["I"], 3)))
                say(f"  ctrl L{ln:6s} {pn:7s} " + " ".join(f"{r['source']}>{r['target']}/{r['swap']}:{r['top'].strip()!r}{r['p_target']:.2f}"
                                                     for r in rows6[-9:]))
        pd.DataFrame(rows6).to_csv(out / "controls_colosseum.csv", index=False)
        say(f"part 6 done {time.time() - t_start:.0f}s")

    if 4 in parts:  # greedy generations, edit on every position incl. generated (Neuronpedia default)
        gens = []
        for ln, layers in {"9-27": range(9, 28), "42-54": range(42, 55), "32-54": range(32, 55)}.items():
            for sk, dk in [("np", "nogain"), ("np", "gain"), ("exch", "gain")]:
                for pn, pp in [("all+gen", None), ("q->end", list(range(3, len(ids))))]:
                    fns = {l: swap_fn(sk, D("Italy", l, dk), D("Spain", l, dk)) for l in layers if l < L}
                    with hooked(h, fns, pp):
                        g = greedy(h, ids)
                    gens.append(dict(layers=ln, swap=sk, dirs=dk, positions=pn, output=g))
                    say(f"  gen L{ln} {sk} {dk} {pn}: {g!r}")
        pd.DataFrame(gens).to_csv(out / "gens_colosseum.csv", index=False)
    say(f"ALLDONE {time.time() - t_start:.0f}s")
    json.dump({"seconds": time.time() - t_start}, open(out / "done.json", "w"))


if __name__ == "__main__":
    main()
