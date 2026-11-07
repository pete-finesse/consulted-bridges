"""Batch runs over many TwoHopFact facts: readouts per condition, and ablations.

Outputs (tidy CSVs in `out_dir`):
  items.csv    one row per fact: its roles' words (bridge, controls, answers) and their pieces
  ranks.csv    uid x condition x lens x layer x role: rank at the last position, and the best
               rank over the question's positions (with that position's token)
  probs.csv    uid x condition x candidate: next-token probability of the true / peer answer
  ablate.csv   uid x ablated role x candidate: answer probability after ablating that role's word
  run.json     the settings used
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from silent_dissent.entity_facts import norm

from . import conditions as C
from . import facts as F
from ..interventions import Edit, answer_probs
from ..readout import readout


def char_to_pos(h, text: str, char: int) -> int:
    from ..text import offsets

    offs = offsets(h, text)
    return next((i for i, (a, b) in enumerate(offs) if b > char), len(offs) - 1)


def leaky(fact) -> bool:
    """Is a content word of the bridge or the answer written in the subject (e.g. "City University
    of New York" -> New York City)? Then the question states the bridge."""
    import re

    from silent_dissent.entity_facts import STOPWORDS

    words = lambda s: {w for w in re.findall(r"[a-z0-9]+", norm(s)) if w not in STOPWORDS and len(w) >= 3}
    return bool((words(fact.e2) | words(fact.e3)) & words(fact.e1))


def prepare(h, facts: list, pool: list, seed: int = 0, prefer: str = "random", keep_leaky: bool = False,
            verbose: bool = False) -> list[dict]:
    """Roles (words) per fact, from the whole pool (so controls come from the full dataset).
    Dropped: facts whose bridge shares a first piece with the answer (Luxembourg, Luxembourg), and
    unless keep_leaky, facts whose subject contains the bridge or answer."""
    from silent_dissent.entity_facts import probe_tokens

    by_country = F.cities_by_country(pool)
    out, n_clash, n_leak = [], 0, 0
    for f in facts:
        if set(probe_tokens(h.tok, f.e2)) & set(probe_tokens(h.tok, f.e3)):
            n_clash += 1
            continue
        if not keep_leaky and leaky(f):
            n_leak += 1
            continue
        out.append({"fact": f, **F.pick_controls(h.tok, f, by_country, pool, seed, prefer)})
    if verbose:
        print(f"{len(out)} facts kept; dropped {n_clash} (bridge and answer share a piece), "
              f"{n_leak} (subject contains the bridge or answer)")
    return out


def items_table(h, items: list[dict]) -> pd.DataFrame:
    from ..text import piece_strings, word_tokens

    rows = []
    for it in items:
        f = it["fact"]
        words = [it[r] for r in F.ROLES if it[r]]
        toks = word_tokens(h, words)
        row = {"uid": f.uid, "category": f.category, "e1": f.e1, "composed": f.composed}
        for r in F.ROLES:
            row[r] = it[r]
            row[f"{r}_pieces"] = "|".join(piece_strings(h, toks[it[r]])) if it[r] else None
        rows.append(row)
    return pd.DataFrame(rows)


def run_readouts(h, items: list[dict], conds: list[str], out_dir: str | Path, paper_fmt: str = "plain",
                 lenses: list[str] | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items_table(h, items).to_csv(out / "items.csv", index=False)
    rank_rows, prob_rows = [], []
    for it in tqdm(items, desc="facts"):
        f = it["fact"]
        roles = [r for r in F.ROLES if it[r]]
        words = [it[r] for r in roles]
        for cond in conds:
            try:
                text, start = C.build(h, f, cond, it["peer_answer"], it["peer_fact"], paper_fmt)
            except ValueError:
                continue
            n = len(h.encode(text))
            p0 = min(char_to_pos(h, text, start), n - 1)
            with warnings.catch_warnings():      # pieces shared between roles: that role gets no rank
                warnings.simplefilter("ignore")
                ro = readout(h, text, words, lenses, positions=list(range(p0, n)))
            R = ro.ranks                                    # [lens, P, layer, word]
            big = np.where(R < 0, np.iinfo(np.int64).max, R)
            best_p = big.argmin(1)                          # [lens, layer, word]
            for li, lens in enumerate(ro.lenses):
                for layer in range(R.shape[2]):
                    for k, (role, w) in enumerate(zip(roles, words)):
                        if R[li, -1, layer, k] < 0:
                            continue
                        bp = ro.positions[best_p[li, layer, k]]
                        rank_rows.append({"uid": f.uid, "bridge": f.e2, "category": f.category, "condition": cond,
                                          "lens": lens, "layer": layer, "role": role, "word": w,
                                          "rank_last": int(R[li, -1, layer, k]),
                                          "rank_best_pos": int(R[li, best_p[li, layer, k], layer, k]),
                                          "best_pos_from_end": bp - n, "best_pos_token": ro.tokens[bp]})
            cands = [c for c in (it["answer"], it["peer_answer"]) if c]
            pr = answer_probs(h, text, cands, top=1)
            for role, c in zip(("answer", "peer_answer"), cands):
                prob_rows.append({"uid": f.uid, "bridge": f.e2, "condition": cond, "role": role, "word": c,
                                  **pr["cands"][c], "top1": pr["top"][0][0]})
    pd.DataFrame(rank_rows).to_csv(out / "ranks.csv", index=False)
    pd.DataFrame(prob_rows).to_csv(out / "probs.csv", index=False)


def run_ablations(h, items: list[dict], cond: str, layers: list[int], out_dir: str | Path,
                  roles=("bridge", "same_country", "capital", "random"), where: str = "question",
                  paper_fmt: str = "plain", lens: str = "jlens") -> pd.DataFrame:
    """For each fact: P(answer), P(peer answer) with nothing ablated, then with each role's word
    ablated at `layers` and at the last position (`where="last"`) or every question position."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in tqdm(items, desc="ablate"):
        f = it["fact"]
        try:
            text, start = C.build(h, f, cond, it["peer_answer"], it["peer_fact"], paper_fmt)
        except ValueError:
            continue
        n = len(h.encode(text))
        pos = "last" if where == "last" else list(range(min(char_to_pos(h, text, start), n - 1), n))
        cands = [c for c in (it["answer"], it["peer_answer"]) if c]
        runs = {"none": []}
        runs.update({r: [Edit("ablate", [it[r]], layers, pos, lens=lens)] for r in roles if it[r]})
        base = None
        for r, edits in runs.items():
            pr = answer_probs(h, text, cands, edits, top=1)
            base = base or pr
            for role, c in zip(("answer", "peer_answer"), cands):
                v = pr["cands"][c]
                rows.append({"uid": f.uid, "bridge": f.e2, "condition": cond, "ablated": r,
                             "ablated_word": it[r] if r != "none" else "", "candidate": role, "word": c, **v,
                             "logp_drop": float(np.log(base["cands"][c]["p_full"] + 1e-30) - np.log(v["p_full"] + 1e-30)),
                             "top1": pr["top"][0][0]})
    df = pd.DataFrame(rows)
    df.to_csv(out / "ablate.csv", index=False)
    (out / "ablate.json").write_text(json.dumps({"condition": cond, "layers": layers, "where": where,
                                                 "lens": lens, "roles": list(roles)}, indent=1))
    return df
