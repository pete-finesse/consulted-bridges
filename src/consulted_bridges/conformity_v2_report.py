"""GPU run 1 tables (plan.md Phase C, scope.md §11 pass marks, fixed before the run): the hidden-answer check, the
dose-response, the block validation, the 2x2 and the patch, from the conformity_v2 and conformity_block runs.

Inputs (one config, both wordings):
  results/conformity_stats_v2/prompts_lo.csv          (python -m consulted_bridges.conformity_stats; model `--model`)
  results/conformity_block[_defer]_<cfg>/shard*/cells.csv, dose.csv, patch.csv
Writes to --out: hidden_check.csv, weakens.csv, dose.csv, validation.csv, twobytwo.csv, twobytwo_counts.csv,
patch.csv, patch_sweep.csv, and prints the headline numbers.

  python -m consulted_bridges.conformity_v2_report --config qwen35_4b_jpp --model "Qwen3.5-4B J++ v2" --out results/v2_run1_qwen35_4b
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from consulted_bridges.conformity import CALIBRATION

T = dict(need=2.0, weakens=2.0, hidden_share=0.75, copy_gap=0.5, copy_min=1.0, rate=0.75, hidden_move=1.0,
         hidden_median=0.5, told_copy=2.0, collapse=2.0, leaned=2.0, sweep_half=0.5, sweep_min=1.0)   # scope.md §11
WORDING = {"neutral": "neutral", "defer": "deferential"}


def load_block(cfg, name):
    out = []
    for frame in ("neutral", "defer"):
        d = f"results/conformity_block_{'' if frame == 'neutral' else 'defer_'}{cfg}"
        fs = [f for f in sorted(glob.glob(f"{d}/shard*/{name}.csv")) if Path(f).stat().st_size > 1]
        out += [pd.read_csv(f) for f in fs]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def hidden_check(p):
    """Need in log-odds by condition and peers, both wordings: label counts, median, and the alone comparison."""
    rows = []
    for (wd, cond, k), d in p.groupby(["wording", "condition", "peers"]):
        if cond not in ("alone", "agree", "hidden", "hidden_sub", "told_ignore", "peer_bridge"):
            continue
        n = len(d)
        rows.append(dict(wording=wd, condition=cond, peers=k, n=n, needed=int((d.label_lo == "needed").sum()),
                         not_needed=int((d.label_lo == "not needed").sum()), mixed=int((d.label_lo == "mixed").sum()),
                         inconclusive=int((d.label_lo == "inconclusive").sum()),
                         unassessed=int((d.label_lo == "unassessed").sum()),
                         need_lo_median=d.need_lo.median(), need_prob_median=d.need.median(),
                         own_drop_median=d.own_drop.median(), ceiling=int(d.ceiling.fillna(False).astype(bool).sum())))
    h = pd.DataFrame(rows)
    alone = h[h.condition == "alone"].set_index("wording")
    h["needed_share"] = h.needed / h.n
    h["alone_needed_share"] = h.wording.map(alone.needed / alone.n)
    hid = h.condition.isin(["hidden", "hidden_sub"])
    h.loc[hid, "tool_works"] = h[hid].needed_share >= T["hidden_share"] * h[hid].alone_needed_share
    return h


def weakens(p):
    """Per question and k: need_lo(hidden k) - need_lo(agree k); weakens if >= 2 (scope.md §7)."""
    rows = []
    for (wd, q), d in p.groupby(["wording", "question"]):
        x = d.set_index(["condition", "peers"]).need_lo
        for k in (1, 2, 3):
            for hc in ("hidden", "hidden_sub"):
                if (hc, k) in x.index and ("agree", k) in x.index:
                    diff = x[(hc, k)] - x[("agree", k)]
                    rows.append(dict(wording=wd, question=q, peers=k, baseline=hc, need_lo_hidden=x[(hc, k)],
                                     need_lo_agree=x[("agree", k)], diff=diff,
                                     weakens=bool(diff >= T["weakens"]) if pd.notna(diff) else None))
    return pd.DataFrame(rows)


def wide(cells):
    key = ["frame", "question", "condition", "peers"]
    w = cells.pivot_table(index=key, columns="cell", values=["lo_right", "lo_peer", "P_right", "P_peer"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    meta = cells.drop_duplicates(key).set_index(key)[["right", "peer_letter", "state_c", "window", "window_source"]]
    return w.join(meta).reset_index()


def validation(w):
    rows = []
    for frame, d in w.groupby("frame"):
        alone_copy = d[d.condition == "copy_alone"].set_index("question").lo_peer_A
        # V1 copy-only: P(peers') under the block back toward the made-up subject's no-peer level
        c = d[d.condition == "copy"].copy()
        c["lift"] = c.lo_peer_A - c.question.map(alone_copy)
        for cell in ("C", "C_late"):
            c["gap"] = (c.lo_peer_A - c[f"lo_peer_{cell}"]) / c.lift
            e = c[c.lift >= T["copy_min"]]
            ok = int((e.gap >= T["copy_gap"]).sum())
            rows.append(dict(frame=frame, row="copy-only: P(peers') back toward the no-peer level", cell=cell,
                             prompts=len(c), eligible=len(e), passing=ok, median=e.gap.median(),
                             passed=bool(len(e) and ok >= T["rate"] * len(e)) if cell == "C" else None))
        # V2 give-ins under disagree: the right letter beats the peers' under the block
        g = d[(d.condition == "disagree") & d.state_c.astype(str).str.contains("give-in")]
        for cell in ("C", "C_late"):
            ok = int((g[f"P_right_{cell}"] > g[f"P_peer_{cell}"]).sum())
            rows.append(dict(frame=frame, row="give-ins under disagree: back to the right letter", cell=cell,
                             prompts=len(g), eligible=len(g), passing=ok,
                             median=(g[f"lo_right_{cell}"] - g.lo_right_A).median(),
                             passed=bool(len(g) and ok >= T["rate"] * len(g)) if cell == "C" else None))
        # V3 hidden prompts: almost no change in P(right)
        hd = d[d.condition.isin(["hidden", "hidden_sub"])]
        for cell in ("C", "C_late"):
            mv = (hd[f"lo_right_{cell}"] - hd.lo_right_A).abs()
            ok = int((mv <= T["hidden_move"]).sum())
            rows.append(dict(frame=frame, row="hidden prompts: almost no change", cell=cell, prompts=len(hd),
                             eligible=len(hd), passing=ok, median=mv.median(),
                             passed=bool(len(hd) and ok >= T["rate"] * len(hd) and mv.median() <= T["hidden_median"])
                             if cell == "C" else None))
        # V4 told to copy: P(peers') drops
        tc = d[d.condition == "told_copy"]
        for cell in ("C", "C_late"):
            dd = tc.lo_peer_A - tc[f"lo_peer_{cell}"]
            ok = int((dd >= T["told_copy"]).sum())
            rows.append(dict(frame=frame, row="told to copy: P(peers') drops", cell=cell, prompts=len(tc),
                             eligible=len(tc), passing=ok, median=dd.median(),
                             passed=bool(len(tc) and ok >= T["rate"] * len(tc)) if cell == "C" else None))
    return pd.DataFrame(rows)


def two_by_two(w):
    d = w[w.condition.isin(["agree", "disagree", "hidden"])].copy()
    d["dB"], d["dC"], d["dD"] = d.lo_right_A - d.lo_right_B, d.lo_right_A - d.lo_right_C, d.lo_right_A - d.lo_right_D
    d["dD_late"] = d.lo_right_A - d.lo_right_D_late
    cb, cd = d.dB >= T["collapse"], d.dD >= T["collapse"]
    d["reading"] = np.select([d.dC >= T["leaned"], ~cb & cd, cb & cd, ~cb & ~cd], ["C well below A (leaned on the peers)",
                             "only D collapses (peers the backup)", "both collapse (bridge needed either way)",
                             "neither collapses"], "only B collapses")
    d.loc[d.dB.isna() | d.dD.isna(), "reading"] = "no window"
    d["D_to_peers"] = d.P_peer_D > d.P_right_D
    counts = (d.groupby(["frame", "condition", "peers", "reading"]).size().rename("n").reset_index())
    return d, counts


def patch(pt, w):
    if pt.empty:
        return pt, pt
    a = w[(w.condition == "agree") & (w.peers == 3)].set_index(["frame", "question"])
    allp = pt[pt.cell.isin(["P_all", "PB_all"])].pivot_table(index=["frame", "question"], columns="cell", values="lo_right")
    allp = allp.join(a[["lo_right_A", "lo_right_B", "lo_right_C", "lo_right_D"]])
    allp = allp.join(pt.drop_duplicates(["frame", "question"]).set_index(["frame", "question"]).patch_source)
    sw = pt[pt.cell == "P_from_L"].join(a.lo_right_A, on=["frame", "question"]).join(allp.P_all, on=["frame", "question"])
    sw["effect"] = sw.lo_right_A - sw.lo_right
    rows = []
    for (frame, q), x in sw.groupby(["frame", "question"]):
        full = x.lo_right_A.iloc[0] - x.P_all.iloc[0]
        if abs(full) < T["sweep_min"]:
            rows.append(dict(frame=frame, question=q, full_effect=full, stops_at=None))
            continue
        x = x.sort_values("from_layer")
        weak = x[(x.effect / full) < T["sweep_half"]]
        rows.append(dict(frame=frame, question=q, full_effect=full,
                         stops_at=int(weak.from_layer.iloc[0]) if len(weak) else None))
    return allp.reset_index(), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen35_4b_jpp")
    ap.add_argument("--model", default="Qwen3.5-4B J++ v2")
    ap.add_argument("--out", default="results/v2_run1_qwen35_4b")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pl = pd.read_csv("results/conformity_stats_v2/prompts_lo.csv", low_memory=False)
    p = pl[pl.model == args.model]
    h = hidden_check(p)
    h.to_csv(out / "hidden_check.csv", index=False)
    wk = weakens(p)
    wk.to_csv(out / "weakens.csv", index=False)
    print(h.to_string(index=False))
    if len(wk):
        print(wk.groupby(["wording", "baseline", "peers"]).weakens.agg(["sum", "count"]).to_string())
    cells = load_block(args.config, "cells")
    if cells.empty:
        print("no block cells yet")
        return
    w = wide(cells)
    v = validation(w)
    v.to_csv(out / "validation.csv", index=False)
    print(v.to_string(index=False))
    d, counts = two_by_two(w)
    d["calibration"] = d.question.isin(CALIBRATION)
    counts = (d[~d.calibration].groupby(["frame", "condition", "peers", "reading"]).size().rename("n").reset_index())
    d.to_csv(out / "twobytwo.csv", index=False)
    counts.to_csv(out / "twobytwo_counts.csv", index=False)
    print(counts.to_string(index=False))
    dose = load_block(args.config, "dose")
    if len(dose):
        a = w.set_index(["frame", "question", "condition", "peers"])
        dose = dose.join(a[["lo_right_A", "lo_right_B"]], on=["frame", "question", "condition", "peers"])
        dose["fall"] = dose.lo_right_A - dose.lo_right
        full = a.reset_index()[a.reset_index().condition.isin(["alone", "hidden", "agree"])]
        full = full[(full.condition == "alone") | (full.peers == 3)].assign(strength=1.0, fall=lambda x: x.lo_right_A - x.lo_right_B)
        dd = pd.concat([dose[["frame", "question", "condition", "peers", "strength", "fall"]],
                        full[["frame", "question", "condition", "peers", "strength", "fall"]]])
        dd = dd[~dd.question.isin(CALIBRATION)]
        ds = dd.groupby(["frame", "condition", "strength"]).fall.median().unstack("strength")
        ds.to_csv(out / "dose.csv")
        print(ds.round(2).to_string())
    pt = load_block(args.config, "patch")
    if len(pt):
        allp, sweep = patch(pt[~pt.question.isin(CALIBRATION)], w)
        allp.to_csv(out / "patch.csv", index=False)
        sweep.to_csv(out / "patch_sweep.csv", index=False)
        print(allp.round(2).to_string(index=False))
        print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
