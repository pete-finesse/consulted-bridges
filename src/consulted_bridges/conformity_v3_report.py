"""GPU run 3 tables (Qwen3.6-27B confirmation, results/v2_run3_qwen36_27b/frozen.md), read by the rules fixed in
scope.md §11 before the run ("Run 3 readings").

Inputs (both wordings, one folder per question):
  results/conformity_v3[_defer]_qwen36_27b_jpp/<q>/shard0/{prompts,windows}.csv
  results/conformity_block_v3[_defer]_qwen36_27b_jpp/<q>/shard0/{cells,dose,patch}.csv
  results/conformity_swap_v3[_defer]_qwen36_27b_jpp/<q>/shard0/swap.csv
  run 2 (development) for the replication: results/conformity_stats_v2/prompts_lo.csv (model "Qwen3.6-27B J++ v2")
  and results/v2_run2_qwen36_27b/twobytwo.csv
Writes to results/v2_run3_qwen36_27b/: prompts_lo.csv, labels.csv, hidden_check.csv, hidden_check_v3.csv, weakens.csv,
validation.csv, twobytwo.csv, twobytwo_counts.csv, destination.csv, dose.csv, patch.csv, patch_sweep.csv, swap.csv,
swap_rates.csv, replication.csv; prints the headline tables.

  python -m consulted_bridges.conformity_v3_report
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from consulted_bridges.conformity_stats import reanalyse
from consulted_bridges.conformity_v2_report import T, hidden_check, patch, two_by_two, validation, weakens, wide

CFG = "qwen36_27b_jpp"
OUT = Path("results/v2_run3_qwen36_27b")
WORDING = {"neutral": "neutral", "defer": "deferential"}
FRAME = {v: k for k, v in WORDING.items()}
B, SEED = 2000, 0
NOT_NEEDED_MAX, B_COLLAPSE_MIN, SWAP_THROUGH_MAX = 0.1, 0.75, 0.1     # scope.md §11, run 3 readings


def sfx(frame):
    return "" if frame == "neutral" else "_defer"


def load_prompts():
    P = []
    for frame in ("neutral", "defer"):
        for d in sorted(glob.glob(f"results/conformity_v3{sfx(frame)}_{CFG}/*/shard0/prompts.csv")):
            if Path(d).stat().st_size <= 1:
                continue
            folder = str(Path(d).parent.parent.relative_to("results"))
            p = reanalyse(folder, pd.read_csv(d))
            P.append(p.assign(wording=WORDING[frame]))
    return pd.concat(P, ignore_index=True)


def load_tab(kind, name):
    out = []
    for frame in ("neutral", "defer"):
        for f in sorted(glob.glob(f"results/conformity_{kind}_v3{sfx(frame)}_{CFG}/*/shard0/{name}.csv")):
            if Path(f).stat().st_size > 1:
                out.append(pd.read_csv(f))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def labels_table(p):
    """Old labels (probability) against log-odds labels, per wording, condition, peers."""
    rows = []
    for (wd, cond, k), d in p.groupby(["wording", "condition", "peers"]):
        rows.append(dict(wording=wd, condition=cond, peers=k, n=len(d),
                         old_needed=int((d.label == "bridge still needed").sum()),
                         old_not_needed=int((d.label == "bridge not needed").sum()),
                         old_pulls=int((d.label == "bridge still pulls").sum()),
                         lo_needed=int((d.label_lo == "needed").sum()), lo_not_needed=int((d.label_lo == "not needed").sum()),
                         lo_mixed=int((d.label_lo == "mixed").sum()), lo_pulls=int((d.label_lo == "still pulls").sum()),
                         lo_inconclusive=int((d.label_lo == "inconclusive").sum()),
                         unassessed=int((d.label_lo == "unassessed").sum() + (d.label == "no early windows").sum()),
                         ceiling=int(d.ceiling.fillna(False).astype(bool).sum()), need_lo_median=d.need_lo.median(),
                         reach_median=d.reach_drop.median(), reach_qend_median=d.reach_drop_qend.median(),
                         reach_pass=int(d.reach_pass.fillna(False).astype(bool).sum()),
                         full_give_in=int((d.state == "full give-in").sum()),
                         partial_give_in=int((d.state == "partial give-in").sum()), P_right_median=d.P_right.median()))
    return pd.DataFrame(rows)


def hidden_check_v3(p, tb):
    """scope.md §11 run-3 reading: on assessed prompts, 'not needed' share and B's collapse share per hidden condition."""
    rows = []
    for (wd, cond, k), d in p[p.condition.isin(["hidden", "hidden_sub"])].groupby(["wording", "condition", "peers"]):
        no_win = int((d.label == "no early windows").sum() + (d.label_lo == "unassessed").sum())
        inc = int((d.label_lo == "inconclusive").sum())
        assessed = len(d) - no_win
        clean = assessed - inc
        nn = int((d.label_lo == "not needed").sum())
        x = tb[(tb.frame == FRAME[wd]) & (tb.condition == cond) & (tb.peers == k)] if len(tb) else pd.DataFrame()
        own = x[x.window_source == "own"] if len(x) else x
        bc = int(((own.lo_right_A - own.lo_right_B) >= T["collapse"]).sum()) if len(own) else None
        rows.append(dict(wording=wd, condition=cond, peers=k, prompts=len(d), no_early_window=no_win, assessed=assessed,
                         inconclusive=inc, assessed_clean=clean, not_needed=nn,
                         not_needed_share=nn / clean if clean else np.nan,
                         needed=int((d.label_lo == "needed").sum()),
                         reach_median_of_nulls=d[d.label_lo.isin(["not needed", "mixed"])].reach_drop.median(),
                         block_prompts=len(own) if len(x) else None, B_collapses=bc,
                         B_collapse_share=(bc / len(own)) if len(x) and len(own) else np.nan))
    h = pd.DataFrame(rows)
    h["fails_not_needed"] = h.not_needed_share > NOT_NEEDED_MAX
    h["fails_B"] = h.B_collapse_share < B_COLLAPSE_MIN
    return h


def validation3(w):
    """conformity_v2_report.validation without the copy-only row: the confirmation set has no made-up-subject prompts
    (they exist only for the calibration questions); the row is reported as "no cases"."""
    has_copy = (w.condition == "copy").any()
    if has_copy:
        return validation(w)
    dummy = [dict(frame=f, question="_none", condition=c, peers=k, lo_peer_A=0.0, lo_peer_C=0.0, lo_peer_C_late=0.0)
             for f in w.frame.unique() for c, k in (("copy_alone", 0), ("copy", 3))]
    w2 = pd.concat([w, pd.DataFrame(dummy)], ignore_index=True)
    w2["question"] = w2.question.astype(object)
    for c in w2.columns:
        if c.startswith(("lo_", "P_")):
            w2[c] = pd.to_numeric(w2[c], errors="coerce")
    v = validation(w2)
    v.loc[v.row.str.startswith("copy-only"), ["passed", "prompts", "eligible", "passing", "median"]] = None
    return v


def wide3(cells):
    w = wide(cells)
    key = ["frame", "question", "condition", "peers"]
    pk = cells.pivot_table(index=key, columns="cell", values="lo_kind", aggfunc="first")
    pk.columns = [f"lo_kind_{c}" for c in pk.columns]
    return w.join(pk, on=key).join(cells.drop_duplicates(key).set_index(key)[["kind_word", "kind_answer"]], on=key)


def destination(d):
    """Disagree prompts: where D sends the answer (right, peers', same-kind letter), counted per wording and peers."""
    x = d[d.condition == "disagree"].copy()
    lr, lp, lk = x.lo_right_D, x.lo_peer_D, x.lo_kind_D
    x["D_top"] = np.select([(lr >= lp) & (lr >= lk), (lp > lr) & (lp >= lk)], ["right", "peers'"], "same-kind")
    x["D_dropped"] = x.dD >= T["collapse"]
    return x.groupby(["frame", "peers", "D_dropped", "D_top"]).size().rename("n").reset_index(), x


def swap_rates(s):
    """Per wording, condition and cell: follow share (M, CM) and everyday get-through share (E, CE); §11 rule."""
    rows = []
    for (frame, cond, fmt), d in s.groupby(["frame", "condition", "format"]):
        r = dict(frame=frame, condition=cond, format=fmt, questions=d.question.nunique())
        for vis, pre in (("visible", ""), ("blocked", "C")):
            m, e = d[d.cell == pre + "M"], d[d.cell == pre + "E"]
            if not len(m):
                continue
            r[f"{vis}_follow"] = int(m.follows.astype(bool).sum())
            r[f"{vis}_n"] = len(m)
            r[f"{vis}_said_right"] = int((m.answered == m.right).sum())
            r[f"{vis}_said_peers"] = int((m.answered.astype(str) == m.peer_letter.astype(str)).sum()) if cond != "alone" else None
            r[f"{vis}_everyday_through"] = int(e.through.fillna(False).astype(bool).sum()) if len(e) else None
            r[f"{vis}_everyday_n"] = len(e)
            r[f"{vis}_control_ok"] = (bool(e.through.fillna(False).astype(bool).sum() <= SWAP_THROUGH_MAX * len(e))
                                      if len(e) else None)
            rr = d[d.cell == pre + "R"]
            r[f"{vis}_removal_said_right"] = int((rr.answered == rr.right).sum())
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- replication: shares with question bootstrap -----
def share_table(p, tb):
    """{name: (count, n)} per wording on one run's prompts (label_lo) and 2x2 rows (reading)."""
    out = {}
    for wd in ("neutral", "deferential"):
        q = p[p.wording == wd]
        a = q[q.condition == "alone"]
        out[(wd, "LO alone: needed")] = ((a.label_lo == "needed").sum(), len(a))
        for k in (1, 2, 3):
            g = q[(q.condition == "agree") & (q.peers == k)]
            out[(wd, f"LO agree {k}: needed")] = ((g.label_lo == "needed").sum(), len(g))
            out[(wd, f"LO agree {k}: not needed")] = ((g.label_lo == "not needed").sum(), len(g))
        d3 = q[(q.condition == "disagree") & (q.peers == 3)]
        out[(wd, "disagree 3: full give-in")] = ((d3.state == "full give-in").sum(), len(d3))
        x = q.set_index(["question", "condition", "peers"]).need_lo
        wk = [x.get((qq, "hidden", 3)) - x.get((qq, "agree", 3)) for qq in q.question.unique()
              if (qq, "hidden", 3) in x.index and (qq, "agree", 3) in x.index]
        wk = [v for v in wk if pd.notna(v)]
        out[(wd, "weakens at 3 peers (vs '?')")] = (sum(v >= T["weakens"] for v in wk), len(wk))
        hid = q[q.condition.isin(["hidden", "hidden_sub"])]
        hid = hid[~hid.label_lo.isin(["unassessed", "inconclusive"]) & (hid.label != "no early windows")]
        out[(wd, "hidden: not needed (assessed)")] = ((hid.label_lo == "not needed").sum(), len(hid))
        t = tb[tb.frame == FRAME[wd]]
        for cond, k in (("agree", 1), ("agree", 2), ("agree", 3), ("disagree", 3), ("hidden", 3)):
            z = t[(t.condition == cond) & (t.peers == k)]
            for rd in ("both collapse (bridge needed either way)", "only D collapses (peers the backup)",
                       "C well below A (leaned on the peers)", "neither collapses", "only B collapses"):
                out[(wd, f"2x2 {cond} {k}: {rd.split(' (')[0]}")] = ((z.reading == rd).sum(), len(z))
        hz = t[t.condition == "hidden"]
        hz = hz[hz.window_source == "own"] if "window_source" in hz else hz
        out[(wd, "2x2 hidden 1-3: B collapses")] = ((hz.dB >= T["collapse"]).sum(), len(hz))
    return out


def bootstrap(p, tb, rng):
    qs = sorted(set(p.question))
    gp = {q: d for q, d in p.groupby("question")}
    gt = {q: d for q, d in tb.groupby("question")}
    draws = {}
    for _ in range(B):
        pick = rng.choice(qs, len(qs))
        dp = pd.concat([gp[q].assign(question=f"{q}#{i}") for i, q in enumerate(pick)], ignore_index=True)
        dt = pd.concat([gt[q].assign(question=f"{q}#{i}") for i, q in enumerate(pick) if q in gt], ignore_index=True)
        for key, (x, n) in share_table(dp, dt).items():
            draws.setdefault(key, []).append(x / n if n else np.nan)
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in draws.items()}


def replication(p3, tb3):
    rng = np.random.default_rng(SEED)
    pl = pd.read_csv("results/conformity_stats_v2/prompts_lo.csv", low_memory=False)
    p2 = pl[pl.model == "Qwen3.6-27B J++ v2"]
    tb2 = pd.read_csv("results/v2_run2_qwen36_27b/twobytwo.csv")
    tb2 = tb2[~tb2.calibration]
    s2, s3 = share_table(p2, tb2), share_table(p3, tb3)
    c2, c3 = bootstrap(p2, tb2, rng), bootstrap(p3, tb3, rng)
    rows = []
    for key in s2:
        (x2, n2), (x3, n3) = s2[key], s3.get(key, (0, 0))
        v2, v3 = (x2 / n2 if n2 else np.nan), (x3 / n3 if n3 else np.nan)
        lo2, hi2 = c2[key]
        lo3, hi3 = c3.get(key, (np.nan, np.nan))
        rows.append(dict(wording=key[0], share=key[1], dev_count=int(x2), dev_n=int(n2), dev=v2, dev_lo=lo2, dev_hi=hi2,
                         conf_count=int(x3), conf_n=int(n3), conf=v3, conf_lo=lo3, conf_hi=hi3,
                         replicates=bool(lo2 <= v3 <= hi2) if n3 and np.isfinite(v3) else None,
                         intervals_overlap=bool(lo3 <= hi2 and lo2 <= hi3) if n3 else None))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    p = load_prompts()
    p.to_csv(OUT / "prompts_lo.csv", index=False)
    print(f"prompts: {len(p)} on {p.question.nunique()} questions; failed rows: none recorded" if len(p) else "no prompts")
    lab = labels_table(p)
    lab.to_csv(OUT / "labels.csv", index=False)
    print(lab[lab.condition.isin(["alone", "agree", "disagree", "hidden", "hidden_sub", "told_copy"])].to_string(index=False))
    h = hidden_check(p)
    h.to_csv(OUT / "hidden_check.csv", index=False)
    wk = weakens(p)
    wk.to_csv(OUT / "weakens.csv", index=False)
    if len(wk):
        print(wk.groupby(["wording", "baseline", "peers"]).weakens.agg(["sum", "count"]).to_string())
    cells = load_tab("block", "cells")
    tb = pd.DataFrame()
    if len(cells):
        w = wide3(cells)
        v = validation3(w)
        v.to_csv(OUT / "validation.csv", index=False)
        print(v.to_string(index=False))
        tb, counts = two_by_two(w)
        tb.to_csv(OUT / "twobytwo.csv", index=False)
        counts.to_csv(OUT / "twobytwo_counts.csv", index=False)
        print(counts.pivot_table(index=["frame", "condition", "peers"], columns="reading", values="n", fill_value=0).to_string())
        dst, _ = destination(tb)
        dst.to_csv(OUT / "destination.csv", index=False)
        print(dst.to_string(index=False))
        dose = load_tab("block", "dose")
        if len(dose):
            a = w.set_index(["frame", "question", "condition", "peers"])
            dose = dose.join(a[["lo_right_A", "lo_right_B"]], on=["frame", "question", "condition", "peers"])
            dose["fall"] = dose.lo_right_A - dose.lo_right
            full = a.reset_index()
            full = full[(full.condition == "alone") | ((full.condition.isin(["hidden", "agree"])) & (full.peers == 3))]
            full = full.assign(strength=1.0, fall=full.lo_right_A - full.lo_right_B)
            dd = pd.concat([dose[["frame", "question", "condition", "peers", "strength", "fall"]],
                            full[["frame", "question", "condition", "peers", "strength", "fall"]]])
            ds = dd.groupby(["frame", "condition", "strength"]).fall.median().unstack("strength")
            ds.to_csv(OUT / "dose.csv")
            print(ds.round(2).to_string())
        pt = load_tab("block", "patch")
        if len(pt):
            allp, sweep = patch(pt, w)
            allp.to_csv(OUT / "patch.csv", index=False)
            sweep.to_csv(OUT / "patch_sweep.csv", index=False)
            print("patch: corr(P_all, C) by frame:",
                  {f: round(x[["P_all", "lo_right_C"]].corr().iloc[0, 1], 2) for f, x in allp.groupby("frame")})
    h3 = hidden_check_v3(p, w if len(cells) else pd.DataFrame())
    h3.to_csv(OUT / "hidden_check_v3.csv", index=False)
    print(h3.to_string(index=False))
    s = load_tab("swap", "swap")
    if len(s):
        s.to_csv(OUT / "swap.csv", index=False)
        sr = swap_rates(s)
        sr.to_csv(OUT / "swap_rates.csv", index=False)
        print(sr.to_string(index=False))
    if len(tb):
        rep = replication(p, tb)
        rep.to_csv(OUT / "replication.csv", index=False)
        print(rep.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
