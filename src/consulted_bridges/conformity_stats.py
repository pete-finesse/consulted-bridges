"""Conformity results with uncertainty: continuous values, intervals by question, and threshold sensitivity.

The labels in prompts_all.csv ("bridge still needed", "still pulls", "not needed", "mixed") cut continuous
values at thresholds set on the Qwen3.6-27B pilot (conformity.TH). Prompts from one question are not
independent, and 14 of the 16 count questions share two facts. So for each run this reports:

  values.csv       the continuous measure behind each label, per condition and peer count:
                   need = bridge drop - max noise drop (alone, agree), pushback = bridge dLO - max noise dLO
                   (disagree), as median and quartiles, by state (holds / give-in) for disagree
  intervals.csv    the headline shares with 95% intervals from a bootstrap that resamples questions (counts:
                   facts), so prompts from one question move together
  sensitivity.csv  the same shares with every threshold scaled by 0.5, 0.75, 1, 1.5, 2

  prompts_lo.csv   every prompt with the re-analysis columns (docs/conformity/scope.md §7, thresholds fixed first):
                   need_lo (log-odds need), label_lo, ceiling (P(right) after the best bridge removal >= 0.9),
                   inconclusive (a control also moved), clean (reach, present, same-kind quiet, onset near alone)
  summary.csv      old label, log-odds label and clean-subset shares side by side, with assessed / unassessed counts

    python -m consulted_bridges.conformity_stats            # all runs found under results/
Writes results/conformity_stats_v2/ (the earlier results/conformity_stats/ is kept as it was).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from consulted_bridges.conformity import CALIBRATION, DISAGREE_LIKE, SPECS, TH

RUNS = [  # (model, question type, wording, folder, drop the six calibration questions as in the main report)
    ("Qwen3.6-27B", "letters", "neutral", "conformity_qwen36_27b", True),
    ("Qwen3.6-27B", "letters", "deferential", "conformity_defer_qwen36_27b", True),
    ("Qwen3.5-4B", "letters", "neutral", "conformity_qwen35_4b", True),
    ("Qwen3.5-4B", "letters", "deferential", "conformity_defer_qwen35_4b", True),
    ("Qwen3.6-27B", "counts", "neutral", "conformity_counts_qwen36_27b", False),
    ("Qwen3.6-27B", "counts", "deferential", "conformity_counts_defer_qwen36_27b", False),
    ("Llama-3.1-8B", "letters", "neutral", "conformity_llama31_8b_it", False),
    ("Llama-3.1-8B", "letters", "deferential", "conformity_defer_llama31_8b_it", False),
    ("Gemma-4-E4B base lens", "letters", "neutral", "conformity_gemma4_e4b_it", False),
    ("Gemma-4-E4B base lens", "letters", "deferential", "conformity_defer_gemma4_e4b_it", False),
    ("Qwen3.6-27B J++", "letters", "neutral", "conformity_qwen36_27b_jpp", True),
    ("Qwen3.6-27B J++", "letters", "deferential", "conformity_defer_qwen36_27b_jpp", True),
    ("Qwen3.5-4B J++", "letters", "neutral", "conformity_qwen35_4b_jpp", True),
    ("Qwen3.5-4B J++", "letters", "deferential", "conformity_defer_qwen35_4b_jpp", True),
    # GPU run 1 (plan.md Phase C): same lens and questions plus the B1 variants (hidden answers, told to copy...)
    ("Qwen3.5-4B J++ v2", "letters", "neutral", "conformity_v2_qwen35_4b_jpp", True),
    ("Qwen3.5-4B J++ v2", "letters", "deferential", "conformity_v2_defer_qwen35_4b_jpp", True),
]
SCALES = [0.5, 0.75, 1.0, 1.5, 2.0]
B = 2000
LO_TH = dict(need=2.0, none_need=1.0, ceiling=0.9, control_drop=0.2, control_dlo=2.0)   # scope.md §7, fixed 8 Nov
EPS = 1e-6


def logit(x):
    x = np.clip(np.asarray(x, float), EPS, 1 - EPS)
    return np.log(x / (1 - x))


def window_end(label):
    return int(str(label).split("–")[-1])


def reanalyse(folder, p):
    """Add the A1 columns to a run's prompts table from its windows.csv (scope.md §7)."""
    fs = [f for f in sorted(Path("results", folder).glob("shard*/windows.csv")) if f.stat().st_size > 1]
    w = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    w["end"] = w.window.map(window_end)
    key = ["question", "condition", "peers"]
    base = p.set_index(key)[["P_right", "onset"]].rename(columns={"P_right": "P_right_base"})
    w = w.join(base, on=key)
    w = w[w.end < w.onset]                                      # early windows only, as the pipeline
    w["fall"] = logit(w.P_right_base) - logit(w.P_right)
    kind = np.where(w.edit == "bridge", "bridge", np.where(w.edit.str.startswith("noise"), "noise",
                    np.where(w.edit == "answer", "answer", "kind")))
    w["kind"] = kind
    best = w.groupby(key + ["kind"]).fall.max().unstack("kind")
    p = p.join(best.add_prefix("fall_"), on=key)
    p["need_lo"] = p.fall_bridge - p.fall_noise
    reach = p.reach_pass.fillna(False).astype(bool)
    ag = ~p.condition.isin(DISAGREE_LIKE)       # need-type prompts: alone, agree, and the B1 hidden / ignore / bridge ones
    p["label_lo"] = np.where(~ag, p.label.str.replace("bridge ", "", regex=False),
                             np.where(p.need_lo >= LO_TH["need"], "needed",
                                      np.where((p.need_lo < LO_TH["none_need"]) & reach, "not needed", "mixed")))
    p.loc[p.need_lo.isna() & ag, "label_lo"] = "unassessed"
    p["ceiling"] = ag & (p.bridge_P_right_after >= LO_TH["ceiling"])
    p["inconclusive"] = np.where(ag, (p.noise_drop >= LO_TH["control_drop"]) | (p.same_kind_drop >= LO_TH["control_drop"]),
                                 p.noise_dLO >= LO_TH["control_dlo"])
    p.loc[p.inconclusive.fillna(False).astype(bool), "label_lo"] = "inconclusive"
    step = int(np.diff(sorted({int(str(x).split()[1].split("–")[0]) for x in w.window.unique()}))[0])
    alone_onset = p[p.condition == "alone"].set_index("question").onset
    p["clean"] = (reach & p.present.fillna(False).astype(bool) & (p.same_kind_drop < LO_TH["control_drop"])
                  & ((p.onset - p.question.map(alone_onset)).abs() <= step))
    return p


def cluster_of(qid):
    """Bootstrap unit: the question, except count questions that share a fact (Austria 9, Australia 6)."""
    s = SPECS[qid]
    if qid.startswith(("N_", "T_")) and s["answer"] in ("9", "6"):
        return {"9": "fact: Austria 9 states", "6": "fact: Australia 6 states"}[s["answer"]]
    return qid


def relabel(p, k=1.0):
    """The labels of conformity.py with every threshold scaled by k."""
    t = {key: v * k for key, v in TH.items()}
    kind = p.same_kind_dLO.fillna(-np.inf)
    pull = (p.bridge_dLO >= p.noise_dLO + t["pull_noise"]) & (p.bridge_dLO >= kind + t["pull_kind"])
    none_dis = (p.bridge_dLO < p.noise_dLO + t["none_noise"]) & p.reach_pass.fillna(False).astype(bool)
    dis = np.where(pull, "pulls", np.where(none_dis, "not needed", "mixed"))
    need = p.need >= t["need"]
    none_ag = (p.need < t["none_need"]) & p.reach_pass.fillna(False).astype(bool)
    ag = np.where(need, "needed", np.where(none_ag, "not needed", "mixed"))
    return pd.Series(np.where(p.condition.isin(DISAGREE_LIKE), dis, ag), index=p.index)


def shares(p, lab):
    """The headline shares (numerator, denominator) on one run, given labels."""
    a, g, s = p.condition == "alone", p.condition == "agree", p.condition == "disagree"
    gi = s & p.state.astype(str).str.contains("give-in")
    ho = s & (p.state == "holds")
    present = p.present.fillna(False).astype(bool)
    out = {"alone: bridge needed": ((a & (lab == "needed")).sum(), a.sum())}
    for k in (1, 2, 3):
        gk = g & (p.peers == k)
        out[f"agree {k} peer(s): still needed"] = ((gk & (lab == "needed")).sum(), gk.sum())
    out["agree: present but not needed"] = ((g & present & (lab == "not needed")).sum(), g.sum())
    out["holds: bridge pulls"] = ((ho & (lab == "pulls")).sum(), ho.sum())
    out["give-ins: bridge pulls"] = ((gi & (lab == "pulls")).sum(), gi.sum())
    out["give-ins: not needed"] = ((gi & (lab == "not needed")).sum(), gi.sum())
    out["disagree 3 peers: gives in"] = ((s & (p.peers == 3) & p.state.astype(str).str.contains("give-in")).sum(),
                                         (s & (p.peers == 3)).sum())
    return out


def shares_lo(p):
    """The agree headline under the log-odds labels, on all prompts and on the clean subset."""
    out = {}
    for name, d in (("all", p), ("clean", p[p.clean.fillna(False).astype(bool)])):
        g = d.condition == "agree"
        for k in (1, 2, 3):
            gk = g & (d.peers == k)
            out[f"[LO {name}] agree {k} peer(s): still needed"] = ((gk & (d.label_lo == "needed")).sum(), gk.sum())
        out[f"[LO {name}] agree: not needed"] = ((g & (d.label_lo == "not needed")).sum(), g.sum())
        out[f"[LO {name}] agree: inconclusive"] = ((g & (d.label_lo == "inconclusive")).sum(), g.sum())
        out[f"[LO {name}] agree: at ceiling"] = ((g & d.ceiling.fillna(False).astype(bool)).sum(), g.sum())
        a = d.condition == "alone"
        out[f"[LO {name}] alone: bridge needed"] = ((a & (d.label_lo == "needed")).sum(), a.sum())
    return out


def auc(p):
    """P(a needed bridge ranks better than a not-needed one) under the J-lens, alone and agree prompts."""
    q = p[p.condition.isin(["alone", "agree"]) & p.presence_rank.notna()]
    lab = relabel(q)
    pos, neg = q.presence_rank[lab == "needed"].to_numpy(float), q.presence_rank[lab == "not needed"].to_numpy(float)
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] < neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def bootstrap(p, rng):
    """95% intervals for each share and the presence AUC, resampling clusters with replacement."""
    p = p.assign(_c=p.question.map(cluster_of))
    groups = {c: d for c, d in p.groupby("_c")}
    keys = list(groups)
    draws = {}
    for _ in range(B):
        d = pd.concat([groups[c] for c in rng.choice(keys, len(keys))], ignore_index=True)
        sh = {**shares(d, relabel(d)), **shares_lo(d)}
        for name, (x, n) in sh.items():
            draws.setdefault(name, []).append(x / n if n else np.nan)
        draws.setdefault("presence AUC, needed vs not needed", []).append(auc(d))
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if np.isfinite(v).any() else (np.nan, np.nan)
            for k, v in ((k, np.asarray(v, float)) for k, v in draws.items())}, len(keys)


def values(p):
    rows = []
    for (cond, k), d in p.groupby(["condition", "peers"]):
        parts = [(None, d)] if cond != "disagree" else [(st, x) for st, x in d.groupby(d.state.astype(str))]
        for st, x in parts:
            col = "pushback" if cond in ("disagree", "copy") else "need"
            v = x[col].dropna()
            if len(v):
                rows.append(dict(condition=cond, peers=k, state=st, measure=col, n=len(v),
                                 q25=v.quantile(.25), median=v.median(), q75=v.quantile(.75)))
    return pd.DataFrame(rows)


def summary(allp, I):
    """Old label, log-odds label and clean-subset shares side by side, per run, plus assessed / unassessed counts."""
    rows = []
    for (model, qt, wd), p in allp.groupby(["model", "questions", "wording"], sort=False):
        ag = p[p.condition == "agree"]
        base = dict(model=model, questions=qt, wording=wd, questions_n=p.question.nunique(), agree_prompts=len(ag),
                    unassessed=int((ag.label_lo == "unassessed").sum()), clean_agree=int(ag.clean.fillna(False).astype(bool).sum()))
        for k in (1, 2, 3):
            g = ag[ag.peers == k]
            gc = g[g.clean.fillna(False).astype(bool)]
            rows.append(dict(base, peers=k, n=len(g),
                             old_still_needed=int((g.label == "bridge still needed").sum()),
                             old_not_needed=int((g.label == "bridge not needed").sum()),
                             lo_still_needed=int((g.label_lo == "needed").sum()),
                             lo_not_needed=int((g.label_lo == "not needed").sum()),
                             lo_inconclusive=int((g.label_lo == "inconclusive").sum()),
                             at_ceiling=int(g.ceiling.fillna(False).astype(bool).sum()),
                             clean_n=len(gc), clean_still_needed=int((gc.label_lo == "needed").sum()),
                             clean_not_needed=int((gc.label_lo == "not needed").sum()),
                             need_lo_median=g.need_lo.median()))
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(0)
    out = Path("results/conformity_stats_v2")
    out.mkdir(parents=True, exist_ok=True)
    V, I, S, P = [], [], [], []
    for model, qt, wd, folder, drop_cal in RUNS:
        f = Path("results") / folder / "prompts_all.csv"
        if not f.exists():
            print(f"skip {folder}: not found")
            continue
        p = pd.read_csv(f)
        if drop_cal:
            p = p[~p.question.isin(CALIBRATION)]
        p = reanalyse(folder, p)
        tag = dict(model=model, questions=qt, wording=wd)
        P.append(p.assign(**tag))
        V += [dict(tag, **r) for r in values(p).to_dict("records")]
        ci, nclus = bootstrap(p, rng)
        lab = relabel(p)
        assert ((lab == "needed") == (p.label == "bridge still needed"))[p.condition.isin(["alone", "agree"])].all(), folder
        for name, (x, n) in {**shares(p, lab), **shares_lo(p)}.items():
            lo, hi = ci[name]
            I.append(dict(tag, share=name, n=int(n), count=int(x), value=x / n if n else np.nan, lo=lo, hi=hi, clusters=nclus))
        lo, hi = ci["presence AUC, needed vs not needed"]
        I.append(dict(tag, share="presence AUC, needed vs not needed", value=auc(p), lo=lo, hi=hi, clusters=nclus))
        for k in SCALES:
            for name, (x, n) in shares(p, relabel(p, k)).items():
                S.append(dict(tag, share=name, scale=k, value=x / n if n else np.nan))
        print(f"{model} {qt} {wd}: {p.question.nunique()} questions, {nclus} clusters", flush=True)
    pd.DataFrame(V).to_csv(out / "values.csv", index=False)
    allp = pd.concat(P, ignore_index=True)
    allp.to_csv(out / "prompts_lo.csv", index=False)
    summary(allp, pd.DataFrame(I)).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(I).to_csv(out / "intervals.csv", index=False)
    s = pd.DataFrame(S).pivot_table(index=["model", "questions", "wording", "share"], columns="scale", values="value")
    s.reset_index().to_csv(out / "sensitivity.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
