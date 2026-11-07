"""Same-question summaries of a batch run.

score = log10(rank + 1), averaged over the band layers (lower = the word ranks better).
Every table uses only facts that have every condition and role it compares ("complete cases"),
and reports both the mean over facts and the mean with each bridge counted once (facts averaged
within their bridge first), with a 95% bootstrap interval that resamples bridges.

  roles.csv       condition x lens x role: mean score at the last position / best question position
  contrasts.csv   condition x lens x control: control score - bridge score within a fact
                  (> 0: the bridge ranks better than that control). same_country is the key one.
  conditions.csv  pair of conditions x lens x role: score(cond) - score(ref) within a fact
                  (< 0: the word ranks better in cond). answer_written vs filler is the association test.
  probs.csv       condition x candidate: mean probability of the true / peer answer
  ablate.csv      ablated role x candidate: mean drop in log P(candidate), same facts for every role
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = [("answer_written", "filler"), ("answer_written", "q_alone"), ("filler", "q_alone"),
         ("np_pressure", "np_agree"), ("np_hidden", "np_agree"), ("np_hidden", "np_pressure"),
         ("paper_pressure", "paper_agree"), ("paper_hidden", "paper_agree"), ("paper_absent", "paper_agree")]
CONTROLS = ("same_country", "capital", "random")


def boot_ci(values: pd.Series, groups: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """(bridge-once mean, lo, hi): mean of per-bridge means, bootstrap over bridges."""
    per = values.groupby(groups.values).mean().to_numpy()
    if len(per) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(per, (n_boot, len(per))).mean(1)
    return float(per.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def scores(ranks: pd.DataFrame, band: list[int]) -> pd.DataFrame:
    d = ranks[ranks.layer.isin(band)].copy()
    d["last"] = np.log10(d.rank_last + 1)
    d["best_pos"] = np.log10(d.rank_best_pos + 1)
    return d.groupby(["uid", "bridge", "condition", "lens", "role"], as_index=False)[["last", "best_pos"]].mean()


def _complete(s: pd.DataFrame, conds, roles) -> pd.DataFrame:
    s = s[s.condition.isin(conds) & s.role.isin(roles)]
    need = len(set(conds)) * len(set(roles)) * s.lens.nunique()
    n = s.groupby("uid").size()
    return s[s.uid.isin(n[n == need].index)]


def _wide(sub: pd.DataFrame, index: list[str], col: str) -> pd.DataFrame:
    w = sub.pivot_table(index=index, columns=col, values=["last", "best_pos"])
    w.columns = [f"{v}|{c}" for v, c in w.columns]
    return w.reset_index()


def _row(vals: pd.Series, bridges: pd.Series, **keys) -> dict:
    m, lo, hi = boot_ci(vals, bridges)
    return {**keys, "n_facts": int(vals.notna().sum()), "n_bridges": int(bridges.nunique()),
            "mean": float(vals.mean()), "mean_bridge_once": m, "ci_lo": lo, "ci_hi": hi}


def summarize(run_dir: str | Path, band: list[int], conds: list[str] | None = None) -> dict[str, pd.DataFrame]:
    run = Path(run_dir)
    out = {}
    if (run / "ranks.csv").exists():
        s = scores(pd.read_csv(run / "ranks.csv"), band)
        conds = conds or sorted(s.condition.unique())
        rows = []
        for role in s.role.unique():
            sub = _complete(s, conds, ["bridge", role])
            for (cond, lens), g in sub[sub.role == role].groupby(["condition", "lens"]):
                for where in ("last", "best_pos"):
                    rows.append(_row(g[where], g.bridge, condition=cond, lens=lens, role=role, position=where))
        out["roles"] = pd.DataFrame(rows)

        rows = []
        for ctrl in CONTROLS:
            sub = _complete(s, conds, ["bridge", ctrl])
            if sub.empty:
                continue
            w = _wide(sub, ["uid", "bridge", "condition", "lens"], "role")
            for (cond, lens), g in w.groupby(["condition", "lens"]):
                for where in ("last", "best_pos"):
                    diff = g[f"{where}|{ctrl}"] - g[f"{where}|bridge"]
                    rows.append(_row(diff, g["bridge"], condition=cond, lens=lens,
                                     contrast=f"{ctrl} - bridge", position=where))
        out["contrasts"] = pd.DataFrame(rows)

        rows = []
        for a, b in PAIRS:
            if a not in conds or b not in conds:
                continue
            sub = _complete(s, [a, b], ["bridge", "same_country"])
            w = _wide(sub, ["uid", "bridge", "lens", "role"], "condition")
            for (lens, role), g in w.groupby(["lens", "role"]):
                for where in ("last", "best_pos"):
                    rows.append(_row(g[f"{where}|{a}"] - g[f"{where}|{b}"], g["bridge"], pair=f"{a} - {b}", lens=lens,
                                     role=role, position=where))
        out["conditions"] = pd.DataFrame(rows)

    if (run / "probs.csv").exists():
        p = pd.read_csv(run / "probs.csv")
        rows = []
        for (cond, role), g in p.groupby(["condition", "role"]):
            rows.append(_row(g.p_full, g.bridge, condition=cond, candidate=role, measure="p_full"))
            rows.append({"condition": cond, "candidate": role, "measure": "share_top1",
                         "mean": float((g.rank_first == 0).mean()), "n_facts": len(g)})
        out["probs"] = pd.DataFrame(rows)

    if (run / "ablate.csv").exists():
        a = pd.read_csv(run / "ablate.csv")
        n_roles = a.ablated.nunique()
        full = a.groupby(["uid", "candidate"]).ablated.nunique()
        keep = full[full == n_roles].reset_index()[["uid", "candidate"]]
        a = a.merge(keep)
        rows = [_row(g.logp_drop, g.bridge, ablated=r, candidate=c) for (r, c), g in a.groupby(["ablated", "candidate"])]
        out["ablate"] = pd.DataFrame(rows)

    for name, df in out.items():
        df.to_csv(run / f"summary_{name}.csv", index=False)
    return out
