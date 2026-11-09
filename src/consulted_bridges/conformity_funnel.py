"""The gate-screen funnel of the v2 candidates (plan.md §3.1, scope.md §3): per family and lens, how many candidates
were drafted and how many pass each check of the trimmed solo scorecard, in order.

  drafted       candidates in the family (questions/proposed/candidates.py)
  knows it      gate 1: the model answers the two-hop question and both one-hop questions right
  single token  gate 2: the bridge word is one token (every edited word was checked at drafting; this re-checks)
  onset         gate 3: removing the answer's own part kills it in some window, with at least one window before it
  need >= 0.5   bridge drop minus the largest everyday-word drop, in windows before the onset
  reach         of those, the edit removes the bridge's name in the same prompt (drop >= 0.5)
  selected      need >= 0.5 and the bridge comes first (consulted_bridges.conformity_select; reach not required)
  sel + reach   selected and reach passes (the four-check gate of plan.md §3.1, without the same-kind check, which the
                trimmed scorecard does not run)
The confirmation half of the split (candidates.split()) is screened like the rest but reported only as these counts
(scope.md §6); nothing else about it is printed.

  python -m consulted_bridges.conformity_funnel --configs qwen35_4b qwen35_4b_jpp --out results/v2_run1_qwen35_4b
  python -m consulted_bridges.conformity_funnel --configs qwen36_27b qwen36_27b_jpp --active --out results/v2_run2_qwen36_27b
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from consulted_bridges.questions.proposed.candidates import ACTIVE, SPECS, family, split

STEPS = ["drafted", "knows it", "single token", "onset", "need >= 0.5", "reach", "selected", "sel + reach"]


def cards(cfg):
    d = f"results/solo_{cfg}_candidates"
    fs = [f for f in sorted(glob.glob(f"{d}/shard*/cards.csv")) if Path(f).stat().st_size > 1]
    ts = [f for f in sorted(glob.glob(f"{d}/shard*/tests.csv")) if Path(f).stat().st_size > 1]
    if not fs:
        raise SystemExit(f"no cards under {d}")
    c = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).drop_duplicates("sentence", keep="last")
    t = pd.concat([pd.read_csv(f) for f in ts], ignore_index=True) if ts else pd.DataFrame(columns=["sentence", "key", "result"])
    gate = lambda g: set(t[t.key.astype(str).str.endswith(f"|{g}") & (t.result == "pass")].sentence)  # noqa: E731
    c["knows_it"] = c.sentence.isin(gate("g1"))
    c["single_token"] = c.sentence.isin(gate("g2"))
    c["onset_ok"] = c.sentence.isin(gate("g3")) & (c.verdict != "ANSWER FROM THE START")
    c["need_ok"] = c.need.fillna(-1) >= 0.5
    c["reach_ok"] = c.reach.fillna(-1) >= 0.5
    c["selected"] = c.need_ok & (c.comes_first == True)  # noqa: E712  (conformity_select's rule)
    return c.set_index("sentence")


def funnel(cfg, active=False):
    c = cards(cfg)
    conf = set(split()["confirm"])
    rows = []
    qs = ACTIVE if active else SPECS    # --active: leave out the candidates DROPPED at the fact pass
    missing = sorted(set(qs) - set(c.index))
    for q in qs:
        r = c.loc[q] if q in c.index else None
        ok = lambda col: bool(r is not None and r[col])  # noqa: E731
        k, s, o, n = ok("knows_it"), ok("single_token"), ok("onset_ok"), ok("need_ok")
        rows.append({"config": cfg, "family": family(q), "split": "confirm" if q in conf else "dev", "question": q,
                     "drafted": True, "knows it": k, "single token": k and s, "onset": k and s and o,
                     "need >= 0.5": k and s and o and n, "reach": k and s and o and n and ok("reach_ok"),
                     "selected": ok("selected"), "sel + reach": ok("selected") and ok("reach_ok"),
                     "screened": r is not None})
    return pd.DataFrame(rows), missing


def table(df):
    g = df.groupby(["config", "split", "family"])[STEPS].sum().astype(int).reset_index()
    tot = df.groupby(["config", "split"])[STEPS].sum().astype(int).reset_index().assign(family="all")
    return pd.concat([g, tot], ignore_index=True).sort_values(["config", "split", "family"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["qwen35_4b", "qwen35_4b_jpp"])
    ap.add_argument("--out", default="results/v2_run1_qwen35_4b")
    ap.add_argument("--active", action="store_true", help="only candidates.ACTIVE (the DROPPED ones left out)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    for cfg in args.configs:
        df, missing = funnel(cfg, args.active)
        if missing:
            print(f"{cfg}: {len(missing)} candidates have no card (counted as failing knows-it)")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    t = table(df)
    t.to_csv(out / "funnel.csv", index=False)
    # per-question pass flags for the development half only (the confirmation half stays counts-only)
    df[df.split == "dev"].to_csv(out / "funnel_dev_questions.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
