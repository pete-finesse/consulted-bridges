"""Summarise the swap re-run (docs/swap_test_rules.md) and the trend cards.

Reads results/swap2_<cfg>/shard*/ (swaps alone at saved onsets) and results/trend_<cfg>/shard*/ (trimmed cards, with
the swap test where the bridge is needed and first). Writes results/swap2_summary/:
  questions.csv  one row per question and model: quality, and next to it the swap headline per rule
                 (whole pre-onset range: follow share, whether it counts, windows counted)
  rules.csv      per model, group and rule: questions, mean follow share, how many counted, source controls
  trend.csv      the trend cards: verdict (bank rule), quality and its factors

  uv run python scripts/swap2_summary.py
"""
import glob
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
MODELS = {"qwen35_4b": "Qwen3.5-4B", "qwen36_27b": "Qwen3.6-27B"}


def cat(pattern):
    fs = sorted(glob.glob(str(R / pattern)))
    fs = [f for f in fs if Path(f).stat().st_size > 1]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def bank_verdict(r):
    """The question bank's rule (consulted_bridges.bank.verdict)."""
    v = str(r.verdict).upper()
    if v.startswith(("CONSULTED", "CATEGORY")):
        if r.isolation == r.isolation and r.isolation < 0.5:
            return "Answer echo"
        if r.specificity == r.specificity and r.specificity < 0.5:
            return "Category effect"
        return "Consulted bridge"
    return str(r.verdict).capitalize()


def main():
    out = R / "swap2_summary"
    out.mkdir(exist_ok=True)
    qrows, srows, trows = [], [], []
    for cfg, model in MODELS.items():
        jobs = pd.read_csv(R / "swap2_jobs" / f"{cfg}.csv")
        if cat(f"swap2_{cfg}/shard*/cards.csv").empty:
            print(f"{cfg}: no results yet")
            continue
        sw = cat(f"swap2_{cfg}/shard*/cards.csv").merge(jobs[["sentence", "group", "quality"]], on="sentence", how="left")
        tr = cat(f"trend_{cfg}/shard*/cards.csv")
        summ = pd.concat([cat(f"swap2_{cfg}/shard*/swapsum.csv"), cat(f"trend_{cfg}/shard*/swapsum.csv")], ignore_index=True)
        if len(tr):
            tr["bank_verdict"] = tr.apply(bank_verdict, axis=1)
            for _, r in tr.iterrows():
                trows.append(dict(model=model, id=r.sentence, question=r.question, bridge=r.bridge, answer=r.answer,
                                  verdict=r.bank_verdict, reason=r.verdict_reason, quality=r.get("quality"),
                                  need=r.get("need"), specificity=r.get("specificity"), isolation=r.get("isolation"),
                                  answer_onset=r.get("answer_onset")))
            tsw = tr[tr.get("swapA_n", pd.Series(dtype=float)).notna()] if "swapA_n" in tr else tr.iloc[:0]
            sw = pd.concat([sw, tsw.assign(group="trend")], ignore_index=True)
        cols = [c for c in sw.columns if c.startswith("swap")]
        for _, r in sw.iterrows():
            qrows.append(dict(model=model, id=r.sentence, group=r.group, quality=r.quality,
                              answer_onset=r.answer_onset, **{c: r[c] for c in cols}))
        whole = summ[summ.whole == True]  # noqa: E712
        whole = whole.merge(pd.DataFrame(qrows)[lambda d: d.model == model][["id", "group"]],
                            left_on="sentence", right_on="id", how="left")
        for (g, rule), d in whole.groupby(["group", "rule"]):
            srows.append(dict(model=model, group=g, rule=rule, questions=len(d),
                              mean_follow=round(d.follow_share.mean(), 3),
                              counts_yes=int((d.counts == True).sum()), counts_no=int((d.counts == False).sum()),  # noqa: E712
                              not_checked=int(d.counts.isna().sum()),
                              mean_follow_where_counts=round(d[d.counts == True].follow_share.mean(), 3),  # noqa: E712
                              source_everyday_follows=int((d.source_everyday_follows == True).sum()),  # noqa: E712
                              source_kind_follows=int((d.source_kind_follows == True).sum())))  # noqa: E712
    q, s, t = pd.DataFrame(qrows), pd.DataFrame(srows), pd.DataFrame(trows)
    q.to_csv(out / "questions.csv", index=False)
    s.to_csv(out / "rules.csv", index=False)
    t.to_csv(out / "trend.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.to_string(index=False))
    if len(t):
        print(t.groupby(["model", "verdict"]).size().unstack(0).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
