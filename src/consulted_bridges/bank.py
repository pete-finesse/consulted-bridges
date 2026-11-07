"""Build the question bank's tables from the latest scorecard results of each model.

  Qwen3.5-4B   v4 (results/card4_qwen35_4b): own-part answer onset, raw answer tokens, two-window onset
  Qwen3.6-27B  results/big_qwen36_27b + big_raw_qwen36_27b, with every question whose onset moved under v4
               replaced by its v4 re-score (results/v4_27b)

Verdicts use one rule for both models (quality's own factors): needed early beats noise and comes first ->
Answer echo if isolation < 0.5, else Category effect if specificity < 0.5, else Consulted bridge.

  uv run python scripts/bank_build.py OUTDIR
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[2] / "results"
FAM = {"L": "Letter of the hidden word", "R": "Leg-count riddle", "O": "Parts of an object", "X": "Reference (round 2)",
       "Q": "Sequences and arithmetic", "W": "Wordplay", "S": "Science and nature", "C": "Culture and games"}
SYM = {"pass": "✓", "fail": "✗"}


def load(globs):
    fs = [f for g in globs for f in sorted(glob.glob(str(R / g)))]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def model_tables(cards_globs, tests_globs, model, strip_prefix=False, keep=None):
    c, t = load(cards_globs), load(tests_globs)
    if strip_prefix:
        for d in (c, t):
            d["sentence"] = d.sentence.str.replace(r"^(v3_|i_)", "", regex=True)
        c, t = c[~c.sentence.str.startswith("v2_")], t[~t.sentence.str.startswith("v2_")]
        t["key"] = t.key.str.replace(r"^(v3_|i_)", "", regex=True)
    if keep is not None:
        c, t = c[c.sentence.isin(keep)], t[t.sentence.isin(keep)]
    c = c.drop_duplicates("sentence", keep="last")
    t = t.drop_duplicates(["sentence", "test"], keep="last")
    c["model"], t["model"] = model, model
    return c, t


def verdict(r):
    v = str(r.verdict).upper()
    if v.startswith(("CONSULTED", "CATEGORY")):
        if r.isolation == r.isolation and r.isolation < 0.5:
            return "Answer echo", "needed and first, but its effect runs through the part it shares with the answer"
        if r.specificity == r.specificity and r.specificity < 0.5:
            return "Category effect", "needed and first, but the part it shares with same-kind words carries most of the effect"
        return "Consulted bridge", "needed early, beats noise, comes first, and its own part carries the effect"
    return str(r.verdict).capitalize(), r.verdict_reason


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    c4, t4 = model_tables(["card4_qwen35_4b/shard*/cards.csv"], ["card4_qwen35_4b/shard*/tests.csv"], "Qwen3.5-4B", True)
    ids = set(c4.sentence)
    c27, t27 = model_tables(["big_qwen36_27b/shard*/cards.csv", "big_raw_qwen36_27b/shard*/cards.csv", "v4_27b/shard*/cards.csv"],
                            ["big_qwen36_27b/shard*/tests.csv", "big_raw_qwen36_27b/shard*/tests.csv", "v4_27b/shard*/tests.csv"],
                            "Qwen3.6-27B", keep=ids)
    v4_27 = set(load(["v4_27b/shard*/cards.csv"]).get("sentence", pd.Series(dtype=str)))
    on = load(["v4_27b/onset27b_qwen36_27b.csv"])
    cards, tests = pd.concat([c4, c27], ignore_index=True), pd.concat([t4, t27], ignore_index=True)
    order = {k: i for i, k in enumerate(FAM)}
    cards["family"] = cards.sentence.str[0].map(FAM)
    cards["fo"] = cards.sentence.str[0].map(order)
    rows = []
    for _, c in cards.sort_values(["fo", "sentence", "model"]).iterrows():
        tt = tests[(tests.sentence == c.sentence) & (tests.model == c.model)]
        gate = lambda g: (tt[tt.test.str.startswith(f"Gate {g}")].result.iloc[0]  # noqa: E731
                          if len(tt[tt.test.str.startswith(f"Gate {g}")]) else "not reached")
        t1 = tt[tt.test.str.startswith("1 ")]
        fails = tt[(tt.result == "fail") & tt.test.str.startswith("Gate")]
        key = t1.what_happened.iloc[0] if len(t1) else (fails.what_happened.iloc[0] if len(fails) else "")
        v, why = verdict(c)
        g = [gate(i) for i in range(1, 5)]
        if c.model == "Qwen3.5-4B":
            note = "v4 scorecard (own-part answer onset)."
        elif c.sentence in v4_27:
            note = "Re-scored with v4: its answer onset moved when found from the answer's own part."
        elif len(on) and c.sentence in set(on.question):
            note = "v4 onset checked: unchanged, so these numbers stand."
        else:
            note = "Gates 1–2 fail, which don't depend on the onset."
        rows.append(dict(key=f"{c.sentence}|{c.model}", id=c.sentence, model=c.model, family=c.family,
                         question=c.question, bridge=c.bridge, answer=c.answer, designed_bad=str(c.predicted).startswith("bad"),
                         verdict=v, why=why, model_said=c.model_said,
                         p_answer=round(c.P_base, 2) if c.P_base == c.P_base else None,
                         gate1=g[0], gate2=g[1], gate3=g[2], gate4=g[3], gates="".join(SYM.get(x, "·") for x in g),
                         need=c.get("need"), specificity=c.get("specificity"), isolation=c.get("isolation"),
                         lead=c.get("lead"), quality=c.get("quality"), key_result=key, note=note, order=int(c.fo) * 1000))
    res = pd.DataFrame(rows)
    for k in ["need", "specificity", "isolation", "quality"]:
        res[k] = pd.to_numeric(res[k], errors="coerce").round(2)
    res["lead"] = pd.to_numeric(res["lead"], errors="coerce").astype("Int64")
    res.loc[res.quality.isna(), "quality"] = 0.0
    qorder = {k: i for i, k in enumerate(res.drop_duplicates("id").id)}
    res["order"] = res.id.map(qorder)
    res.to_csv(out / "results.csv", index=False)
    tests = tests.assign(id=tests.sentence, key=tests.sentence + "|" + tests.model + "|" + tests.test)
    tests[["key", "id", "model", "order", "test", "asks", "what_happened", "value", "result"]].to_csv(out / "tests.csv", index=False)
    print(res.groupby(["model", "verdict"]).size().unstack(0).fillna(0).astype(int).to_string())
    print(res[res.quality >= 0.5][["id", "model", "quality"]].to_string(index=False))


if __name__ == "__main__":
    main()
