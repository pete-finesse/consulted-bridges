"""Build the question bank's tables (results.csv, tests.csv) from the latest scorecard results of each model.

Three sets of questions, each on both models:
  Question bank   the 62 core questions.
                  Qwen3.5-4B   v4 (results/card4_qwen35_4b): own-part answer onset, raw answer tokens, two-window onset.
                  Qwen3.6-27B  results/big_qwen36_27b + big_raw_qwen36_27b, with every question whose onset moved
                               under v4 replaced by its v4 re-score (results/v4_27b).
  Close variant   38 proposed questions (ids V_...), v4.1: results/prop_qwen35_4b, results/prop_qwen36_27b.
  Trend question  23 proposed questions (ids T_...), trimmed v4.2 card with swaps: results/trend_<config>
                  (added when that run exists).

Swap test (docs/swap_test_rules.md): Neuronpedia's move over the whole pre-onset range, rule A (question tokens) and
rule B (q->end). Headline per question from results/swap2_<config> (swaps-only runs) or the trend cards. It sits next
to quality, not inside it. Questions in results/swap2_jobs without results yet are marked "queued".

Verdicts use one rule for every model and set (quality's own factors): needed early, beats noise and comes first ->
Answer echo if isolation < 0.5, else Category effect if specificity < 0.5, else Consulted bridge.

  uv run python -m consulted_bridges.bank OUTDIR
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[2] / "results"
FAM = {"L": "Letter of the hidden word", "R": "Leg-count riddle", "O": "Parts of an object", "X": "Reference (round 2)",
       "Q": "Sequences and arithmetic", "W": "Wordplay", "S": "Science and nature", "C": "Culture and games"}
# close variants join the family of the question they vary; trend questions are new families
VAR_FAM = {"V_LM": "L", "V_LP": "L", "V_LU": "L", "V_T": "C", "V_D": "Q", "V_R": "R"}
TREND_FAM = {"T_place": "Number fact about a place", "T_code": "Code number of a place", "T_emblem": "Emblem of a place",
             "T_rule": "Convention of a place", "T_named": "Number fact about a named thing",
             "T_fact": "Letter label by fact", "T_bad": "Trend controls (designed to fail)"}
SETS = {"bank": "Question bank", "variant": "Close variant", "trend": "Trend question"}
MODELS = {"qwen35_4b": "Qwen3.5-4B", "qwen36_27b": "Qwen3.6-27B"}
SYM = {"pass": "✓", "fail": "✗"}
SWAP_COLS = [f"swap{k}_{c}" for k in "AB" for c in ("follow", "n", "counts", "windows_counted", "windows")]
# what each scorecard test is for (HANDOFF section 3): needed by gates / quality, reported only, or the old exchange swap
ROLE = {"Gate": "Gate", "1 ": "Scored", "2 ": "Scored", "3 ": "Scored", "6 ": "Scored", "7 ": "Scored", "7b": "Scored",
        "Quality": "Scored", "4 ": "Reported only", "5 ": "Reported only", "8 ": "Old exchange swap", "9 ": "Old exchange swap"}


def load(globs):
    fs = [f for g in globs for f in sorted(glob.glob(str(R / g))) if Path(f).stat().st_size > 1]  # skip empty shards
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def strip(s):
    return s.str.replace(r"^(v3_|i_)", "", regex=True)


def model_tables(cards_globs, tests_globs, model, qset, strip_prefix=False, keep=None):
    c, t = load(cards_globs), load(tests_globs)
    if c.empty:
        return c, t
    if strip_prefix:
        for d in (c, t):
            d["sentence"] = strip(d.sentence)
        c, t = c[~c.sentence.str.startswith("v2_")], t[~t.sentence.str.startswith("v2_")]
    if keep is not None:
        c, t = c[c.sentence.isin(keep)], t[t.sentence.isin(keep)]
    c = c.drop_duplicates("sentence", keep="last").copy()
    t = t.drop_duplicates(["sentence", "test"], keep="last").copy()
    c["model"], t["model"], c["set"] = model, model, qset
    return c, t


def family(sid):
    for p, f in TREND_FAM.items():
        if sid.startswith(p):
            return f
    for p, f in VAR_FAM.items():
        if sid.startswith(p + "_"):
            return FAM[f]
    return FAM.get(sid[0], "Other")


def verdict(r):
    v = str(r.verdict).upper()
    if v.startswith(("CONSULTED", "CATEGORY")):
        if r.isolation == r.isolation and r.isolation < 0.5:
            return "Answer echo", "needed and first, but its effect runs through the part it shares with the answer"
        if r.specificity == r.specificity and r.specificity < 0.5:
            return "Category effect", "needed and first, but the part it shares with same-kind words carries most of the effect"
        return "Consulted bridge", "needed early, beats noise, comes first, and its own part carries the effect"
    return str(r.verdict).capitalize(), r.verdict_reason


def swaps():
    """Swap headline per (question, model), the per-window summary as test rows, and the queued jobs."""
    head, summ, queued = [], [], set()
    for cfg, model in MODELS.items():
        jobs = load([f"swap2_jobs/{cfg}.csv"])
        if len(jobs):
            queued |= {(s, model) for s in strip(jobs.sentence)}
        for run in (f"swap2_{cfg}", f"trend_{cfg}"):
            h, s = load([f"{run}/shard*/cards.csv"]), load([f"{run}/shard*/swapsum.csv"])
            if len(h) and "swapA_follow" in h:
                h = h.assign(sentence=strip(h.sentence), model=model)
                head.append(h[["sentence", "model"] + [c for c in SWAP_COLS if c in h]].dropna(subset=["swapA_n", "swapB_n"], how="all"))
            if len(s):
                summ.append(s.assign(sentence=strip(s.sentence), model=model))
    head = pd.concat(head, ignore_index=True).drop_duplicates(["sentence", "model"], keep="last") if head else pd.DataFrame(columns=["sentence", "model"] + SWAP_COLS)
    summ = pd.concat(summ, ignore_index=True).drop_duplicates(["sentence", "model", "rule", "layers"], keep="last") if summ else pd.DataFrame()
    return head, summ, queued


RULE = {"A": "question tokens only", "B": "question to the end of the prompt (the tokens removal edits)"}


def swap_tests(summ, bridges):
    rows = []
    for i, s in enumerate(summ.sort_values(["sentence", "model", "rule", "whole", "layers"], ascending=[1, 1, 1, 0, 1]).itertuples()):
        where = "all layers before the answer forms" if s.whole else f"layers {s.layers}"
        counts = "not checked" if s.counts != s.counts else ("counts" if str(s.counts) == "True" else "doesn't count")
        b = bridges.get((s.sentence, s.model), "the bridge")
        follow = (f"{int(s.n_follow)} of {int(s.n_informative)} informative targets followed" if s.n_informative
                  else "no informative target")
        ctrl = (f"everyday words ({s.everyday}): {int(s.n_everyday_through)} of {int(s.n_everyday_readable)} readable got through"
                if s.n_everyday_readable else f"no everyday word ({s.everyday}) was readable")
        rows.append(dict(sentence=s.sentence, model=s.model, test=f"Swap {s.rule} · {where}",
                         asks=f"Move “{b}” onto another word of its kind, on the {RULE.get(s.rule, s.rule)}, in {where}: "
                              "does the answer become that word's answer?",
                         what_happened=f"Moved “{b}” onto {s.targets}: {follow} (removal alone gives “{s.fallback}”). "
                                       f"Control: {ctrl}, so this setting {counts}.",
                         value=s.follow_share, result=counts, role="Swap", order=100 + i))
    return pd.DataFrame(rows)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    c4, t4 = model_tables(["card4_qwen35_4b/shard*/cards.csv"], ["card4_qwen35_4b/shard*/tests.csv"], "Qwen3.5-4B", "bank", True)
    ids = set(c4.sentence)
    c27, t27 = model_tables(["big_qwen36_27b/shard*/cards.csv", "big_raw_qwen36_27b/shard*/cards.csv", "v4_27b/shard*/cards.csv"],
                            ["big_qwen36_27b/shard*/tests.csv", "big_raw_qwen36_27b/shard*/tests.csv", "v4_27b/shard*/tests.csv"],
                            "Qwen3.6-27B", "bank", keep=ids)
    parts = [(c4, t4), (c27, t27)]
    for cfg, model in MODELS.items():
        parts.append(model_tables([f"prop_{cfg}/shard*/cards.csv"], [f"prop_{cfg}/shard*/tests.csv"], model, "variant"))
        parts.append(model_tables([f"trend_{cfg}/shard*/cards.csv"], [f"trend_{cfg}/shard*/tests.csv"], model, "trend", True))
    v4_27 = set(load(["v4_27b/shard*/cards.csv"]).get("sentence", pd.Series(dtype=str)))
    on = load(["v4_27b/onset27b_qwen36_27b.csv"])
    cards = pd.concat([p[0] for p in parts if len(p[0])], ignore_index=True)
    tests = pd.concat([p[1] for p in parts if len(p[1])], ignore_index=True)
    cards = cards.drop(columns=[c for c in SWAP_COLS if c in cards])
    head, summ, queued = swaps()
    cards = cards.merge(head, on=["sentence", "model"], how="left")

    fams = list(FAM.values()) + list(TREND_FAM.values())
    cards["family"] = cards.sentence.map(family)
    cards["fo"] = cards.family.map({f: i for i, f in enumerate(fams)})
    cards["so"] = cards["set"].map({k: i for i, k in enumerate(SETS)})
    rows = []
    for _, c in cards.sort_values(["fo", "so", "sentence", "model"]).iterrows():
        tt = tests[(tests.sentence == c.sentence) & (tests.model == c.model)]
        gate = lambda g: (tt[tt.test.str.startswith(f"Gate {g}")].result.iloc[0]  # noqa: E731
                          if len(tt[tt.test.str.startswith(f"Gate {g}")]) else "not reached")
        t1 = tt[tt.test.str.startswith("1 ")]
        fails = tt[(tt.result == "fail") & tt.test.str.startswith("Gate")]
        key = t1.what_happened.iloc[0] if len(t1) else (fails.what_happened.iloc[0] if len(fails) else "")
        v, why = verdict(c)
        g = [gate(i) for i in range(1, 5)]
        failed = next((f"G{i + 1}" for i, x in enumerate(g) if x == "fail"), "")
        if c["set"] == "variant":
            note = "Proposed: a close variant of a question-bank question. v4.1 scorecard, run Nov 6."
        elif c["set"] == "trend":
            note = "Proposed: a new question type built from what the passing questions share. Trimmed v4.2 scorecard."
        elif c.model == "Qwen3.5-4B":
            note = "v4 scorecard (own-part answer onset)."
        elif c.sentence in v4_27:
            note = "Re-scored with v4: its answer onset moved when found from the answer's own part."
        elif len(on) and c.sentence in set(on.question):
            note = "v4 onset checked: unchanged, so these numbers stand."
        else:
            note = "Gates 1–2 fail, which don't depend on the onset."
        has_swap = c.get("swapA_n") == c.get("swapA_n") or c.get("swapB_n") == c.get("swapB_n")
        swap = "run" if has_swap else ("queued" if (c.sentence, c.model) in queued else "not run")
        rows.append(dict(key=f"{c.sentence}|{c.model}", id=c.sentence, model=c.model, set=SETS[c["set"]],
                         proposed=c["set"] != "bank", family=c.family,
                         question=c.question, bridge=c.bridge, answer=c.answer, designed_bad=str(c.predicted).startswith("bad"),
                         verdict=v, why=why, gate_failed=failed, model_said=c.model_said,
                         p_answer=round(c.P_base, 2) if c.P_base == c.P_base else None,
                         gate1=g[0], gate2=g[1], gate3=g[2], gate4=g[3], gates="".join(SYM.get(x, "·") for x in g),
                         need=c.get("need"), specificity=c.get("specificity"), isolation=c.get("isolation"),
                         lead=c.get("lead"), quality=c.get("quality"), swap=swap,
                         **{k: c.get(k) for k in SWAP_COLS}, key_result=key, note=note))
    res = pd.DataFrame(rows)
    for k in ["need", "specificity", "isolation", "quality", "swapA_follow", "swapB_follow"]:
        res[k] = pd.to_numeric(res[k], errors="coerce").round(2)
    for k in ["lead"] + [c for c in SWAP_COLS if not c.endswith(("follow", "counts"))]:
        res[k] = pd.to_numeric(res[k], errors="coerce").astype("Int64")
    res.loc[res.quality.isna(), "quality"] = 0.0
    qorder = {k: i for i, k in enumerate(res.drop_duplicates("id").id)}
    res["order"] = res.id.map(qorder)
    res.to_csv(out / "results.csv", index=False)

    tests["role"] = tests.test.map(lambda t: next((r for p, r in ROLE.items() if t.startswith(p)), "Reported only"))
    bridges = {(r.id, r.model): r.bridge for r in res.itertuples()}
    if len(summ):
        tests = pd.concat([tests, swap_tests(summ, bridges)], ignore_index=True)
    tests = tests.assign(id=tests.sentence, key=tests.sentence + "|" + tests.model + "|" + tests.test)
    tests[["key", "id", "model", "order", "role", "test", "asks", "what_happened", "value", "result"]].to_csv(out / "tests.csv", index=False)
    print(res.groupby(["set", "model", "verdict"]).size().unstack(1).fillna(0).astype(int).to_string())
    print(res.groupby(["set", "model", "swap"]).size().unstack(2).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
