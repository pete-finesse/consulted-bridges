"""Where every question stands, so the ones we use and the ones we no longer use are never confused.

Four statuses, each question in exactly one (tests/test_questions_status.py checks the partition):

  in_use        in a conformity question list: conformity.QUESTIONS / COUNTS for the Qwen models, or `selected`
                in a results/conformity_select_<config>.csv for another model. The development set.
  screened_out  run through a solo screen and not selected on any model. Kept: they are the funnel's denominator
                and must not be deleted or re-run by accident.
  candidate     drafted for the v2 runs (proposed/candidates.py), not yet screened on any model.
  retired       formats or sets we no longer run: the scorecard-v2 sentences (v2.py), every ideas/ set (all came
                out untestable), and the group-format pilot (group.py, superseded by conformity.py).

Module by module (for a later cleanup: nothing here is moved yet, because scorecard, bank and the legacy cards
import these paths):

  core.py              round-3 bank (L_ letters, R_ legs, O_ parts, X_ reference)   in_use (6 L_) / screened_out
  proposed/similar.py  V_ close variants                                           in_use (letters) / screened_out
  proposed/counts.py   N_ counts                                                   in_use (15) / screened_out
  proposed/trend.py    T_ trend questions                                          in_use (1) / screened_out
  proposed/candidates.py  C_ v2 candidates                                         candidate
  v2.py                scorecard-v2 sentences and the word tables core.py imports  retired (SPECS); tables in use
  ideas/*.py           W_ language, Q_ sequence, S_ science, C_<topic> culture      retired
  group.py             group-format pilot prompt                                   retired

    uv run python -m consulted_bridges.questions.status      # counts per status, and the in-use list per model
"""
from __future__ import annotations

import glob
import importlib
from pathlib import Path

MODULES = {
    "core": "consulted_bridges.questions.core",
    "v2": "consulted_bridges.questions.v2",
    "proposed.similar": "consulted_bridges.questions.proposed.similar",
    "proposed.counts": "consulted_bridges.questions.proposed.counts",
    "proposed.trend": "consulted_bridges.questions.proposed.trend",
    "proposed.candidates": "consulted_bridges.questions.proposed.candidates",
    "ideas.language": "consulted_bridges.questions.ideas.language",
    "ideas.sequence": "consulted_bridges.questions.ideas.sequence",
    "ideas.science": "consulted_bridges.questions.ideas.science",
    "ideas.culture": "consulted_bridges.questions.ideas.culture",
}
RETIRED_MODULES = {"v2", "ideas.language", "ideas.sequence", "ideas.science", "ideas.culture"}
CANDIDATE_MODULES = {"proposed.candidates"}
SCREENED_MODULES = {"core", "proposed.similar", "proposed.counts", "proposed.trend"}


def specs_by_module():
    return {name: dict(importlib.import_module(m).SPECS) for name, m in MODULES.items()}


def in_use_by_model(root=Path(__file__).resolve().parents[3]):
    """Question ids each model's conformity run uses: the fixed Qwen lists, plus `selected` rows of any
    results/conformity_select_<config>.csv."""
    from consulted_bridges.conformity import COUNTS, QUESTIONS
    out = {}
    for cfg, qs in QUESTIONS.items():
        out.setdefault(cfg, set()).update(qs)
    for cfg, qs in COUNTS.items():
        out.setdefault(cfg, set()).update(qs)
    for f in sorted(glob.glob(str(root / "results" / "conformity_select_*.csv"))):
        cfg = Path(f).stem[len("conformity_select_"):]
        if cfg.endswith("_candidates"):      # a candidate screen's selection, not a run list (candidates_selected)
            continue
        import pandas as pd
        t = pd.read_csv(f)
        out.setdefault(cfg, set()).update(t[t.selected].question)
    out.pop("tiny", None)
    return out


def candidates_selected(root=Path(__file__).resolve().parents[3]):
    """Candidates that passed a model's gate screen (results/conformity_select_<config>_candidates.csv), by config.
    They stay status "candidate" until a conformity run lists them."""
    import pandas as pd
    out = {}
    for f in sorted(glob.glob(str(root / "results" / "conformity_select_*_candidates.csv"))):
        t = pd.read_csv(f)
        out[Path(f).stem[len("conformity_select_"):-len("_candidates")]] = set(t[t.selected].question)
    return out


def status():
    """{question id: status}, every question in exactly one."""
    mods = specs_by_module()
    used = set().union(*in_use_by_model().values())
    st = {}
    for name, S in mods.items():
        for k in S:
            if name in RETIRED_MODULES:
                st[k] = "retired"
            elif name in CANDIDATE_MODULES:
                st[k] = "candidate"
            elif k in used:
                st[k] = "in_use"
            else:
                st[k] = "screened_out"
    return st


if __name__ == "__main__":
    import collections
    st = status()
    print(collections.Counter(st.values()))
    for cfg, qs in sorted(in_use_by_model().items()):
        print(f"{cfg}: {len(qs)} in use")
