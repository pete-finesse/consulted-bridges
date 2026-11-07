"""Does the swap work when set like Neuronpedia's? (NEXT_EXPERIMENTS 1f, test 1)

Peter's Neuronpedia test (Qwen3.6-27B): swapping " Italy" -> " Spain" in layers 9-27 turns the Colosseum letter
from "I" into "S". The scorecard's swap acts on layers 6..onset (12-53 in the 27B) and rarely works.
For each letter question, swap the bridge with alternatives of the same kind (different first letter) and with
everyday words (control), at several layer ranges, all positions from the question on. Record the answer.
  follow = the model answers the swapped word's first letter.

  uv run python scripts/swap_check.py --config qwen36_27b
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd

from consulted_bridges.scorecard import Card  # noqa: E402
from consulted_bridges.questions.core import SPECS as S3  # noqa: E402
from consulted_bridges.interventions import Edit  # noqa: E402
from consulted_bridges.load import load  # noqa: E402

SV = importlib.import_module("consulted_bridges.questions.proposed.similar").SPECS
QUESTIONS = {"L_colosseum": S3["L_colosseum"], "L_gates": S3["L_gates"], "L_piaf": S3["L_piaf"],
             "L_mozart": S3["L_mozart"], "L_york": S3["L_york"], "V_LM_eiffel": SV["V_LM_eiffel"],
             "V_LM_machu": SV["V_LM_machu"], "V_LM_fuji": SV["V_LM_fuji"]}
COUNTRIES = ["Spain", "France", "Germany", "Japan", "Brazil", "Canada", "Egypt", "Mexico"]
CITIES = ["Munich", "Paris", "Tokyo", "Berlin", "Rome", "Madrid", "Chicago", "Lyon"]
EVERYDAY = ["table", "river", "garden"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen36_27b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    h = load(args.config)
    L = None
    rows = []
    for name, S in QUESTIONS.items():
        c = Card(h, name, S)
        L = c.L
        sc = c.sc
        b = S["bridge"][0]
        pool = COUNTRIES if S["user"].find("country") >= 0 and "city" not in S["user"] else CITIES
        alts = [w for w in pool if w[0] != b[0] and w != b and c.one_token(w)][:5]
        ranges = {"scorecard (6..onset-1)": list(range(sc(6), sc(54))),
                  "Neuronpedia 9-27": [l for l in range(round(9 * c.f), round(28 * c.f))] if L != 64 else list(range(9, 28)),
                  "early (first quarter)": list(range(0, L // 4)),
                  "early-mid": list(range(L // 8, 3 * L // 8)),
                  "mid": list(range(L // 4, L // 2)),
                  "late (before answer)": list(range(L // 2, 27 * L // 32))}
        base = c.gen(c.text)
        print(f"\n[{name}] bridge {b!r}, model says {base!r}", flush=True)
        for rn, ls in ranges.items():
            ls = [l for l in ls if l < L] or [L - 1]
            hits, tries = 0, 0
            got = []
            for w in alts + EVERYDAY:
                try:
                    ans = c.gen(c.text, [Edit("swap", [b, w], ls, c.ALL)]).replace("<|im_end|>", "").strip()
                except (ValueError, IndexError) as err:
                    print(f"  skip {w}: {err}", flush=True)
                    continue
                letter = ans.strip("*. ")[:1].upper()
                follow = letter == w[0].upper()
                if w in alts:
                    tries += 1
                    hits += follow
                rows.append(dict(question=name, bridge=b, range=rn, layers=f"{ls[0]}-{ls[-1]}", swapped_in=w,
                                 kind="same kind" if w in alts else "everyday", answer=ans, follows=follow))
                got.append(f"{w}->{ans!r}")
            print(f"  {rn:24s} layers {ls[0]}-{ls[-1]}: {hits}/{tries} follow | " + ", ".join(got), flush=True)
    out = Path(args.out or f"results/swap_check_{Path(args.config).stem}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
