"""Why is specificity low? For a bridge and its same-kind controls, in the layers before the answer forms:

  overlap        cosine between the bridge's and the control's J-lens directions (mean over the early
                 layers), next to the mean |cosine| of random word pairs at the same layers (baseline)
  control        drop from removing the control's whole direction (what the scorecard used)
  control-only   drop from removing only the part of the control NOT shared with the bridge
                 (if this is ~0, the control's damage came from the part it shares with the bridge)
  bridge-only    drop from removing only the part of the bridge NOT shared with the control
                 (if this stays high, there is a bridge-specific part and the model uses it)

  uv run python scripts/spec_check.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import torch


from consulted_bridges.legacy.bridge_card3 import Card, NULL_POOL  # noqa: E402
from consulted_bridges.questions.core import SPECS  # noqa: E402
from consulted_bridges.interventions import Edit, directions  # noqa: E402
from consulted_bridges.load import load  # noqa: E402

CHECKS = {
    "L_colosseum": ["Spain", "France", "Germany"],
    "L_mozart": ["Vienna", "Graz", "Munich"],
    "L_gates": ["Boston", "Chicago"],
    "L_york": ["Ottawa", "Vancouver"],
    "X_spider": ["ticks", "ants", "bees", "dogs"],
}


def unit(h, w, l):
    return directions(h, Edit("ablate", [w], [l]), l)[0][0].float()


def main():
    h = load("qwen35_4b")
    rows = []
    for name, ctrls in CHECKS.items():
        S = SPECS[name]
        c = Card(h, name, S)
        forms = S["answers"][S["answer"]]
        base = c.P(c.text, forms)
        pa = [c.drop(base, c.P(c.text, forms, c.rm(S["answer_dirs"], w))) for w in c.wins]
        onset = c.wins[next(i for i, d in enumerate(pa) if d >= 0.5)][0]
        early = [w for w in c.wins if w[-1] < onset]
        layers = sorted({l for w in early for l in w})
        b = S["bridge"]
        best = lambda edits_for: max(c.drop(base, c.P(c.text, forms, edits_for(w))) for w in early)  # noqa: E731
        bfull = best(lambda w: c.rm(b, w))
        # random-pair baseline at the same layers
        rng = random.Random(0)
        pool = [w for w in NULL_POOL if c.one_token(w)]
        with torch.no_grad():
            rand = []
            for _ in range(40):
                x, y = rng.sample(pool, 2)
                rand.append(sum(abs(float(unit(h, x, l) @ unit(h, y, l))) for l in layers) / len(layers))
        rbase = sum(rand) / len(rand)
        print(f"\n{name}: {S['user']}\n  bridge {b[0]!r} -> {S['answer']}; base P {base:.2f}; answer onset L{onset}; "
              f"early layers {layers[0]}-{layers[-1]}; removing {b[0]!r}: drop {bfull:.2f}; random-pair |cos| {rbase:.2f}",
              flush=True)
        for w in ctrls:
            if not c.one_token(w):
                print(f"  {w}: not a single token, skipped")
                continue
            with torch.no_grad():
                cos = sum(float(unit(h, w, l) @ unit(h, b[0], l)) for l in layers) / len(layers)
            full = best(lambda win: c.rm([w], win))
            conly = best(lambda win: [Edit("ablate_excl", [w] + b, win, c.ALL)])
            bonly = best(lambda win: [Edit("ablate_excl", [bb, w], win, c.ALL) for bb in b])
            rows.append(dict(question=name, bridge=b[0], answer=S["answer"], control=w, overlap=round(cos, 3),
                             random_pair=round(rbase, 3), bridge_drop=round(bfull, 3), control_drop=round(full, 3),
                             control_only_drop=round(conly, 3), bridge_only_drop=round(bonly, 3),
                             specificity=round(max(0, 1 - full / bfull), 3) if bfull else None))
            print(f"  {w:10s} overlap {cos:.2f} | remove {w}: {full:.2f} | {w} minus {b[0]}: {conly:.2f} | "
                  f"{b[0]} minus {w}: {bonly:.2f}", flush=True)
    out = Path("results/spec_check_qwen35_4b.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("ALLDONE wrote", out, flush=True)


if __name__ == "__main__":
    main()
