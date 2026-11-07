"""Is "spider" a real bridge? Does the model consult it (in the middle layers) to decide the answer?

Same prompt as spider_legs.py. Tests, all on the J-lens "spider" direction:
A. Map: ablate spider, and swap spider -> bird / ant / dog, in sliding windows of layers,
   next to two control edits on strongly present, off-path words. Where does editing spider change
   the answer, and does it change it in the predicted direction?
B. Prediction test: swap spider -> animals with known leg counts (ant 6, dog 4, bird 2, snake 0,
   crab 10, octopus 8 = same as spider, a built-in control). If the model reads "spider" to decide,
   the answer should follow the swapped-in animal's leg count.
C. Matched controls: swap other words that are about as strongly present as spider (measured as
   the size of their component in the residual) for "bird". If those also give "2", the swap is just
   planting "bird", not showing spider is special.
D. Where in the prompt: the same spider edits on the question's tokens only vs the template tokens
   (after the question, where "spiders" ranks first).

Run on the pod:  uv run python scripts/spider_bridge.py [--config qwen35_4b]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch


from consulted_bridges.interventions import Edit, directions, generate  # noqa: E402
from consulted_bridges.load import load  # noqa: E402
from consulted_bridges.readout import readout  # noqa: E402
from consulted_bridges.text import chat, resolve_positions  # noqa: E402
from consulted_bridges.legacy.spider_legs import USER, capture  # noqa: E402
from consulted_bridges.legacy import spider_legs as SL  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

SL.CANDIDATES = ["Eight", "Six", "Four", "Two", "Ten", "Zero", "None", "8", "6", "4", "2", "10", "0"]
WORD = {0: ["zero", "none", "0", "no "], 2: ["two", "2"], 4: ["four", "4"], 6: ["six", "6"],
        8: ["eight", "8"], 10: ["ten", "10"]}
ANIMALS = {"ant": ("ants", 6), "bee": ("bees", 6), "dog": ("dogs", 4), "cat": ("cats", 4), "bird": ("birds", 2),
           "human": ("humans", 2), "snake": ("snakes", 0), "fish": ("fish", 0), "crab": ("crabs", 10),
           "octopus": ("octopuses", 8)}
Q = "span:How many legs do the animals that spin webs have? Answer in one word."
ALL = "from:How many"
TEMPLATE = "from:<|im_end|>"


def legs_in(ans: str):
    a = ans.lower().replace("<|im_end|>", "").strip(" .*")
    for n, ws in WORD.items():
        if any(a.startswith(w.strip()) for w in ws):
            return n
    return None


def spider_edit(kind, other, layers, pos, L):
    layers = [l for l in layers if l < L] or [L - 1]
    if kind == "ablate":
        return [Edit("ablate", ["spider", "spiders", "Spider"], layers, pos)]
    o, os_ = (other, ANIMALS[other][0]) if other in ANIMALS else (other, other + "s")
    return [Edit("swap", ["spider", o], layers, pos), Edit("swap", ["spiders", os_], layers, pos)]


def word_swap(word, other, layers, pos, L):
    layers = [l for l in layers if l < L] or [L - 1]
    return [Edit("swap", [word, other], layers, pos)]


@torch.no_grad()
def amount(h, store, word, layers, pos):
    """Mean |component| of the residual on the word's unit J-lens direction, as a fraction of |h|."""
    vals = []
    for l in layers:
        D = directions(h, Edit("ablate", [word], [l]), l)[0].to(store[l].device)
        H = store[l][pos].float()
        vals.append(((H @ D.T).abs().max(-1).values / H.norm(dim=-1)).max().item())
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen35_4b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    h = load(args.config)
    out = Path(args.out or f"results/spider_bridge_{Path(args.config).stem}")
    out.mkdir(parents=True, exist_ok=True)
    L = h.n_layers
    text = chat(h, USER, "")
    rows = []

    def run(section, name, edits, expect=None):
        try:
            r = SL.probs(h, text, edits)
        except ValueError as err:              # a word with no usable token piece
            print(f"[{section}] {name:44s} -> skipped: {err}")
            return None
        ans = generate(h, text, edits, max_new_tokens=5).replace("<|im_end|>", "").strip()
        c = r["cands"]
        ro = readout(h, text, ["spider"], lenses=["jlens"], edits=edits, dedupe=False)
        sp = ro.best_over_positions(0)
        sp = sp[(sp.layer >= min(12, L - 1)) & (sp.layer < min(30, L))]["spider"].min()
        row = {"section": section, "run": name, "answer": ans, "legs": legs_in(ans), "expect": expect,
               "P_eight": c["Eight"]["p_full"] + c["8"]["p_full"], "spider_best_rank": int(sp)}
        if expect is not None:
            ks = {0: ["Zero", "None", "0"], 2: ["Two", "2"], 4: ["Four", "4"], 6: ["Six", "6"], 8: ["Eight", "8"],
                  10: ["Ten", "10"]}[expect]
            row["P_expect"] = sum(c[k]["p_full"] for k in ks)
        rows.append(row)
        exp = f" expect={expect} P={row.get('P_expect', 0):.2f}" if expect is not None else ""
        print(f"[{section}] {name:44s} -> {ans!r:10s} P(8)={row['P_eight']:.2f}{exp} spider_rank={row['spider_best_rank']}")
        sys.stdout.flush()
        return row

    run("base", "no edit", [])

    # where the controls are as strong as spider
    ids, store = capture(h, text)
    allpos = resolve_positions(h, text, ALL)
    mid = [l for l in range(12, 22) if l < L] or [L - 1]
    cands = ["spider", "spiders", "animals", "legs", "webs", "insects", "humans", "dogs", "ways", "sentence",
             "word", "following", "living", "yarn", "Answer"]
    amt = {w: amount(h, store, w, mid, allpos) for w in cands}
    print("\nStrength in the residual (max over positions, mean over layers 12-21; fraction of |h|):")
    print(", ".join(f"{w}={v:.3f}" for w, v in sorted(amt.items(), key=lambda x: -x[1])))

    # A. layer map
    print("\n== A. Sliding 6-layer windows, all question+template positions ==")
    for s in range(0, L, 3):
        win = list(range(s, min(s + 6, L)))
        tag = f"L{win[0]}-{win[-1]}"
        run("A", f"ablate spider {tag}", spider_edit("ablate", None, win, ALL, L))
        for o in ("bird", "ant", "dog"):
            run("A", f"swap spider->{o} {tag}", spider_edit("swap", o, win, ALL, L), ANIMALS[o][1])
        run("A", f"control ablate animals {tag}", [Edit("ablate", ["animals"], [l for l in win if l < L], ALL)])
        run("A", f"control swap sentence->bird {tag}", word_swap("sentence", "bird", win, ALL, L), 2)
        if s + 6 >= L:
            break

    # B. prediction test
    print("\n== B. Swap spider -> animal with a known leg count ==")
    ranges = {"mid 12-21": mid, "8-23": [l for l in range(8, 24) if l < L], "all": list(range(L))}
    for rn, ls in ranges.items():
        for o, (_, n) in ANIMALS.items():
            run("B", f"swap spider->{o} [{rn}]", spider_edit("swap", o, ls, ALL, L), n)

    # C. matched controls
    print("\n== C. Swap other strongly present words -> bird (does any swap just plant 'bird'?) ==")
    for rn, ls in ranges.items():
        for w in [w for w in cands if w not in ("spider", "spiders")]:
            run("C", f"swap {w}->bird [{rn}] (strength {amt[w]:.3f})", word_swap(w, "bird", ls, ALL, L), 2)
            run("C", f"ablate {w} [{rn}]", [Edit("ablate", [w], ls, ALL)])

    # D. positions
    print("\n== D. Question tokens vs template tokens ==")
    for pn, pos in {"question": Q, "template": TEMPLATE}.items():
        for rn, ls in ranges.items():
            run("D", f"ablate spider [{rn}] @{pn}", spider_edit("ablate", None, ls, pos, L))
            run("D", f"swap spider->bird [{rn}] @{pn}", spider_edit("swap", "bird", ls, pos, L), 2)
            run("D", f"swap spider->ant [{rn}] @{pn}", spider_edit("swap", "ant", ls, pos, L), 6)

    df = pd.DataFrame(rows)
    df.to_csv(out / "edits.csv", index=False)
    pd.Series(amt).to_csv(out / "strength.csv")
    b = df[(df.section == "B")]
    print(f"\nB: answer matched the swapped animal's leg count in {(b.legs == b.expect).sum()} of {len(b)}")
    print("wrote", out)


if __name__ == "__main__":
    main()
