"""ON THE POD: a quick check that a new config loads and reads sensibly (chat format, answer probability, single tokens, lens ranks)."""
import sys
from consulted_bridges.load import load
from consulted_bridges.text import chat
from consulted_bridges.interventions import answer_probs
h = load(sys.argv[1]); print(type(h.model).__name__, h.n_layers, h.band, h.device)
t = chat(h, "What letter does the name of the country where the Colosseum is located start with? Answer with one letter.")
print(repr(t[-200:]))
r = answer_probs(h, t, ["I", "R"]); print(r["top"]); print({k: round(v["p_full"], 3) for k, v in r["cands"].items()})
print("pieces", [len(h.encode(" " + w)) for w in ["Italy", "Rome", "France", "table"]])
from consulted_bridges.readout import readout
ro = readout(h, t, ["Italy", "France", "Tokyo"], lenses=["jlens"], positions=[-1])
print("band ranks", [[int(ro.ranks[0, -1][l, j]) for l in h.band] for j in range(3)])
