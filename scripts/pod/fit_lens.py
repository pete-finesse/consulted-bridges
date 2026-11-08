"""ON THE POD: fit a config's J-lens on its own model when the config asks for one (a `jlens_fit` block and
lens.kwargs.path), with the paper's own fitting code (silent_dissent.jlens, as repos/silent-dissent/scripts/
fit_jlens.py). Skips if the lens file already exists; an interrupted fit resumes from <path>.ckpt.

    uv run python scripts/pod/fit_lens.py --config gemma4_e4b_it
"""
import argparse
import logging
from pathlib import Path

from silent_dissent import jlens
from silent_dissent.model import load_model

from consulted_bridges.load import read_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

cfg = read_config(args.config)
fc = cfg.get("jlens_fit")
out = cfg["lens"]["kwargs"].get("path")
if not fc or not out:
    raise SystemExit(f"{args.config}: no lens to fit (uses a published lens)")
if Path(out).exists():
    raise SystemExit(f"{out} exists: nothing to do")
Path(out).parent.mkdir(parents=True, exist_ok=True)
prompts = jlens.load_corpus(fc["corpus"], fc["n_prompts"], fc.get("seed", 0))
m = cfg["model"]
model, tok = load_model(m["name"], m.get("dtype", "bfloat16"), "auto", m.get("chat_template_kwargs"))
jlens.fit(model, tok, prompts, out, dim_batch=fc.get("dim_batch", 8), max_seq_len=fc.get("max_seq_len", 128))
print(f"fitted on {len(prompts)} prompts -> {out}")
