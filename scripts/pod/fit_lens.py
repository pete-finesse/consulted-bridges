"""ON THE POD: fit a config's J-lens on its own model when the config asks for one (a `jlens_fit` block and
lens.kwargs.path), with the paper's own fitting code (silent_dissent.jlens, as repos/silent-dissent/scripts/
fit_jlens.py). Skips if the lens file already exists; an interrupted fit resumes from <path>.ckpt.

    uv run python scripts/pod/fit_lens.py --config gemma4_e4b_it_refit                # one GPU
    uv run python scripts/pod/fit_lens.py --config gemma4_e4b_it_refit --shard 0/8    # pod 0 of 8: prompts 0, 8, 16, ...
    uv run python scripts/pod/fit_lens.py --config gemma4_e4b_it_refit --merge        # n_prompts-weighted mean of the shards
"""
import glob
import argparse
import logging
from pathlib import Path

from silent_dissent import jlens
from silent_dissent.model import load_model

from consulted_bridges.load import read_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--shard", default=None, help="i/n: fit prompts i, i+n, i+2n, ...")
ap.add_argument("--merge", action="store_true")
args = ap.parse_args()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

cfg = read_config(args.config)
fc = cfg.get("jlens_fit")
out = cfg["lens"]["kwargs"].get("path")
if not fc or not out:
    raise SystemExit(f"{args.config}: no lens to fit (uses a published lens)")
if Path(out).exists():
    raise SystemExit(f"{out} exists: nothing to do")
if args.merge:
    shards = sorted(p for p in glob.glob(f"{out}.shard*of*") if not p.endswith(".ckpt"))
    n_expected = {p.rsplit("of", 1)[1] for p in shards}
    if not shards or len(n_expected) != 1 or len(shards) != int(n_expected.pop()):
        raise SystemExit(f"incomplete or mixed shard set: {shards}")
    jlens.merge(shards, out)
    raise SystemExit(f"merged {len(shards)} shards -> {out}")
Path(out).parent.mkdir(parents=True, exist_ok=True)
prompts = jlens.load_corpus(fc["corpus"], fc["n_prompts"], fc.get("seed", 0))
if args.shard:
    i, n = map(int, args.shard.split("/"))
    prompts, out = prompts[i::n], f"{out}.shard{i}of{n}"
    if Path(out).exists():
        raise SystemExit(f"{out} exists: nothing to do")
m = cfg["model"]
model, tok = load_model(m["name"], m.get("dtype", "bfloat16"), "auto", m.get("chat_template_kwargs"))
jlens.fit(model, tok, prompts, out, dim_batch=fc.get("dim_batch", 8), max_seq_len=fc.get("max_seq_len", 128))
print(f"fitted on {len(prompts)} prompts -> {out}")
