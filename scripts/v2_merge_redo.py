"""GPU run 1: fold the re-runs of questions that lost prompts to CUDA out-of-memory into the main run folders.

Each re-run (results/conformity_v2_redo_<cfg>/<frame>_<question>/shard0, and its block run under
results/conformity_block_redo_<cfg>/) holds the whole question, all prompts. Its rows replace that question's rows in
the main run's shards, and the re-run is copied in as its own shard folder (shard_redo_<question>), so every reader
that globs shard*/ sees each question exactly once.

  python scripts/v2_merge_redo.py --config qwen35_4b_jpp
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def fold(main: Path, redo: Path, q: str, files):
    for f in files:
        for p in sorted(main.glob(f"shard*/{f}")):
            if p.stat().st_size <= 1:
                continue
            t = pd.read_csv(p)
            if "question" in t and (t.question == q).any():
                t[t.question != q].to_csv(p, index=False)
                print(f"  dropped {int((t.question == q).sum())} rows of {q} from {p}")
    dst = main / f"shard_redo_{q}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(redo / "shard0", dst)
    print(f"  copied {redo}/shard0 -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen35_4b_jpp")
    args = ap.parse_args()
    cfg = args.config
    R = Path("results")
    for d in sorted((R / f"conformity_v2_redo_{cfg}").glob("*_*")):
        frame, q = d.name.split("_", 1)
        sfx = "" if frame == "neutral" else "defer_"
        print(f"{frame} {q}")
        fold(R / f"conformity_v2_{sfx}{cfg}", d, q, ["prompts.csv", "windows.csv"])
        b = R / f"conformity_block_redo_{cfg}" / d.name
        if b.exists():
            fold(R / f"conformity_block_{sfx}{cfg}", b, q, ["cells.csv", "dose.csv", "patch.csv"])


if __name__ == "__main__":
    main()
