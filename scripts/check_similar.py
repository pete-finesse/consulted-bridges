"""Check the proposed close variants: every edited word must be one token with a leading space.
  uv run python scripts/proposed/check_similar.py
"""
import sys
from pathlib import Path

from transformers import AutoTokenizer  # noqa: E402

from consulted_bridges.questions.proposed.similar import SPECS  # noqa: E402

t = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
one = lambda w: len(t(" " + w, add_special_tokens=False)["input_ids"]) == 1  # noqa: E731
bad = {}
for k, S in SPECS.items():
    words = S["bridge"] + S["same_kind"] + S["swaps"] + S["answer_dirs"]
    b = [w for w in words if not one(w)]
    if b:
        bad[k] = b
    assert S["answer"] in S["answers"], k
    assert S["q_start"] in S["user"], k
print(len(SPECS), "proposals;", "all single-token" if not bad else f"multi-token words: {bad}")
