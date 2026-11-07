"""Prompts, word pieces and token positions."""
from __future__ import annotations

import warnings

from silent_dissent.entity_facts import probe_tokens
from silent_dissent.prompts import render


def chat(h, user: str, prefill: str = "", system: str | None = None) -> str:
    """Chat prompt as text: the model's chat template (Qwen: thinking off, from the config)
    with `prefill` already written at the start of the assistant's reply, as in Neuronpedia."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return render(h.tok, msgs, prefix=prefill)


def word_tokens(h, words: list[str], dedupe: bool = True) -> dict[str, list[int]]:
    """Token ids that stand for each word, the paper's way: the first token of " <w>" for each
    content word of the name (silent_dissent.entity_facts.probe_tokens). A word's rank is the best
    rank over its ids. With `dedupe`, ids shared by two words are dropped from both, as in the
    paper (a shared piece can't tell them apart); a word left with no id gets no rank."""
    raw = {w: probe_tokens(h.tok, w) for w in words}
    if not dedupe:
        return raw
    count: dict[int, int] = {}
    for ids in raw.values():
        for t in set(ids):
            count[t] = count.get(t, 0) + 1
    out = {w: [t for t in ids if count[t] == 1] for w, ids in raw.items()}
    for w in words:
        if raw[w] and not out[w]:
            warnings.warn(f"{w!r} shares all its first pieces with another word; it gets no rank")
    return out


def piece_strings(h, ids: list[int]) -> list[str]:
    return [h.tok.decode([t]) for t in ids]


def offsets(h, text: str) -> list[tuple[int, int]]:
    return h.tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]


def resolve_positions(h, text: str, spec, n: int | None = None) -> list[int]:
    """Absolute token positions in the prompt `text` from a spec:
      "last" | "all" | int (negative = from the end) | list of ints |
      "span:<s>"  the tokens overlapping the last occurrence of substring s |
      "from:<s>"  from the first token of the last occurrence of s to the end."""
    n = n if n is not None else len(h.encode(text))
    if spec is None or spec == "last":
        return [n - 1]
    if spec == "all":
        return list(range(n))
    if isinstance(spec, int):
        return [spec % n]
    if isinstance(spec, (list, tuple)):
        return sorted({int(p) % n for p in spec})
    if isinstance(spec, str) and ":" in spec:
        kind, sub = spec.split(":", 1)
        at = text.rfind(sub)
        if at < 0:
            raise ValueError(f"{sub!r} not found in the prompt")
        offs = offsets(h, text)
        hit = [i for i, (a, b) in enumerate(offs) if a < at + len(sub) and b > at]
        if kind == "span":
            return hit
        if kind == "from":
            return list(range(hit[0], n))
    if isinstance(spec, str) and spec.lstrip("-").isdigit():
        return [int(spec) % n]
    raise ValueError(f"bad position spec {spec!r}")
