"""The swap test (rules fixed in docs/swap_test_rules.md before the run).

Move the bridge's J-lens component onto another word (Neuronpedia's move, interventions kind "move") and check
whether the answer becomes that word's own answer. Two position rules:
  A  the question's own words (from its first word up to, not including, <|im_end|>)
  B  from the question's first word to the end of the prompt (the same tokens as removal)
Layer sets: each scorecard window that ends before the answer onset, and all layers 0 .. onset-1 ("whole").
A setting (question, rule, layer set) COUNTS if at least one everyday word is readable there and none gets through.

  swap_test(card, onset) -> (rows, summary)     one row per edit; one summary row per setting
"""
from __future__ import annotations

from .interventions import Edit
from .text import chat, resolve_positions

EVERYDAY = ["table", "river", "garden", "lamp", "bottle", "window"]   # first 3 that qualify are used
N_TARGETS, N_EVERYDAY = 4, 3


def rule_positions(card) -> dict[str, list[int]]:
    h, text = card.h, card.text
    ids = h.encode(text)
    qend = resolve_positions(h, text, card.ALL)
    ends = [i for i, t in enumerate(ids) if h.tok.decode([t]) == "<|im_end|>"]
    cut = ends[-1] if ends else len(ids)
    return {"A": [p for p in qend if p < cut], "B": qend}


def layer_sets(card, onset: int) -> dict[str, list[int]]:
    wins = [w for w in card.wins if w[-1] < onset]
    out = {f"{w[0]}-{w[-1]}": w for w in wins}
    out["whole"] = list(range(onset)) or card.wins[0]
    return out


def pick_words(card) -> dict:
    """Targets, everyday words and source controls, chosen from the unedited model only (no swap outcome)."""
    S, h = card.S, card.h
    right = S["answer"]
    belief = {}

    def own(w):
        if w not in belief:
            belief[w] = card.classify(card.gen(chat(h, S["check"].format(w), "")))
        return belief[w]

    bridge = S["bridge"]
    targets, seen = [], {right}
    for w in S["swaps"]:
        if len(targets) == N_TARGETS:
            break
        if w in bridge or not card.one_token(w):
            continue
        a = own(w)
        if a != "other" and a not in seen:
            targets.append(w)
            seen.add(a)
    everyday = [w for w in EVERYDAY if card.one_token(w) and own(w) != right][:N_EVERYDAY]
    unused = next((w for w in S["swaps"] if w not in targets and w not in bridge and card.one_token(w)), None)
    sources = [("source_everyday", "table")] + ([("source_kind", unused)] if unused else [])
    return dict(targets=targets, everyday=everyday, sources=sources, belief=belief)


def swap_test(card, onset: int):
    S = card.S
    b = S["bridge"][0]
    right = S["answer"]
    words = pick_words(card)
    own = words["belief"]
    rows, summary = [], []
    for rule, pos in rule_positions(card).items():
        for ln, layers in layer_sets(card, onset).items():
            # the move contains a removal of the bridge (b only, the word the move takes from)
            fb = card.classify(card.gen(card.text, [Edit("ablate", [b], layers, pos)]))
            base = dict(sentence=card.name, rule=rule, layers=ln, whole=ln == "whole", n_layers=len(layers),
                        positions=f"{pos[0]}-{pos[-1]}", fallback=fb)
            rows.append(dict(base, role="removal", source=b, target=None, own_answer=None, answered=fb))
            n_inf = n_follow = n_read = n_through = 0
            src_follow = {}
            plan = ([("target", b, w) for w in words["targets"]] + [("everyday", b, w) for w in words["everyday"]] +
                    [(role, s, words["targets"][0]) for role, s in words["sources"] if words["targets"]])
            for role, src, tgt in plan:
                out = card.gen(card.text, [Edit("move", [src, tgt], layers, pos)])
                got = card.classify(out)
                a = own.get(tgt) or card.classify(card.gen(chat(card.h, S["check"].format(tgt), "")))
                informative = a not in ("other", fb, right)
                hit = informative and got == a
                if role == "target" and informative:
                    n_inf += 1
                    n_follow += hit
                if role == "everyday" and informative:
                    n_read += 1
                    n_through += hit
                if role.startswith("source"):
                    src_follow[role] = hit
                rows.append(dict(base, role=role, source=src, target=tgt, own_answer=a, model_said=out, answered=got,
                                 informative=informative, follows=hit))
            counts = None if n_read == 0 else n_through == 0
            summary.append(dict(base, right=right, targets=", ".join(words["targets"]),
                                everyday=", ".join(words["everyday"]), n_informative=n_inf, n_follow=n_follow,
                                follow_share=None if n_inf == 0 else round(n_follow / n_inf, 3),
                                n_everyday_readable=n_read, n_everyday_through=n_through, counts=counts,
                                source_everyday_follows=src_follow.get("source_everyday"),
                                source_kind_follows=src_follow.get("source_kind")))
    return rows, summary


def headline(summary) -> dict:
    """Per question: the whole pre-onset range under each rule (the reported number; windows are detail)."""
    out = {}
    for r in summary:
        if r["whole"]:
            k = r["rule"]
            out[f"swap{k}_follow"] = r["follow_share"]
            out[f"swap{k}_n"] = r["n_informative"]
            out[f"swap{k}_counts"] = r["counts"]
    for k in ("A", "B"):
        wins = [r for r in summary if r["rule"] == k and not r["whole"]]
        out[f"swap{k}_windows_counted"] = sum(r["counts"] is True for r in wins)
        out[f"swap{k}_windows"] = len(wins)
    return out
