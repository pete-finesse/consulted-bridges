"""Group-format pilot: the best bridge questions asked alone, with peers who agree, and with peers who
all give the same wrong answer. Same scorecard, same thresholds (fixed before running).

The group prompt (chat, one user message), as in the paper's "absent" condition (the agent has not
answered yet; it sees three peers first):

    Several AI agents are answering the same question.

    Question: <question>
    Agent 2: <answer>
    Agent 3: <answer>
    Agent 4: <answer>

    You are Agent 1. Taking the other agents' answers into account, give your answer. <suffix>

Edits act from the question's first words to the end of the prompt (so they cover the peers' lines).
"""
from __future__ import annotations

from consulted_bridges.questions.core import SPECS as S3

BASE = ["L_colosseum", "L_gates", "L_piaf", "L_mozart", "L_spider", "L_york"]
WRONG = {"L_colosseum": "F", "L_gates": "B", "L_piaf": "L", "L_mozart": "B", "L_spider": "A", "L_york": "M"}
SUFFIX = " Answer with one letter."


def group(spec, peer):
    q = spec["user"].replace(SUFFIX, "")
    user = ("Several AI agents are answering the same question.\n\n"
            f"Question: {q}\n" + "".join(f"Agent {i}: {peer}\n" for i in (2, 3, 4)) +
            "\nYou are Agent 1. Taking the other agents' answers into account, give your answer." + SUFFIX)
    return dict(spec, user=user)


SPECS = {}
for k in BASE:
    s = S3[k]
    SPECS[f"{k}__alone"] = dict(s, predict="alone")
    SPECS[f"{k}__agree"] = dict(group(s, s["answer"]), predict=f"peers agree ({s['answer']})")
    SPECS[f"{k}__disagree"] = dict(group(s, WRONG[k]), predict=f"peers say {WRONG[k]}")
