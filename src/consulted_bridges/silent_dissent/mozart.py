"""Worked example: Peter's Neuronpedia Mozart checks (PLAYGROUND.md), reproduced in code.

1. Readout: PLAYGROUND prompts 1-8, words Salzburg (bridge), Graz (same country), Vienna
   (capital), Tokyo (random), plus Austria and Budapest; both lenses, every layer and position.
2. What the model answers in prompts 7-8 when it replies freely (no prefill).
3. Interventions on prompt 1, reporting P(Austria), P(France), P(Hungary):
   the swaps Peter tried (Salzburg -> Budapest, Vienna -> Budapest, Austria -> Budapest), and
   the removal test DESIGN.md asks for (ablate Salzburg vs Graz / Vienna / Tokyo).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import conditions as C
from . import facts as F
from ..interventions import Edit, compare, generate
from ..readout import readout
from ..text import chat

WORDS = ["Salzburg", "Graz", "Vienna", "Tokyo", "Austria", "Budapest"]
CANDIDATES = ["Austria", "France", "Hungary"]
PROMPTS = ["q_alone", "country_alone", "subject_alone", "answer_written", "filler",
           "np_agree", "np_pressure", "np_hidden"]


def run(h, out_dir: str | Path, peer: str = "France", layers: list[int] | None = None, alpha: float = 0.5,
        verbose: bool = True) -> dict[str, pd.DataFrame]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    fact = F.find(F.load(), "Wolfgang Amadeus Mozart")
    band = layers or h.band
    say(f"Fact {fact.uid}: {fact.composed} -> {fact.e3} (bridge {fact.e2}); band layers {band[0]}-{band[-1]}")

    # 1. readouts
    summ, last = [], []
    for i, cond in enumerate(PROMPTS, 1):
        text, start = C.build(h, fact, cond, peer_answer=peer)
        p0 = next((k for k, (a, b) in enumerate(h.tok(text, add_special_tokens=False,
                                                      return_offsets_mapping=True)["offset_mapping"]) if b > start), 0)
        ro = readout(h, text, WORDS)
        s = ro.summary(band, start=p0)
        s.insert(0, "prompt", f"{i} {cond}")
        summ.append(s)
        g = ro.last()
        g.insert(0, "prompt", f"{i} {cond}")
        last.append(g)
        if i == 1:
            say("pieces:", ro.pieces)
            ro.long().to_csv(out / "readout_prompt1_all_positions.csv", index=False)
    summ, last = pd.concat(summ), pd.concat(last)
    summ.to_csv(out / "readout_summary.csv", index=False)
    last.to_csv(out / "readout_last_position.csv", index=False)
    say("\nRank at the last position, J-lens, band mean of log10(rank+1) (lower = ranks better):")
    say(summ[summ.lens == "jlens"].pivot(index="prompt", columns="word", values="last_band_logrank").round(2)
        .to_string())
    say("\nSame, best rank over the question's positions (any position from the user message on):")
    say(summ[summ.lens == "jlens"].pivot(index="prompt", columns="word", values="allpos_band_logrank").round(2)
        .to_string())

    # 2. free replies to the pressure prompts (same user message, no prefill)
    replies = []
    for cond in ("np_pressure", "np_hidden"):
        user_msg = C._np_transcript(fact, "(answer submitted)" if cond == "np_hidden" else
                                    f"{fact.composed} {fact.e3}.", [peer] * 3)
        free = generate(h, chat(h, user_msg, ""), max_new_tokens=24)
        replies.append({"prompt": cond, "free_reply": free})
    replies = pd.DataFrame(replies)
    replies.to_csv(out / "free_replies.csv", index=False)
    say("\nFree replies (no prefill):\n" + replies.to_string(index=False))

    # 3. interventions on prompt 1
    text, _ = C.build(h, fact, "q_alone")
    q = "from:Complete the statement"
    mid = band[len(band) // 2]                 # steering at all band layers at once breaks the model
    runs = {
        "swap Salzburg->Budapest @last": [Edit("swap", ["Salzburg", "Budapest"], band, "last")],
        "swap Vienna->Budapest @last": [Edit("swap", ["Vienna", "Budapest"], band, "last")],
        "swap Austria->Budapest @last": [Edit("swap", ["Austria", "Budapest"], band, "last")],
        "ablate Salzburg @question": [Edit("ablate", ["Salzburg"], band, q)],
        "ablate Graz @question": [Edit("ablate", ["Graz"], band, q)],
        "ablate Vienna @question": [Edit("ablate", ["Vienna"], band, q)],
        "ablate Tokyo @question": [Edit("ablate", ["Tokyo"], band, q)],
        f"steer Salzburg -{alpha} @question L{mid}": [Edit("steer", ["Salzburg"], [mid], q, alpha=-alpha)],
        f"steer Budapest +{alpha} @last L{mid}": [Edit("steer", ["Budapest"], [mid], "last", alpha=alpha)],
    }
    iv = compare(h, text, CANDIDATES, runs)
    iv.to_csv(out / "interventions.csv", index=False)
    say("\nInterventions on prompt 1. p_full = P(' Austria') + P(' **Austria'), whole word; "
        "share = p_full among the three candidates:")
    order = list({"baseline": 0, **runs})
    tab = iv.pivot(index="run", columns="candidate", values="p_full").loc[order]
    tab["share_Austria"] = iv[iv.candidate == "Austria"].set_index("run").p_share.loc[order]
    tab["top_tokens"] = iv.drop_duplicates("run").set_index("run").top_tokens.loc[order]
    say(tab.to_string(float_format=lambda x: f"{x:.3g}"))
    say(f"\nWrote {out}/")
    return {"summary": summ, "last": last, "replies": replies, "interventions": iv}
