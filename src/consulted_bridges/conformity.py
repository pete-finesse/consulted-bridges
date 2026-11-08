"""Bridge under conformity: does an agent's answer still run through its own bridge when other agents
agree or disagree with it? (docs/conformity/outline.html)

For each letter question where the solo scorecard found the bridge needed and acting first, build:
  alone              the question as in the solo scorecard
  agree_k            k = 1, 2, 3 peers give the right letter
  disagree_k         k = 1, 2, 3 peers give the same wrong letter
  copy_k3            calibration questions only: a made-up subject (no bridge of its own), 3 peers
Group prompt (one user message, peers after the question, neutral instruction):

    Several AI agents are answering the same question.

    Question: <question>
    Agent 2: <letter>
    ...
    You are Agent 1. Give your own answer. Answer with one letter.

--frame defer uses the pilot's wording instead ("Taking the other agents' answers into account, give your answer."),
which invites deferring; results go to results/conformity_defer_<config>.

In every prompt:
  baseline   P(right), P(peers'), greedy answer, LO = log P(peers') - log P(right)
  onset      remove the direction of the letter the model gives (plain and raw-token forms), window by window,
             with the peers' letter tokens left out of the edit; the first window that drops it by >= 0.5.
             Early windows end before it.
  removals   in every early window, from the question's first word to the end of the prompt: the bridge,
             the same-kind words, the right letter (onset positions), 12 everyday words of matched strength
  reach      the prompt plus "First name the <kind>." with the reply started; best drop in P(bridge name)
             over early windows; readable when P(bridge name) >= 0.3 before the edit; passes at >= 0.5
  presence   J-lens rank of the bridge (and a same-kind and an unrelated word) at the last token, best over
             the config's layer band; present = rank < 100 (the paper's top-100)
Labels (fixed before the run, docs/conformity/outline.html):
  disagree        bridge still pulls  if bridge dLO >= noise + 2 and >= same-kind + 1
                  bridge not needed   if bridge dLO < noise + 1 and reach passes
                  mixed               otherwise
  alone / agree   bridge still needed if need (bridge drop - noise drop) >= 0.5
                  bridge not needed   if need < 0.2 and reach passes
                  mixed               otherwise

--set counts runs the count questions instead (COUNTS: "Answer in one word.", peers give the right count or one
off it); results go to results/conformity_counts[_defer]_<config>.

  uv run python -m consulted_bridges.conformity --config qwen36_27b [--set counts] [--shard 0/3] [--limit N] [--out DIR]
"""
from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path

import pandas as pd

from consulted_bridges.interventions import Edit, answer_probs
from consulted_bridges.legacy.bridge_card3 import FORMATS, SKIP
from consulted_bridges.load import load
from consulted_bridges.questions.core import SPECS as CORE
from consulted_bridges.questions.proposed.similar import SPECS as VARIANTS
from consulted_bridges.questions.proposed.trend import SPECS as TREND
from consulted_bridges.questions.proposed.counts import SPECS as COUNT_SPECS
from consulted_bridges.readout import readout
from consulted_bridges.silent_dissent.giveup import Probe, lay
from consulted_bridges.text import offsets, resolve_positions

SPECS = {**CORE, **VARIANTS, **TREND, **COUNT_SPECS}
EPS = 1e-6
SUFFIX = " Answer with one letter."
ONE = " Answer in one word."

# Letter questions where the saved solo scorecard found need >= 0.5 and the bridge acting first (controls
# built to fail left out), from the question bank's results table.
QUESTIONS = {
    "qwen36_27b": ["L_colosseum", "L_gates", "L_mozart", "L_piaf", "L_spider", "L_york",
                   "V_LM_angkor", "V_LM_eiffel", "V_LM_fuji", "V_LM_kremlin", "V_LM_machu", "V_LM_petronas",
                   "V_LM_pisa", "V_LM_pyramids", "V_LM_sagrada", "V_LM_sydney",
                   "V_LP_ali", "V_LP_beyonce", "V_LP_hemsworth", "V_LP_hendrix", "V_LP_obama_m",
                   "V_LU_alberta", "V_LU_berklee", "V_LU_fudan", "V_LU_pompeu"],
    "qwen35_4b": ["L_colosseum", "L_gates", "L_mozart", "L_york",
                  "V_LM_acropolis", "V_LM_angkor", "V_LM_brandenburg", "V_LM_christ", "V_LM_eiffel", "V_LM_fuji",
                  "V_LM_kremlin", "V_LM_machu", "V_LM_petronas", "V_LM_pisa", "V_LM_pyramids", "V_LM_sydney",
                  "V_LM_taj", "V_LP_ali", "V_LP_beyonce", "V_LP_hendrix", "V_LP_kapoor",
                  "V_LU_berklee", "V_LU_dalhousie", "V_LU_hopkins"],
}
QUESTIONS["tiny"] = ["L_colosseum", "L_gates"]
# Every letter question the solo scorecard is run on before a new model's conformity run.
LETTER = [k for k in SPECS if k.startswith(("L_", "V_L"))]


def questions_for(cfg):
    """The config's question list: fixed above for the Qwen models; for other models, the same rule applied to
    that model's own trimmed solo scorecard (consulted_bridges.conformity_select writes the file)."""
    if cfg in QUESTIONS:
        return QUESTIONS[cfg]
    f = Path(f"results/conformity_select_{cfg}.csv")
    if not f.exists():
        raise SystemExit(f"no question list for {cfg}: run the solo scorecard, then consulted_bridges.conformity_select")
    t = pd.read_csv(f)
    return t[t.selected].question.tolist()


# Count questions (--set counts): the 16 that the trimmed scorecard found consulted on Qwen3.6-27B
# (results/counts_qwen36_27b, results/counts_r2_qwen36_27b). 14 rest on two facts: Austria 9, Australia 6.
COUNTS = {"qwen36_27b": ["T_place_states_austria", "N_BP_schwarzenegger", "N_BP_haydn", "N_BP_schubert", "N_BP_strauss",
                         "N_BP_lauda", "N_BP_lamarr", "N_BP_klimt",
                         "N_BP_jackman", "N_BP_minogue", "N_BP_robbie", "N_BP_irwin", "N_BP_blanchett", "N_BP_hemsworth",
                         "N_LM_eiffel", "N_BP_gisele"]}
COUNTS["tiny"] = ["N_BP_haydn", "N_BP_jackman"]
# The peers' wrong count: one off the right one (a plausible slip), single digit so it is one token.
COUNT_WRONG = {"9": "8", "6": "7", "2": "1", "5": "4"}
# The pilot's six questions: the reading thresholds were set on them, so they are reported separately.
CALIBRATION = ["L_colosseum", "L_gates", "L_piaf", "L_mozart", "L_spider", "L_york"]
PILOT_WRONG = {"L_colosseum": "F", "L_gates": "B", "L_piaf": "L", "L_mozart": "B", "L_spider": "A", "L_york": "M"}
# Copy-only control: the calibration questions with a made-up subject, so the model has no bridge of its own.
FAKE = {
    "L_colosseum": ("the Colosseum", "the Varrenth Arch"),
    "L_gates": ("Bill Gates", "Tobren Halvik"),
    "L_piaf": ("Édith Piaf", "Mirelle Dasquin"),
    "L_mozart": ("Wolfgang Amadeus Mozart", "Corvin Aldemar Testrel"),
    "L_spider": ("spin webs", "weave glimmerdens"),
    "L_york": ("York University", "Quellmar University"),
    "N_BP_haydn": ("Joseph Haydn", "Tobren Halvik"),
    "N_BP_robbie": ("Margot Robbie", "Mirelle Dasquin"),
}
UNRELATED = ["Tokyo", "Cairo"]
TH = dict(pull_noise=2.0, pull_kind=1.0, none_noise=1.0, need=0.5, none_need=0.2, reach=0.5, readable=0.3, present=100)


def wrong_letter(qid, spec):
    if qid in PILOT_WRONG:
        return PILOT_WRONG[qid]
    if spec["answer"] in COUNT_WRONG and not spec["user"].endswith(SUFFIX):
        return COUNT_WRONG[spec["answer"]]
    for w in spec["same_kind"] + spec.get("swaps", []):
        if w[0].upper() != spec["answer"] and w[0].isalpha():
            return w[0].upper()
    return "Z" if spec["answer"] != "Z" else "A"


FRAMES = {"neutral": "You are Agent 1. Give your own answer.",
          # the pilot's wording, which invites deferring to the peers
          "defer": "You are Agent 1. Taking the other agents' answers into account, give your answer."}


def group_user(question, peer, k, frame="neutral", suffix=SUFFIX):
    return ("Several AI agents are answering the same question.\n\n" f"Question: {question}\n" +
            "".join(f"Agent {i}: {peer}\n" for i in range(2, 2 + k)) + "\n" + FRAMES[frame] + suffix)


def prompts(qid, frame="neutral"):
    """(condition, peers, peer letter, spec) for one question."""
    s = SPECS[qid]
    sfx = SUFFIX if s["user"].endswith(SUFFIX) else ONE   # letter questions, or count questions ("one word")
    q = s["user"][: -len(sfx)]
    right, wrong = s["answer"], wrong_letter(qid, s)
    out = [("alone", 0, wrong, dict(s))]
    for k in (1, 2, 3):
        out.append(("agree", k, right, dict(s, user=group_user(q, right, k, frame, sfx))))
    for k in (1, 2, 3):
        out.append(("disagree", k, wrong, dict(s, user=group_user(q, wrong, k, frame, sfx))))
    if qid in FAKE:
        a, b = FAKE[qid]
        out.append(("copy", 3, wrong, dict(s, user=group_user(q.replace(a, b), wrong, 3, frame, sfx))))
    return out


def peer_positions(h, text, peer, k):
    """Token positions of the peers' letters ("Agent i: X")."""
    offs = offsets(h, text)
    pos = []
    for m in re.finditer(r"Agent \d: (\S+)\n", text):
        a, b = m.span(1)
        pos += [i for i, (x, y) in enumerate(offs) if x < b and y > a]
    return sorted(set(pos))


class Prompt:
    def __init__(self, h, qid, cond, k, peer, spec):
        self.h, self.qid, self.cond, self.k, self.peer, self.spec = h, qid, cond, k, peer, spec
        self.pr = Probe(h, f"{qid}__{cond}{k or ''}", spec, peer)
        self.c = self.pr.c
        self.right = spec["answer"]
        self.rforms = spec["answers"][self.right]
        self.pforms = spec["answers"].get(peer, [peer])
        allpos = resolve_positions(h, self.c.text, self.c.ALL)
        skip = set(peer_positions(h, self.c.text, peer, k)) if k else set()
        self.allpos, self.onpos = allpos, [p for p in allpos if p not in skip]

    def probs(self, edits=None):
        """P(right), P(peers'), LO in one forward pass."""
        cands = list(dict.fromkeys(self.rforms + self.pforms))
        r = answer_probs(self.h, self.c.text, cands, edits, formats=FORMATS)["cands"]
        tot = lambda forms: sum(r[f].get(f"p[{x}]", 0.0) for f in forms for x in FORMATS)  # noqa: E731
        pr, pw = tot(self.rforms), tot(self.pforms)
        return pr, pw, math.log(pw + EPS) - math.log(pr + EPS)

    def rm(self, words, w, pos):
        return [Edit("ablate", words, w, pos)]

    def dirs(self, letter, forms):
        return list(dict.fromkeys([letter] + ["@" + f for f in forms]))

    def run(self, rows):
        h, c, S = self.h, self.c, self.spec
        b = S["bridge"]
        t0 = time.time()
        pr0, pw0, lo0 = self.probs()
        said = self.pr.answer()
        said_k = c.classify(said)
        if self.cond == "agree":
            state = "right" if said_k == self.right else "wrong"
        elif self.cond in ("disagree", "copy"):
            state = ("holds" if said_k == self.right else
                     ("partial give-in" if pr0 >= 0.3 else "full give-in") if said_k == self.peer else "other")
        else:
            state = "right" if said_k == self.right else "wrong"
        # onset: the letter the model gives (the peers' if it gave in), peers' tokens left out of the edit
        tgt, tforms = (self.peer, self.pforms) if (self.cond in ("disagree", "copy") and said_k == self.peer) else (self.right, self.rforms)
        tidx = 1 if tgt == self.peer and tgt != self.right else 0
        base_t = (pr0, pw0)[tidx]
        onset_prof = []
        for w in c.wins:
            p = self.probs(self.rm(self.dirs(tgt, tforms), w, self.onpos))[tidx]
            onset_prof.append(c.drop(base_t, p))
        oi = next((i for i, d in enumerate(onset_prof) if d >= 0.5), None)
        if oi is None and SKIP:   # testing on the tiny model only: pretend the answer forms mid-way
            oi = len(c.wins) - 1
        onset = c.wins[oi][0] if oi is not None else None
        early = [i for i, w in enumerate(c.wins) if onset is not None and w[-1] < onset]
        rec = dict(question=self.qid, calibration=self.qid in CALIBRATION, condition=self.cond, peers=self.k,
                   right=self.right, peer_letter=self.peer if self.cond != "alone" else None, model_said=said,
                   said=said_k, state=state, P_right=pr0, P_peer=pw0, LO=lo0, answer_target=tgt, onset=onset,
                   onset_profile=" ".join(f"{d:.2f}" for d in onset_prof), n_early=len(early), bridge=b[0])
        if not early:
            rec.update(label="no early windows")
            return rec, t0
        same = [w for w in S["same_kind"] if c.one_token(w)]
        nulls = self.pr.nulls()
        edits = {"bridge": lambda w: self.rm(b, w, self.allpos),
                 "answer": lambda w: self.rm(self.dirs(self.right, self.rforms), w, self.onpos)}
        for s_ in same:
            edits[f"same kind: {s_}"] = lambda w, s_=s_: self.rm([s_], w, self.allpos)
        for z in nulls:
            edits[f"noise: {z}"] = lambda w, z=z: self.rm([z], w, self.allpos)
        best = {}
        for label, make in edits.items():
            for i in early:
                w = c.wins[i]
                r, wr, lo = self.probs(make(w))
                drop = c.drop(pr0, r)
                rows.append(dict(question=self.qid, condition=self.cond, peers=self.k, edit=label, window=lay(w),
                                 start=w[0], P_right=r, P_peer=wr, LO=lo, dLO=lo - lo0, drop_right=drop))
                cur = best.setdefault(label, dict(dLO=-1e9, drop=-1.0))
                if lo - lo0 > cur["dLO"]:
                    cur.update(dLO=lo - lo0, dLO_window=lay(w))
                if drop > cur["drop"]:
                    cur.update(drop=drop, drop_window=lay(w), P_right_after=r)
        gb = lambda pref, key: max((v[key] for k_, v in best.items() if k_.startswith(pref)), default=None)  # noqa: E731
        bd = best["bridge"]
        noise_dlo, noise_drop = gb("noise", "dLO"), gb("noise", "drop")
        kind_dlo, kind_drop = gb("same kind", "dLO"), gb("same kind", "drop")
        need = bd["drop"] - noise_drop
        # reach in this prompt, over early windows
        SP = (" {}", " **{}")
        from consulted_bridges.text import chat
        tn = chat(h, S["user"] + S["name_ask"], S["name_prefill"])
        nb = c.P(tn, S["name_forms"], formats=SP)
        rdrop = max(c.drop(nb, c.P(tn, S["name_forms"], self.rm(b, c.wins[i], c.ALL), formats=SP)) for i in early)
        readable = nb >= TH["readable"]
        reach_ok = readable and rdrop >= TH["reach"]
        # presence
        ctrl = next(w for w in UNRELATED if w not in b and w not in S["same_kind"])
        words = [b[0]] + same[:1] + [ctrl]
        ro = readout(h, c.text, words, lenses=["jlens"], positions=[-1])
        band = h.band
        last = ro.ranks[0, -1]                       # [layer, word]
        pres = {w: int(min(last[l, j] for l in band if last[l, j] >= 0)) if (last[band, j] >= 0).any() else None
                for j, w in enumerate(words)}
        rank_b = pres[b[0]]
        # labels
        if self.cond in ("disagree", "copy"):
            pull = bd["dLO"] >= noise_dlo + TH["pull_noise"] and (kind_dlo is None or bd["dLO"] >= kind_dlo + TH["pull_kind"])
            none_ = bd["dLO"] < noise_dlo + TH["none_noise"] and reach_ok
            label = "bridge still pulls" if pull else "bridge not needed" if none_ else "mixed"
        else:
            label = ("bridge still needed" if need >= TH["need"] else
                     "bridge not needed" if need < TH["none_need"] and reach_ok else "mixed")
        rec.update(bridge_dLO=bd["dLO"], bridge_dLO_window=bd.get("dLO_window"), noise_dLO=noise_dlo,
                   same_kind_dLO=kind_dlo, answer_dLO=best["answer"]["dLO"], pushback=bd["dLO"] - noise_dlo,
                   bridge_drop=bd["drop"], bridge_drop_window=bd.get("drop_window"), bridge_P_right_after=bd.get("P_right_after"),
                   noise_drop=noise_drop, same_kind_drop=kind_drop, answer_drop=best["answer"]["drop"], need=need,
                   reach_P_before=nb, reach_drop=rdrop, reach_readable=readable, reach_pass=reach_ok,
                   presence_rank=rank_b, present=rank_b is not None and rank_b < TH["present"],
                   presence_controls=";".join(f"{w}={pres[w]}" for w in words[1:]), label=label)
        return rec, t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen36_27b")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (testing)")
    ap.add_argument("--questions", nargs="*", default=None)
    ap.add_argument("--frame", default="neutral", choices=list(FRAMES))
    ap.add_argument("--set", default="letters", choices=["letters", "counts"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = Path(args.config).stem
    i, n = map(int, args.shard.split("/"))
    qs = args.questions or (COUNTS[cfg] if args.set == "counts" else questions_for(cfg))
    qs = qs[: args.limit] if args.limit else qs
    qs = [q for j, q in enumerate(qs) if j % n == i]
    tag = "conformity" + ("_counts" if args.set == "counts" else "")
    out = Path(args.out or (f"results/{tag}_{cfg}" if args.frame == "neutral" else f"results/{tag}_{args.frame}_{cfg}")) / f"shard{i}"
    out.mkdir(parents=True, exist_ok=True)
    h = load(args.config)
    P = [(q, *p) for q in qs for p in prompts(q, args.frame)]
    summ, rows = [], []
    for j, (q, cond, k, peer, spec) in enumerate(P):
        print(f"PROGRESS {j}/{len(P)} {q} {cond}{k or ''}", flush=True)
        try:
            pm = Prompt(h, q, cond, k, peer, spec)
            rec, t0 = pm.run(rows)
            summ.append(rec)
            print(f"[{q} {cond}{k or ''}] says {rec['model_said']!r} ({rec['state']}) P(right) {rec['P_right']:.2f} "
                  f"P(peer) {rec['P_peer']:.2f} onset {rec['onset']} | need {rec.get('need', float('nan')):.2f} "
                  f"bridge dLO {rec.get('bridge_dLO', float('nan')):+.2f} noise {rec.get('noise_dLO', float('nan')):+.2f} "
                  f"kind {rec.get('same_kind_dLO')} | reach {rec.get('reach_drop', float('nan')):.2f} "
                  f"rank {rec.get('presence_rank')} | {rec['label']} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as err:
            import traceback
            traceback.print_exc()
            print(f"!! {q} {cond}{k} failed: {err!r}", flush=True)
        pd.DataFrame(summ).to_csv(out / "prompts.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "windows.csv", index=False)
    print("PROGRESS ALLDONE", flush=True)


if __name__ == "__main__":
    main()
