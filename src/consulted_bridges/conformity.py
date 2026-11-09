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

--variants adds conditions (docs/conformity/plan.md B1; none changes the conditions above):
  hidden_k           peers write "?" (one token, so positions line up with the letter prompts)      k = 1, 2, 3
  hidden_sub_k       peers write "(answer submitted)"                                                k = 1, 2, 3
  disagree_g_k       peers give a SECOND wrong letter G (wrong_letter2), for the F-versus-G contrast  k = 1, 2, 3
  told_copy_k3       3 wrong peers, instruction "Repeat Agent 2's answer."
  told_ignore_k3     3 wrong peers, instruction "Ignore the other agents and answer on your own."
  peer_bridge_k      peers state the bridge, not the answer: "The country is Italy."                 k = 1, 3
  peer_bridge_wrong_k  peers state a same-kind word with another letter: "The country is Spain."     k = 1, 3
Disagree-like conditions (DISAGREE_LIKE) get holds / give-in states and pull labels; the rest get need labels.
--own-part adds the removal of the bridge's own part (bridge minus its same-kind words, ablate_excl) to every
early window, recorded as own_drop / own_dLO (B4).

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

from consulted_bridges.interventions import Edit, answer_probs, answer_probs_many
from consulted_bridges.legacy.bridge_card3 import FORMATS, SKIP
from consulted_bridges.load import load
from consulted_bridges.questions.core import SPECS as CORE
from consulted_bridges.questions.proposed.similar import SPECS as VARIANTS
from consulted_bridges.questions.proposed.trend import SPECS as TREND
from consulted_bridges.questions.proposed.counts import SPECS as COUNT_SPECS
from consulted_bridges.questions.proposed.candidates import SPECS as CANDIDATES
from consulted_bridges.readout import readout
from consulted_bridges.silent_dissent.giveup import Probe, lay
from consulted_bridges.text import offsets, resolve_positions

SPECS = {**CORE, **VARIANTS, **TREND, **COUNT_SPECS, **CANDIDATES}
EPS = 1e-6
BATCH = int(__import__('os').environ.get('CB_BATCH', '8'))   # edited copies of a prompt per forward pass
SUFFIX = " Answer with one letter."
ONE = " Answer in one word."
NUM1 = " Answer with a single number."     # the v2 number candidates (candidates.py)


def suffix_of(user):
    """The answer-format line a question ends with (letter, number or one word)."""
    return next((x for x in (SUFFIX, NUM1) if user.endswith(x)), ONE)

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
# The v2 candidates (questions/proposed/candidates.py), unscreened. Kept apart from LETTER so the existing screens
# and conformity_select keep their 37-question list; the candidate screen runs them with `--sets candidates`.
CANDIDATE = sorted(CANDIDATES)


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
COUNT_WRONG.update({"8": "7", "7": "6", "4": "3", "3": "2", "1": "2"})   # same rule for other models' counts (1 -> 2: no zero)


def counts_for(cfg):
    """A model's count questions: the fixed list above, else the ones its own trimmed count screen
    (results/counts*_<config>/shard*/cards.csv) calls consulted, the rule that made the 27B list."""
    if cfg in COUNTS:
        return COUNTS[cfg]
    fs = [f for f in sorted(Path("results").glob(f"counts*_{cfg}/shard*/cards.csv")) if f.stat().st_size > 1]
    if not fs:
        raise SystemExit(f"no count screen for {cfg}: run the scorecard on the count questions first")
    t = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return sorted(set(t[t.verdict.fillna("").str.startswith("CONSULTED")].sentence))
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


# The peers' wrong answer. Two rules (scope.md §11, run 3):
#   "kind"  (runs 1-2, the development runs): the first same-kind or swap word's letter whose letter differs from the
#           right one (Spain's S for the Colosseum). Removing the bridge can send the model to exactly that letter,
#           so D's destination in the 2x2 could not be read.
#   "free"  (from run 3 on, the default): letters - a letter that no same-kind or swap word starts with and that
#           differs from the right letter, chosen by a hash of the question id (fixed, no model involved); numbers -
#           the nearest single digit that is neither the right answer nor a same-kind word's answer (the old one-off
#           choice first); categories (yes/no, left/right, even/odd, continents) - the other answer.
# The pilot's six calibration questions keep their fixed letters under both rules.
# CB_WRONG_RULE=kind (or --wrong-rule kind on the drivers) reproduces the earlier runs exactly.
WRONG_RULE = __import__("os").environ.get("CB_WRONG_RULE", "free")
WRONG_RULES = ("free", "kind")
FREE_LETTERS = "ABCDEFGHIJKLMNOPRSTUVW"          # Q, X, Y, Z left out: implausible first letters for these kinds
# Answers of the same-kind words in the number families (candidates.py; standard facts), for the "free" rule.
NUM_FACTS = {**{m: str(i + 1) for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                                      "August", "September", "October", "November", "December"])},
             **{p: str(i + 1) for i, p in enumerate(["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
                                                     "Uranus", "Neptune"])},
             "hydrogen": "1", "helium": "2", "lithium": "3", "beryllium": "4", "boron": "5", "carbon": "6",
             "nitrogen": "7", "oxygen": "8", "fluorine": "9", "neon": "10",
             "guitar": "6", "violin": "4", "cello": "4", "viola": "4", "banjo": "5", "ukulele": "4", "mandolin": "8",
             "baseball": "9", "basketball": "5", "hockey": "6", "volleyball": "6", "handball": "7", "polo": "4",
             "soccer": "11", "football": "11", "rugby": "15", "cricket": "11"}


def answer_format(spec):
    """'letter', 'number' or 'category', from the answer keys."""
    keys = list(spec["answers"])
    if all(len(k) == 1 and k.isalpha() for k in keys):
        return "letter"
    if spec["answer"].isdigit():
        return "number"
    return "category"


def _kind_initials(spec):
    return {w[0].upper() for w in spec["same_kind"] + spec.get("swaps", []) if w[:1].isalpha()}


def _free_letters(qid, spec, n=2):
    import hashlib
    bad = _kind_initials(spec) | {spec["answer"]}
    pool = [L for L in FREE_LETTERS if L not in bad]
    i = int(hashlib.sha256(qid.encode()).hexdigest(), 16) % len(pool)
    return [pool[(i + j) % len(pool)] for j in range(n)]


def _free_numbers(spec, n=2):
    right = spec["answer"]
    bad = {right} | {NUM_FACTS[w] for w in spec["same_kind"] if w in NUM_FACTS}
    first = COUNT_WRONG.get(right)
    r = int(right)
    order = ([first] if first else []) + [str(r + d) for k in range(1, 9) for d in (k, -k)]
    out = []
    for x in order:
        if x not in out and x not in bad and x.isdigit() and 1 <= int(x) <= 9 and x in spec["answers"]:
            out.append(x)
    return out[:n]


def _free_categories(spec, n=2):
    return [k for k in spec["answers"] if k != spec["answer"]][:n]


def free_wrong(qid, spec, n=2):
    f = answer_format(spec)
    return (_free_letters(qid, spec, n) if f == "letter" else _free_numbers(spec, n) if f == "number"
            else _free_categories(spec, n))


def wrong_letter(qid, spec, rule=None):
    """The peers' wrong answer under WRONG_RULE (see above)."""
    rule = rule or WRONG_RULE
    if qid in PILOT_WRONG:
        return PILOT_WRONG[qid]
    if rule == "free":
        return free_wrong(qid, spec)[0]
    if spec["answer"] in COUNT_WRONG and not spec["user"].endswith(SUFFIX):
        return COUNT_WRONG[spec["answer"]]
    for w in spec["same_kind"] + spec.get("swaps", []):
        if w[0].upper() != spec["answer"] and w[0].isalpha():
            return w[0].upper()
    return "Z" if spec["answer"] != "Z" else "A"


FRAMES = {"neutral": "You are Agent 1. Give your own answer.",
          # the pilot's wording, which invites deferring to the peers
          "defer": "You are Agent 1. Taking the other agents' answers into account, give your answer.",
          # instructions for the told_copy / told_ignore conditions (not run-wide frames)
          "copy": "You are Agent 1. Repeat Agent 2's answer.",
          "ignore": "You are Agent 1. Ignore the other agents and answer on your own."}
RUN_FRAMES = ["neutral", "defer"]
HIDDEN = {"hidden": "?", "hidden_sub": "(answer submitted)"}
VARIANTS = ["hidden", "hidden_sub", "disagree_g", "told_copy", "told_ignore", "peer_bridge", "peer_bridge_wrong"]
# conditions read like disagree: the model may give in to the peers' letter; pull labels
DISAGREE_LIKE = {"disagree", "copy", "disagree_g", "told_copy", "told_ignore", "peer_bridge_wrong"}
OWN_PART = False    # --own-part: also remove the bridge's own part (bridge minus same-kind words) in every early window


def wrong_letter2(qid, spec, rule=None):
    """A second wrong letter, different from the right letter and from wrong_letter(): the F-versus-G contrast."""
    rule = rule or WRONG_RULE
    right, wrong = spec["answer"], wrong_letter(qid, spec, rule)
    if rule == "free" and qid not in PILOT_WRONG:
        return next(x for x in free_wrong(qid, spec, 3) if x not in (right, wrong))
    for w in spec["same_kind"] + spec.get("swaps", []):
        L = w[0].upper()
        if L.isalpha() and L not in (right, wrong):
            return L
    return next(L for L in "GKQXZJ" if L not in (right, wrong))


def swap_target(spec):
    """(word, its answer): the first same-kind word whose answer differs from the right one. Letters: its first
    letter; numbers: NUM_FACTS; categories: none (None, None). The 2x2 records P(this answer) as D's other
    destination, and the swap test (conformity_swap) moves the bridge onto this word."""
    f = answer_format(spec)
    for w in spec["same_kind"]:
        a = w[0].upper() if f == "letter" else NUM_FACTS.get(w) if f == "number" else None
        if a and a != spec["answer"] and a in spec["answers"]:
            return w, a
    return None, None


def kind_word(spec):
    """'country' from name_ask ' First name the country.'"""
    return spec["name_ask"].strip().rstrip(".").split()[-1]


def group_user(question, peer, k, frame="neutral", suffix=SUFFIX):
    return ("Several AI agents are answering the same question.\n\n" f"Question: {question}\n" +
            "".join(f"Agent {i}: {peer}\n" for i in range(2, 2 + k)) + "\n" + FRAMES[frame] + suffix)


def prompts(qid, frame="neutral", extra=()):
    """(condition, peers, peer string, spec) for one question. `extra` adds VARIANTS; with none, the output is
    exactly the original run's."""
    s = SPECS[qid]
    sfx = suffix_of(s["user"])   # letter questions, v2 number questions, or count questions ("one word")
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
    for v in extra:
        if v not in VARIANTS:
            raise ValueError(f"unknown variant {v!r}")
        if v in HIDDEN:
            for k in (1, 2, 3):
                out.append((v, k, HIDDEN[v], dict(s, user=group_user(q, HIDDEN[v], k, frame, sfx))))
        elif v == "disagree_g":
            g = wrong_letter2(qid, s)
            for k in (1, 2, 3):
                out.append((v, k, g, dict(s, user=group_user(q, g, k, frame, sfx))))
        elif v in ("told_copy", "told_ignore"):
            out.append((v, 3, wrong, dict(s, user=group_user(q, wrong, 3, v.split("_")[1], sfx))))
        elif v in ("peer_bridge", "peer_bridge_wrong"):
            word = s["bridge"][0] if v == "peer_bridge" else s["same_kind"][0]
            sent = f"The {kind_word(s)} is {word}."
            for k in (1, 3):
                out.append((v, k, sent, dict(s, user=group_user(q, sent, k, frame, sfx), peer_word=word)))
    return out


def peer_positions(h, text, peer, k):
    """Token positions of the peers' answers ("Agent i: X", the whole answer, however many words)."""
    offs = offsets(h, text)
    pos = []
    for m in re.finditer(r"Agent \d: (.+)\n", text):
        a, b = m.span(1)
        pos += [i for i, (x, y) in enumerate(offs) if x < b and y > a]
    return sorted(set(pos))


def reach_positions(h, tn, user, qspec):
    """Positions of the naming prompt `tn` that it shares with the original prompt: from the question's first word
    (`qspec`, Card.ALL) to the last token inside the user text `user`, nothing of the added instruction after it."""
    at = tn.rfind(user)
    if at < 0:
        raise ValueError("user text not found in the naming prompt")
    end = at + len(user)
    offs = offsets(h, tn)
    return [p for p in resolve_positions(h, tn, qspec) if offs[p][1] <= end and offs[p][1] > 0]


class Prompt:
    def __init__(self, h, qid, cond, k, peer, spec):
        self.h, self.qid, self.cond, self.k, self.peer, self.spec = h, qid, cond, k, peer, spec
        self.pr = Probe(h, f"{qid}__{cond}{k or ''}", spec, peer)
        self.c = self.pr.c
        self.right = spec["answer"]
        self.rforms = spec["answers"][self.right]
        # the peers' letter: for peer_bridge(_wrong) the first letter of the word they name; for hidden peers none
        self.hidden = cond in HIDDEN
        self.pletter = spec["peer_word"][0].upper() if "peer_word" in spec else peer
        self.pforms = [] if self.hidden else spec["answers"].get(self.pletter, [self.pletter])
        allpos = resolve_positions(h, self.c.text, self.c.ALL)
        skip = set(peer_positions(h, self.c.text, peer, k)) if k else set()
        self.allpos, self.onpos = allpos, [p for p in allpos if p not in skip]

    @property
    def dis(self):
        return self.cond in DISAGREE_LIKE

    def probs(self, edits=None):
        """P(right), P(peers'), LO in one forward pass."""
        cands = list(dict.fromkeys(self.rforms + self.pforms))
        return self._summ(answer_probs(self.h, self.c.text, cands, edits, formats=FORMATS)["cands"])

    def probs_many(self, edit_sets):
        """probs() for each list of edits, several per forward pass (answer_probs_many)."""
        cands = list(dict.fromkeys(self.rforms + self.pforms))
        return [self._summ(r["cands"]) for r in answer_probs_many(self.h, self.c.text, cands, edit_sets, formats=FORMATS,
                                                                  chunk=BATCH)]

    def probs_x(self, edits, answers):
        """P(right), P(peers') and P(each answer in `answers`), one forward pass (Block and Patch allowed)."""
        forms = {a: self.spec["answers"].get(a, [a]) for a in answers if a}
        cands = list(dict.fromkeys(self.rforms + self.pforms + [f for v in forms.values() for f in v]))
        r = answer_probs(self.h, self.c.text, cands, edits, formats=FORMATS)["cands"]
        tot = lambda fs: sum(r[f].get(f"p[{x}]", 0.0) for f in fs for x in FORMATS)  # noqa: E731
        pr, pw, _ = self._summ(r)
        return pr, pw, {a: tot(v) for a, v in forms.items()}

    def _summ(self, r):
        tot = lambda forms: sum(r[f].get(f"p[{x}]", 0.0) for f in forms for x in FORMATS)  # noqa: E731
        pr, pw = tot(self.rforms), tot(self.pforms)
        if self.hidden:                               # no peers' answer to score
            return pr, float("nan"), float("nan")
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
        if self.dis:
            state = ("holds" if said_k == self.right else
                     ("partial give-in" if pr0 >= 0.3 else "full give-in") if said_k == self.pletter else "other")
        else:
            state = "right" if said_k == self.right else "wrong"
        # onset: the letter the model gives (the peers' if it gave in), peers' tokens left out of the edit
        gave_in = self.dis and said_k == self.pletter
        tgt, tforms = (self.pletter, self.pforms) if gave_in else (self.right, self.rforms)
        tidx = 1 if gave_in and tgt != self.right else 0
        base_t = (pr0, pw0)[tidx]
        onset_prof = [c.drop(base_t, ps[tidx]) for ps in
                      self.probs_many([self.rm(self.dirs(tgt, tforms), w, self.onpos) for w in c.wins])]
        oi = next((i for i, d in enumerate(onset_prof) if d >= 0.5), None)
        if oi is None and SKIP:   # testing on the tiny model only: pretend the answer forms mid-way
            oi = len(c.wins) - 1
        onset = c.wins[oi][0] if oi is not None else None
        early = [i for i, w in enumerate(c.wins) if onset is not None and w[-1] < onset]
        rec = dict(question=self.qid, calibration=self.qid in CALIBRATION, condition=self.cond, peers=self.k,
                   right=self.right, peer_letter=(None if self.cond == "alone" or self.hidden else self.pletter),
                   peer_text=self.peer if self.cond != "alone" else None, model_said=said,
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
        if OWN_PART and same:        # B4: the bridge minus what it shares with its same-kind words
            edits["bridge own part"] = lambda w: [Edit("ablate_excl", b + same, w, self.allpos)]
        best = {}
        jobs = [(label, make, i) for label, make in edits.items() for i in early]
        res = self.probs_many([make(c.wins[i]) for _, make, i in jobs])
        for (label, make, i), (r, wr, lo) in zip(jobs, res):
            w = c.wins[i]
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
        own = best.get("bridge own part", {})
        need = bd["drop"] - noise_drop
        # reach in this prompt, over early windows
        SP = (" {}", " **{}")
        from consulted_bridges.text import chat
        tn = chat(h, S["user"] + S["name_ask"], S["name_prefill"])
        nb = c.P(tn, S["name_forms"], formats=SP)
        shared = reach_positions(h, tn, S["user"], c.ALL)
        # scope.md §9 fix (run 3 on): the edit covers only the positions the naming prompt shares with the original
        # prompt (the question's first word to the end of the user text), not the added "First name the ..." or the
        # started reply, where the name is being produced. The old q->end edit is kept as reach_drop_qend.
        rdrop = max(c.drop(nb, c.P(tn, S["name_forms"], self.rm(b, c.wins[i], shared), formats=SP)) for i in early)
        rdrop_qend = max(c.drop(nb, c.P(tn, S["name_forms"], self.rm(b, c.wins[i], c.ALL), formats=SP)) for i in early)
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
        if self.dis:
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
                   own_drop=own.get("drop"), own_dLO=own.get("dLO"),
                   reach_P_before=nb, reach_drop=rdrop, reach_drop_qend=rdrop_qend, reach_positions="shared",
                   reach_readable=readable, reach_pass=reach_ok, wrong_rule=WRONG_RULE,
                   presence_rank=rank_b, present=rank_b is not None and rank_b < TH["present"],
                   presence_controls=";".join(f"{w}={pres[w]}" for w in words[1:]), label=label)
        return rec, t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen36_27b")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None, help="first N questions only (testing)")
    ap.add_argument("--questions", nargs="*", default=None)
    ap.add_argument("--frame", default="neutral", choices=RUN_FRAMES)
    ap.add_argument("--set", default="letters", choices=["letters", "counts"])
    ap.add_argument("--variants", nargs="*", default=[], choices=VARIANTS, help="extra conditions (B1)")
    ap.add_argument("--own-part", action="store_true", help="also remove the bridge's own part (B4)")
    ap.add_argument("--wrong-rule", default=None, choices=WRONG_RULES,
                    help="the peers' wrong answer: free (default, run 3 on) or kind (runs 1-2)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global OWN_PART, WRONG_RULE
    OWN_PART = args.own_part
    WRONG_RULE = args.wrong_rule or WRONG_RULE
    cfg = Path(args.config).stem
    i, n = map(int, args.shard.split("/"))
    qs = args.questions or (counts_for(cfg) if args.set == "counts" else questions_for(cfg))
    qs = qs[: args.limit] if args.limit else qs
    qs = [q for j, q in enumerate(qs) if j % n == i]
    tag = "conformity" + ("_counts" if args.set == "counts" else "")
    out = Path(args.out or (f"results/{tag}_{cfg}" if args.frame == "neutral" else f"results/{tag}_{args.frame}_{cfg}")) / f"shard{i}"
    out.mkdir(parents=True, exist_ok=True)
    h = load(args.config)
    P = [(q, *p) for q in qs for p in prompts(q, args.frame, tuple(args.variants))]
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
