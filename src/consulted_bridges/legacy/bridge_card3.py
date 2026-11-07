"""Bridge scorecard v3 (round 3). Same as v2 except: gate 4 checks reach IN THE SAME PROMPT
(the question asked to name the bridge first), collateral damage is reported only, and each card gets a
QUALITY score: reach-gated need x specificity x isolation (see docs/scorecard/v3_v4.md).

v2 notes follow.
 Does the model consult a hidden middle step (the bridge) to get the answer?

Revised after the review (docs/reviews/review_scorecard.md). Definitions are fixed BEFORE running new sentences.
P = the model's probability of the right answer (summed over its written forms, plain or **bold**).
drop = 1 - P_after / P_before. Edits remove a word's J-lens direction from the residual stream, from
the question's first word to the end of the prompt, in 6-layer windows (L0-5, L3-8, ..., L27-31).

Gates (if one fails, the sentence is UNTESTABLE, and the card says which):
  G1 knows it      the model answers the two-hop question right, and both one-hop questions too.
  G2 clean words   every edited word is a single token (no generic pieces like " F" for Funchal).
  G3 answer onset  removing the answer's own direction kills P in some window. Its first such window is
                   the ANSWER ONSET: from there on the model is writing the answer.
  G4 edit reaches  in the one-hop question whose answer IS the bridge, removing the bridge's direction
     the bridge    (in a window that ends before the answer onset) drops P(bridge) by >= 0.5.
  If the answer onset is in the first window, the answer is present from the start: "ANSWER FROM THE START".

Tests (all measured in windows that END BEFORE the answer onset, "early windows"):
  1 needed early    best drop from removing the bridge in an early window.
  2 beats noise     needed early minus the largest early drop from 12 everyday words (table, chair...)
                    of matching strength. Pass >= 0.5.
  3 comes first     the bridge's first window with drop >= 0.5 starts >= 3 layers before the answer onset.
  4 beats own kind  needed early minus the largest early drop from removing words of the bridge's kind
                    that share its answer (Vienna for Salzburg; tick for spider). Pass >= 0.3.
  5 no collateral   the same edit on two unrelated questions (gold -> Au, largest planet -> Jupiter)
                    drops their answers by <= 0.1.
  Reported, not in the verdict:
  6 spread out      drop when removing the bridge over layers 8-23, and over all layers.
  7 own part        best early drop removing only the part of the bridge NOT shared with the answer.
  8 swaps follow    swap the bridge for another word over the early layers: share of swaps that give that
                    word's own answer (as the model states it), not counting the fallback answer.
  9 same-answer     share of swaps to a word with the same answer that keep it.

Verdicts:
  CONSULTED BRIDGE   gates pass, beats noise >= 0.5, comes first, no collateral, beats own kind >= 0.3
                     (or no same-kind word available: "consulted (kind not checked)").
  CATEGORY EFFECT    as consulted, but words of its kind do the same (beats own kind < 0.3).
  TANGLED            beats noise >= 0.5, but it doesn't come first (it acts with the answer, not before).
  SPREAD OUT         beats noise < 0.5, but removing over 16 or all layers drops >= 0.5.
  BYSTANDER          beats noise < 0.2 and the 16-layer and all-layer drops < 0.2.
  UNCLEAR            anything else.

  uv run python scripts/bridge_card2.py [--spec E_spider ...] [--shard 0/3] [--config qwen35_4b]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch


from consulted_bridges.questions.core import SPECS  # noqa: E402
from consulted_bridges.interventions import Edit, answer_probs, directions, generate  # noqa: E402
from consulted_bridges.load import load  # noqa: E402
from consulted_bridges.text import chat, resolve_positions  # noqa: E402
from silent_dissent.model import get_layers, layer_output  # noqa: E402

FORMATS = ("{}", "**{}")
NULL_POOL = ("table river window garden pencil cloud bottle chair mountain jacket candle ladder carpet button "
             "kettle wallet pillow mirror basket hammer ribbon saddle lantern blanket ticket helmet barrel kitchen "
             "rocket shoe door bread spoon brick").split()
UNRELATED = [("What is the chemical symbol for gold? Answer in one word.", ["Au"]),
             ("What is the largest planet in the Solar System? Answer in one word.", ["Jupiter"])]
SKIP = bool(os.environ.get("CARD2_SKIP_GATES"))   # testing only: run every step on the tiny model
TH = dict(beats_noise=0.5, comes_first=3, beats_kind=0.3, collateral=0.1, gate=0.5, bystander=0.2, spread=0.5)


def fmt_layers(ls):
    return f"layers {ls[0]}–{ls[-1]}" if len(ls) > 1 else f"layer {ls[0]}"


def q(w):
    return f"“{w}”"


class Card:
    def __init__(self, h, name, S, verbose=True):
        self.h, self.name, self.S, self.say = h, name, S, (print if verbose else (lambda *a, **k: None))
        self.L = h.n_layers
        # layer numbers below are for a 32-layer model; sc() scales them to this model's depth
        self.f = self.L / 32
        sc = self.sc = lambda x: max(1, round(x * self.f))
        self.wins = [list(range(s, min(s + sc(6), self.L))) for s in range(0, self.L - sc(3), sc(3))]
        self.text = chat(h, S["user"], "")
        self.ALL = f"from:{S['q_start']}"
        self.flat = {f: k for k, forms in S["answers"].items() for f in forms}

    # --- helpers -------------------------------------------------------------------------------
    def one_token(self, w):
        return SKIP or len(self.h.encode(" " + w)) == 1

    def P(self, text, forms, edits=None, formats=FORMATS):
        r = answer_probs(self.h, text, forms, edits, formats=formats)
        key = lambda f: "p_plain" if f == " {}" else "p_bold" if f == " **{}" else f"p[{f}]"  # answer_probs' own names
        return sum(v.get(key(f), 0.0) for v in r["cands"].values() for f in formats)

    def gen(self, text, edits=None):
        return generate(self.h, text, edits, max_new_tokens=6).replace("<|im_end|>", "").strip()

    @staticmethod
    def match(ans, forms):
        a = re.sub(r"[*_`\"“”]", "", ans).strip(" .\n").lower()
        for f in sorted(forms, key=len, reverse=True):
            if re.match(rf"{re.escape(f.lower())}\b", a):
                return True
        return False

    def classify(self, ans, answers=None):
        answers = answers or self.S["answers"]
        for k, forms in answers.items():
            if self.match(ans, forms):
                return k
        return "other"

    def rm(self, words, layers, pos=None):
        return [Edit("ablate", words, layers, pos or self.ALL)]

    def drop(self, base, p):
        return max(0.0, 1 - p / base) if base > 0 else 0.0

    @torch.no_grad()
    def strength(self, store, word, layers, pos):
        v = []
        for l in layers:
            D = directions(self.h, Edit("ablate", [word], [l]), l)[0].to(store[l].device)
            H = store[l][pos].float()
            v.append(((H @ D.T).abs().max(-1).values / H.norm(dim=-1)).max().item())
        return sum(v) / len(v)

    @torch.no_grad()
    def capture(self):
        store = {}
        hs = [b.register_forward_hook(lambda _m, _i, o, l=l: store.__setitem__(l, layer_output(o)[0].detach()))
              for l, b in enumerate(get_layers(self.h.model))]
        try:
            self.h.model(input_ids=torch.tensor([self.h.encode(self.text)], device=self.h.device))
        finally:
            for x in hs:
                x.remove()
        return store

    # --- the card ------------------------------------------------------------------------------
    def run(self):
        S, h, text = self.S, self.h, self.text
        ans_key, ans_forms = S["answer"], S["answers"][S["answer"]]
        # answer directions: the given words plus the raw first token of each written form ("S", "Eight")
        S = dict(S, answer_dirs=list(dict.fromkeys(S["answer_dirs"] + ["@" + f for f in ans_forms])))
        b = S["bridge"]
        A = q(ans_forms[0])
        res = dict(sentence=self.name, group=S["group"], predicted=S["predict"], question=S["user"],
                   bridge=" / ".join(b), answer=ans_forms[0], hop1_question=S["hop1"], hop2_question=S["hop2"])
        tests, swaps, controls, profile = [], [], [], []

        def test(key, label, asks, value, result, what):
            tests.append(dict(key=f"{self.name}|{key}", sentence=self.name, order=len(tests), test=label, asks=asks,
                              value=None if value is None else round(value, 3), result=result, what_happened=what))

        def finish(verdict, why):
            res.update(verdict=verdict, verdict_reason=why)
            self.say(f"[{self.name}] {verdict}: {why}")
            return res, tests, swaps, controls, profile

        # G1: knows it
        base = self.P(text, ans_forms)
        said = self.gen(text)
        res.update(model_said=said, P_base=round(base, 3))
        h1 = chat(h, S["hop1"], "")
        h1_said = self.gen(h1)
        h2 = chat(h, S["hop2"], "")
        h2_said = self.gen(h2)
        k0, k1, k2 = self.classify(said) == ans_key, self.match(h1_said, S["hop1_forms"]), self.classify(h2_said) == ans_key
        res.update(hop1_said=h1_said, hop2_said=h2_said)
        ok = k0 and k1 and k2
        test("g1", "Gate 1 · knows it", "Does the model answer the question, and both one-hop questions, right?",
             None, "pass" if ok else "fail",
             f"Two-hop: says {q(said)} (right answer {A}, P = {base:.2f}). "
             f"Hop 1 {q(S['hop1'])} → {q(h1_said)}. Hop 2 {q(S['hop2'])} → {q(h2_said)}.")
        if not ok and not SKIP:
            which = [n for n, k in [("the two-hop question", k0), ("hop 1", k1), ("hop 2", k2)] if not k]
            return finish("UNTESTABLE", "the model gets " + " and ".join(which) + " wrong")

        # G2: clean words
        bad = [w for w in b if not self.one_token(w)]
        test("g2", "Gate 2 · clean words", "Is every edited word a single token?", None, "fail" if bad else "pass",
             ("Bridge words split into pieces: " + ", ".join(bad)) if bad else
             f"{', '.join(q(w) for w in b)} are single tokens.")
        if bad and not SKIP:
            return finish("UNTESTABLE", "the bridge word is not a single token")
        same_kind = [w for w in S["same_kind"] if self.one_token(w)]
        sw_words = [w for w in S["swaps"] if self.one_token(w)]
        dropped = [w for w in S["same_kind"] + S["swaps"] if not self.one_token(w)]

        # profiles: bridge and answer, every window
        pb = [self.drop(base, self.P(text, ans_forms, self.rm(b, w))) for w in self.wins]
        pa = [self.drop(base, self.P(text, ans_forms, self.rm(S["answer_dirs"], w))) for w in self.wins]
        # answer onset: the first of two consecutive windows that both kill the answer (or the last window),
        # so one noisy early window can't set it
        onset_i = next((i for i, d in enumerate(pa) if d >= TH["gate"] and (i == len(pa) - 1 or pa[i + 1] >= TH["gate"])), None)
        test("g3", "Gate 3 · answer onset", "Does removing the answer's own direction kill it, and from which layers?",
             max(pa), "pass" if onset_i is not None else "fail",
             (f"Remove the answer {A} itself: first kills it in {fmt_layers(self.wins[onset_i])} "
              f"(P {base:.2f} → {base * (1 - pa[onset_i]):.2f}, drop {pa[onset_i]:.2f}). "
              f"That's where the model starts writing the answer.") if onset_i is not None else
             f"Removing the answer {A} itself never kills it (best drop {max(pa):.2f}), so these edits can't test this question.")
        if onset_i is None and SKIP:
            onset_i = len(self.wins) // 2
        if onset_i is None:
            return finish("UNTESTABLE", "removing the answer's own direction doesn't kill it")
        onset = self.wins[onset_i][0]
        res["answer_onset"] = onset
        early = [i for i, w in enumerate(self.wins) if w[-1] < onset]
        if SKIP and not early:
            early = [0]
        def add_profile(nprof):
            for i, w in enumerate(self.wins):
                profile.append(dict(key=f"{self.name}|{w[0]}", sentence=self.name, window=fmt_layers(w), start=w[0],
                                    bridge_drop=round(pb[i], 3), answer_drop=round(pa[i], 3),
                                    noise_drop=None if nprof is None else round(max(v[i] for v in nprof.values()), 3)))
        if not early:
            add_profile(None)
            return finish("ANSWER FROM THE START", f"removing the answer already kills it in {fmt_layers(self.wins[onset_i])}, "
                          "before any window where the bridge could act first")

        # G4: edit reaches the bridge IN THIS PROMPT: same question, asked to name the bridge first
        SP = (" {}", " **{}")
        tn = chat(h, S["user"] + S["name_ask"], S["name_prefill"])
        named = self.gen(tn)
        nb = self.P(tn, S["name_forms"], formats=SP)
        vd = [(self.drop(nb, self.P(tn, S["name_forms"], self.rm(b, self.wins[i]), formats=SP)), i) for i in early]
        V, vi = max(vd)
        res.update(reach=round(V, 3), named_as=named)
        test("g4", "Gate 4 · edit reaches the bridge",
             "Same question, asked to name the bridge first: does removing the bridge (before the answer onset) kill its name?",
             V, "pass" if V >= TH["gate"] else "fail",
             f"Ask {q(S['user'] + S['name_ask'])} with the reply starting {q(S['name_prefill'])} (model continues: {q(named)}). "
             f"Remove {q(b[0])} in {fmt_layers(self.wins[vi])} → P({q(b[0])}) goes {nb:.2f} → {nb * (1 - V):.2f}. Drop {V:.2f}.")
        # low reach only blocks NEGATIVE verdicts (a bystander could just be an edit that misses);
        # a positive result (answer collapses, matched controls don't) stands on its own
        reach_ok = V >= TH["gate"]

        # null words matched on strength
        store = self.capture()
        allpos = resolve_positions(h, text, self.ALL)
        mid = [l for l in range(self.sc(8), self.sc(21)) if l < self.L] or list(range(self.L))
        sb = self.strength(store, b[0], mid, allpos)
        stn = {w: self.strength(store, w, mid, allpos) for w in NULL_POOL if self.one_token(w)}
        nulls = sorted(stn, key=lambda w: abs(stn[w] - sb))[:12]
        nprof = {w: [self.drop(base, self.P(text, ans_forms, self.rm([w], win))) for win in self.wins] for w in nulls}
        add_profile(nprof)
        for n in nulls:
            e = max(nprof[n][i] for i in early)
            controls.append(dict(key=f"{self.name}|null|{n}", sentence=self.name, word=n, kind="everyday word (noise)",
                                 strength=round(stn[n], 3), early_drop=round(e, 3)))

        # 1 needed early, 2 beats noise
        ne, ni = max((pb[i], i) for i in early)
        noise, nw = max((max(nprof[n][i] for i in early), n) for n in nulls)
        nw_i = max(early, key=lambda i: nprof[nw][i])
        res.update(needed_early=round(ne, 3), noise=round(noise, 3), beats_noise=round(ne - noise, 3))
        test("t1", "1 · Needed early", "Does removing the bridge, before the answer onset, kill the answer?", ne, "info",
             f"Remove {q(b[0])} in {fmt_layers(self.wins[ni])} → P({A}) goes {base:.2f} → {base * (1 - ne):.2f}. Drop {ne:.2f}.")
        test("t2", "2 · Beats noise", "Is that drop bigger than removing everyday words of the same strength?",
             ne - noise, "pass" if ne - noise >= TH["beats_noise"] else "fail",
             f"Biggest drop from 12 everyday words: {noise:.2f} ({q(nw)}, {fmt_layers(self.wins[nw_i])}). "
             f"Bridge {ne:.2f} − noise {noise:.2f} = {ne - noise:.2f} (pass ≥ {TH['beats_noise']}).")

        # 3 comes first
        bon = next((self.wins[i][0] for i, d in enumerate(pb) if d >= TH["gate"]), None)
        first = bon is not None and bon <= onset - self.sc(TH["comes_first"])
        res.update(bridge_onset=bon, comes_first=first)
        test("t3", "3 · Comes first", "Does the bridge start to matter at least 3 layers before the answer does?",
             None if bon is None else onset - bon, "pass" if first else "fail",
             (f"{q(b[0])} first matters from layer {bon}; the answer {A} from layer {onset} → "
              f"{onset - bon} layers earlier.") if bon is not None else
             f"Removing {q(b[0])} never drops P by 0.5 in any window; the answer {A} matters from layer {onset}.")

        # 4 beats own kind
        lay = self.wins[ni]
        kd = {}
        for w in same_kind:
            kd[w] = max(self.drop(base, self.P(text, ans_forms, self.rm([w], self.wins[i]))) for i in early)
            controls.append(dict(key=f"{self.name}|kind|{w}", sentence=self.name, word=w,
                                 kind=f"same kind as {b[0]}, same answer", strength=None, early_drop=round(kd[w], 3)))
        if kd:
            kw = max(kd, key=kd.get)
            bk = ne - kd[kw]
            test("t4", "4 · Beats own kind", "Does removing the bridge matter more than removing another word of the same kind (another city, another animal)?",
                 bk, "pass" if bk >= TH["beats_kind"] else "fail",
                 f"Remove {q(kw)} (same kind) → P({A}) goes {base:.2f} → {base * (1 - kd[kw]):.2f}. Drop {kd[kw]:.2f}. "
                 f"Bridge {ne:.2f} − {kd[kw]:.2f} = {bk:.2f} (pass ≥ {TH['beats_kind']}).")
        else:
            bk = None
            test("t4", "4 · Beats own kind", "Does removing the bridge matter more than removing another word of the same kind (another city, another animal)?",
                 None, "not checked", "No single-token word of the same kind shares this answer.")
        res["beats_kind"] = None if bk is None else round(bk, 3)

        # 5 no collateral
        coll, parts = 0.0, []
        for uq, uforms in UNRELATED:
            ut = chat(h, uq, "")
            u0 = self.P(ut, uforms)
            u1 = self.P(ut, uforms, self.rm(b, lay, "from:" + " ".join(uq.split()[:2])))
            d = self.drop(u0, u1)
            coll = max(coll, d)
            parts.append(f"{q(uq)} → P({q(uforms[0])}) {u0:.2f} → {u1:.2f}")
        res["collateral"] = round(coll, 3)
        test("t5", "5 · Collateral damage (reported)", "Does the same edit leave unrelated questions alone?", coll,
             "info",
             f"Same edit ({fmt_layers(lay)}): " + "; ".join(parts) + f". Biggest drop {coll:.2f} (pass ≤ {TH['collateral']}).")

        # 6 spread out
        r16 = self.drop(base, self.P(text, ans_forms, self.rm(b, [l for l in range(self.sc(8), self.sc(24)) if l < self.L])))
        rall = self.drop(base, self.P(text, ans_forms, self.rm(b, list(range(self.L)))))
        res.update(spread_16=round(r16, 3), spread_all=round(rall, 3))
        test("t6", "6 · Spread out (reported)", "What if the bridge is removed over many layers at once?",
             max(r16, rall), "info",
             f"Remove {q(b[0])} in layers {self.sc(8)}–{self.sc(24) - 1} → P({A}) {base:.2f} → {base * (1 - r16):.2f} (drop {r16:.2f}); "
             f"in all layers → {base * (1 - rall):.2f} (drop {rall:.2f}).")

        # 7 own part
        own = max(self.drop(base, self.P(text, ans_forms, [Edit("ablate_excl", [w] + S["answer_dirs"], self.wins[i], self.ALL)
                                                           for w in b])) for i in early)
        res["own_part"] = round(own, 3)
        if same_kind:
            bonly = max(self.drop(base, self.P(text, ans_forms, [Edit("ablate_excl", [w] + same_kind, self.wins[i], self.ALL)
                                                               for w in b])) for i in early)
        else:
            bonly = None
        res["bridge_only"] = None if bonly is None else round(bonly, 3)
        test("t7b", "7b · Bridge's own part vs its kind (reported)",
             "Remove only the part of the bridge NOT shared with any same-kind word: does the answer still collapse?",
             bonly, "info", (f"Remove {q(b[0])} minus {' + '.join(q(w) for w in same_kind)} (early layers) → P({A}) "
                             f"{base:.2f} → {base * (1 - bonly):.2f}. Drop {bonly:.2f}.") if bonly is not None else
             "No same-kind word to compare with.")
        test("t7", "7 · Own part (reported)", "Does removing only the part of the bridge not shared with the answer still kill it?",
             own, "info", f"Remove {q(b[0])} minus {A} (early layers) → P({A}) {base:.2f} → {base * (1 - own):.2f}. Drop {own:.2f}.")

        # 8, 9 swaps over the early layers (at least layer 6 on)
        sl = [l for l in range(self.sc(6), onset)] or [l for l in range(onset)]
        if not sl:
            sl = self.wins[early[0]]
        fallback = self.classify(self.gen(text, self.rm(b, sl)))
        hit, keep = [], []
        for w in sw_words:
            belief_text = self.gen(chat(h, S["check"].format(w), ""))
            belief = self.classify(belief_text)
            try:
                out = self.gen(text, [Edit("swap", [b[0], w], sl, self.ALL)])
            except ValueError:
                continue
            got = self.classify(out)
            if belief == "other":
                outcome = "no usable belief"
            elif belief == ans_key:
                outcome = "kept the answer" if got == ans_key else "lost the answer"
                keep.append(got == ans_key)
            elif belief == fallback:
                outcome = "gave it, but it's also the fallback" if got == belief else "didn't follow"
            else:
                outcome = "followed the swap" if got == belief else "didn't follow"
                hit.append(got == belief)
            swaps.append(dict(key=f"{self.name}|{w}", sentence=self.name, swapped_in=w, layers=fmt_layers(sl),
                              model_says_its_answer=belief_text, model_answered=out, outcome=outcome))
        F = sum(hit) / len(hit) if hit else None
        K = sum(keep) / len(keep) if keep else None
        res.update(swaps_follow=None if F is None else round(F, 3), same_answer_keeps=None if K is None else round(K, 3),
                   fallback=fallback)
        test("t8", "8 · Swaps follow (reported)", "Swap in another word: does the answer become that word's answer?",
             F, "info", (f"{sum(hit)} of {len(hit)} swaps gave the swapped word's own answer "
                         f"(fallback answer with the bridge removed: {q(fallback)}).") if hit else "No usable swaps.")
        test("t9", "9 · Same-answer swap (reported)", "Swap in a word with the same answer: is the answer kept?",
             K, "info", f"{sum(keep)} of {len(keep)} kept the answer." if keep else "No same-answer swap word available.")
        if dropped:
            res["words_skipped"] = ", ".join(dropped)

        # quality: the share of the answer that rests on this bridge alone
        need = max(0.0, min(1.0, ne - noise))
        spec_old = 1.0 if bk is None else max(0.0, min(1.0, bk / ne)) if ne > 0 else 0.0
        spec = 1.0 if bonly is None else max(0.0, min(1.0, bonly / ne)) if ne > 0 else 0.0   # strict (spec_check2)
        iso = max(0.0, min(1.0, own / ne)) if ne > 0 else 0.0
        lead = None if bon is None else onset - bon
        quality = need * spec * iso if first else 0.0
        res.update(need=round(need, 3), specificity=round(spec, 3), specificity_old=round(spec_old, 3), isolation=round(iso, 3), lead=lead,
                   quality=round(quality, 3))
        test("q", "Quality", "Need × specificity × isolation (0 if the bridge doesn't come first)", quality, "info",
             f"Need {need:.2f} (drop beyond noise) × specificity {spec:.2f} (share left when only the bridge's own part, not shared with same-kind words, is removed) × "
             f"isolation {iso:.2f} (share left when the part shared with the answer is kept) = {quality:.2f}"
             + ("" if first else " → 0 because the bridge doesn't act before the answer") + ".")

        # verdict
        bn = ne - noise
        if bn >= TH["beats_noise"] and first:
            if bk is None:
                return finish("CONSULTED (KIND NOT CHECKED)", "needed early, beats noise, comes first; no same-kind word to compare")
            if bk >= TH["beats_kind"]:
                return finish("CONSULTED BRIDGE", "needed early, beats noise, comes first, beats its own kind, no collateral")
            return finish("CATEGORY EFFECT", f"needed and first, but removing {q(kw)} does about the same")
        if bn >= TH["beats_noise"] and not first:
            return finish("TANGLED", "needed, but it doesn't act before the answer does")
        if not reach_ok:
            return finish("UNTESTABLE", "no positive result, and the edit can't remove the bridge from this prompt, so a negative can't be read")
        if max(r16, rall) >= TH["spread"]:
            return finish("SPREAD OUT", "short windows don't matter, but removing it over many layers does")
        if bn < TH["bystander"] and max(r16, rall) < TH["bystander"]:
            return finish("BYSTANDER", "the edit reaches the bridge, but removing it doesn't change the answer")
        return finish("UNCLEAR", "in between")  # reach_ok here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", default=None)
    ap.add_argument("--ideas", action="store_true", help="run the question sets in scripts/ideas/*_specs.py")
    ap.add_argument("--all", action="store_true", help="round-3 questions plus the idea sets")
    ap.add_argument("--group", action="store_true", help="group-format pilot (scripts/group_specs.py)")
    ap.add_argument("--config", default="qwen35_4b")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    h = load(args.config)
    specs = SPECS
    if args.ideas or args.all:
        import importlib
        specs = {}
        for m in ("consulted_bridges.questions.ideas.sequence", "consulted_bridges.questions.ideas.language", "consulted_bridges.questions.ideas.science", "consulted_bridges.questions.ideas.culture"):
            specs.update(importlib.import_module(m).SPECS)
        if args.all:
            specs = {**SPECS, **specs}
    if args.group:
        from consulted_bridges.questions.group import SPECS as specs
    names = args.spec or list(specs)
    i, n = map(int, args.shard.split("/"))
    names = names[i::n]
    out = Path(args.out or f"results/card3_{Path(args.config).stem}") / f"shard{i}"
    out.mkdir(parents=True, exist_ok=True)
    acc = {k: [] for k in ("cards", "tests", "swaps", "controls", "profile")}
    for k, name in enumerate(names):
        print(f"PROGRESS {k}/{len(names)} starting {name}", flush=True)
        try:
            parts = Card(h, name, specs[name]).run()
        except Exception as err:
            import traceback
            traceback.print_exc()
            print(f"!! {name} failed: {err!r}", flush=True)
            continue
        acc["cards"].append(parts[0])
        for key, rows in zip(("tests", "swaps", "controls", "profile"), parts[1:]):
            acc[key].extend(rows)
        for key, rows in acc.items():
            pd.DataFrame(rows).to_csv(out / f"{key}.csv", index=False)
    print(f"PROGRESS {len(names)}/{len(names)} ALLDONE", flush=True)


if __name__ == "__main__":
    main()
