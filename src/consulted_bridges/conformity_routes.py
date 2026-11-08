"""Follow-up to the conformity run: when removing the bridge leaves the answer intact, what is carrying it?

--test swap (the default, the one reported): Neuronpedia's move of the bridge onto another word of its kind
(Italy -> Spain), by the rules fixed in docs/swap_test_rules.md (Rule A: the question's own words; the whole
range of layers before the prompt's answer onset). If the answer still reads the bridge, swapping it should
pull the answer toward the swapped-in word's letter even when the peers agree ("backup route"); if nothing
reads it, the swap does nothing ("present but not used"). Run in every alone, agree and give-in prompt with
early windows, plus the hidden-letter prompts below.
  targets      up to 4 words from the question's swap list: single token, initial not the right letter, not
               the peers' letter, not an earlier target's
  everyday     the first 3 of table, river, garden, lamp, bottle, window whose initial is not the right letter,
               the peers' or a target's (move the bridge onto them: should not get through)
  fallback     the answer with the bridge removed at the same layers and positions; a target whose letter is
               the fallback is uninformative
  source ctrl  move "table", and the first unused word of the swap list, onto the first target
  follows      the swapped-in word's letter becomes the top letter among the letters scored; shift = change in log P(target) - log P(right)

--test copy (stopped after ~100 prompts): removing the peers' letter direction at their tokens moved the
balance no more than 12 everyday words did, on the 27B give-ins, so it does not block copying. Kept for the
record; its partial results are in results/conformity_routes/<frame>_<config>/.

Two routes can give the right letter when peers agree: the bridge route (Colosseum -> Italy -> I) and the
copy route (read the letter the peers wrote). Removing the bridge alone cannot tell "present but not used"
(only copying carries the answer) from "backup route" (either route is enough). This module removes the copy
route, then both, and also asks the question with the peers' letters hidden.

Prompts (selected from the main run's prompts_all.csv, so onsets and labels are the ones already reported):
  agree     where the bridge was "not needed" or "mixed"
  give-in   disagree prompts where the model gave in (full or partial)
  hidden    3 peers who write "(answer submitted)" instead of a letter: the group and the wording stay,
            the letter to copy is gone. Run through the main experiment's full measurement (own onset).

Edits in agree and give-in prompts:
  copy block    remove the peers' letter direction (plain and raw-token forms) at the peers' letter tokens,
                in every layer. Control: each of 12 everyday words removed the same way.
  both          copy block plus the bridge removed in each early window (the main run's onset); control:
                copy block plus each everyday word in each early window.
Measures (drop = 1 - P_after / P_before, as in the scorecard):
  copy_need          drop in P(right) from the copy block minus the largest everyday-word drop
  need_after_block   with the copy block in place: bridge drop minus the largest everyday-word drop,
                     both measured from the copy-blocked P(right)
  give-ins: copy_dLO (negative = toward the right letter; LO is 0 by construction in agree prompts, where
            the peers' letter is the right one), restored (the right letter wins once the copy
            route is blocked), pushback_after_block (bridge dLO minus everyday-word dLO, copy block in place)
Labels for agree prompts (thresholds as in the main run: 0.5 needed, 0.2 not needed):
  copying carries it     copy_need >= 0.5 and need_after_block < 0.5
  backup route           copy_need < 0.2 and need_after_block >= 0.5
  both contribute        copy_need >= 0.5 and need_after_block >= 0.5
  neither removal kills  copy_need < 0.2 and need_after_block < 0.2
  mixed                  otherwise

  uv run python -m consulted_bridges.conformity_routes --config qwen36_27b --frame neutral [--shard 0/3]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from consulted_bridges.conformity import (CALIBRATION, FRAMES, SPECS, SUFFIX, TH, Prompt, group_user,
                                          peer_positions, prompts, wrong_letter)
from consulted_bridges.interventions import Edit, answer_probs
from consulted_bridges.legacy.bridge_card3 import FORMATS
from consulted_bridges.load import load
from consulted_bridges.text import resolve_positions

HIDDEN = "(answer submitted)"
EVERYDAY = ["table", "river", "garden", "lamp", "bottle", "window"]


def main_dir(cfg, frame):
    return Path(f"results/conformity_{cfg}" if frame == "neutral" else f"results/conformity_{frame}_{cfg}")


def selected(cfg, frame, main=None):
    """(qid, condition, k, main-run row) for the prompts this follow-up re-measures."""
    p = pd.read_csv(Path(main or main_dir(cfg, frame)) / "prompts_all.csv")
    a = p[(p.condition == "agree") & p.label.isin(["bridge not needed", "mixed"])]
    g = p[(p.condition == "disagree") & p.state.str.contains("give-in") & (p.n_early > 0)]
    rows = pd.concat([a, g]).sort_values(["question", "condition", "peers"])
    return [(r.question, r.condition, int(r.peers), r) for r in rows.itertuples()], sorted(p.question.unique())


def selected_swap(cfg, frame, main=None):
    p = pd.read_csv(Path(main or main_dir(cfg, frame)) / "prompts_all.csv")
    keep = (p.condition == "agree") | ((p.condition == "disagree") & p.state.str.contains("give-in"))
    if frame == "neutral":                 # the alone prompt is the same under both wordings
        keep |= p.condition == "alone"
    rows = p[keep & (p.n_early > 0)].sort_values(["question", "condition", "peers"])
    return [(r.question, r.condition, int(r.peers), r) for r in rows.itertuples()], sorted(p.question.unique())


class Swap(Prompt):
    def run_swap(self, main, rows):
        import math
        c, S, h = self.c, self.spec, self.h
        b = S["bridge"][0]
        t0 = time.time()
        qtext = SPECS[self.qid]["user"].replace(SUFFIX, "")
        pos = resolve_positions(h, c.text, "span:" + qtext)
        layers = list(range(0, int(main.onset)))
        avoid = {self.right, self.peer if self.cond != "alone" else None}
        targets, used = [], set()
        for w in S["swaps"]:
            L_ = w[0].upper()
            if c.one_token(w) and L_.isalpha() and L_ not in avoid and L_ not in used and w != b:
                targets.append(w); used.add(L_)
            if len(targets) == 4:
                break
        every = [w for w in EVERYDAY if c.one_token(w) and w[0].upper() not in avoid | used][:3]
        letters = list(dict.fromkeys([self.right] + ([self.peer] if self.cond != "alone" else []) +
                                     [w[0].upper() for w in targets + every]))

        def P(edits=None):
            r = answer_probs(h, c.text, letters, edits, formats=FORMATS)["cands"]
            return {L_: sum(r[L_].get(f"p[{x}]", 0.0) for x in FORMATS) for L_ in letters}

        top = lambda p: max(p, key=p.get)  # noqa: E731
        lo = lambda p, a: math.log(p[a] + 1e-9) - math.log(p[self.right] + 1e-9)  # noqa: E731
        base = P()
        fb = top(P([Edit("ablate", [b], layers, pos)]))
        rec = dict(question=self.qid, calibration=self.qid in CALIBRATION, condition=self.cond, peers=self.k,
                   right=self.right, peer_letter=self.peer if self.cond != "alone" else None, main_label=main.label,
                   main_state=main.state, main_need=main.need, main_pushback=getattr(main, "pushback", None),
                   P_right=base[self.right], P_right_main=main.P_right, top=top(base), onset=main.onset,
                   n_pos=len(pos), fallback=fb, targets=" ".join(targets), everyday=" ".join(every))
        res = []
        for w in targets:
            p = P([Edit("move", [b, w], layers, pos)])
            T = w[0].upper()
            r_ = dict(kind="target", word=w, letter=T, top=top(p), follows=top(p) == T, informative=T != fb,
                      P_target=p[T], P_target_base=base[T], P_right_after=p[self.right], shift=lo(p, T) - lo(base, T))
            res.append(r_)
        for w in every:
            p = P([Edit("move", [b, w], layers, pos)])
            T = w[0].upper()
            res.append(dict(kind="everyday", word=w, letter=T, top=top(p), follows=top(p) == T, informative=T != fb,
                            P_target=p[T], P_target_base=base[T], P_right_after=p[self.right], shift=lo(p, T) - lo(base, T)))
        if targets:
            T = targets[0][0].upper()
            spare = next((w for w in S["swaps"] if w not in targets and w != b and c.one_token(w)), None)
            for src in ["table"] + ([spare] if spare else []):
                p = P([Edit("move", [src, targets[0]], layers, pos)])
                res.append(dict(kind="source control", word=f"{src}->{targets[0]}", letter=T, top=top(p),
                                follows=top(p) == T, informative=True, P_target=p[T], P_target_base=base[T],
                                P_right_after=p[self.right], shift=lo(p, T) - lo(base, T)))
        for r_ in res:
            rows.append(dict(question=self.qid, condition=self.cond, peers=self.k, **r_))
        tg = [r_ for r_ in res if r_["kind"] == "target" and r_["informative"]]
        ev = [r_ for r_ in res if r_["kind"] == "everyday" and r_["informative"]]
        sc = [r_ for r_ in res if r_["kind"] == "source control"]
        rec.update(n_targets=len(tg), n_follow=sum(r_["follows"] for r_ in tg),
                   follow_share=(sum(r_["follows"] for r_ in tg) / len(tg)) if tg else None,
                   median_shift=float(pd.Series([r_["shift"] for r_ in tg]).median()) if tg else None,
                   everyday_readable=len(ev), everyday_through=sum(r_["follows"] for r_ in ev),
                   counts=bool(ev) and not any(r_["follows"] for r_ in ev),
                   source_follow=sum(r_["follows"] for r_ in sc),
                   source_max_shift=max((r_["shift"] for r_ in sc), default=None))
        return rec, t0


class Routes(Prompt):
    def run_routes(self, main, rows):
        c, S = self.c, self.spec
        b = S["bridge"]
        t0 = time.time()
        pr0, pw0, lo0 = self.probs()
        early = [i for i, w in enumerate(c.wins) if w[-1] < main.onset]
        peers = peer_positions(self.h, c.text, self.peer, self.k)
        L = list(range(c.L))
        block = Edit("ablate", self.dirs(self.peer, self.pforms), L, peers)
        nulls = self.pr.nulls()
        rec = dict(question=self.qid, calibration=self.qid in CALIBRATION, condition=self.cond, peers=self.k,
                   right=self.right, peer_letter=self.peer, main_label=main.label, main_state=main.state,
                   main_need=main.need, main_pushback=main.pushback, P_right=pr0, P_peer=pw0, LO=lo0,
                   P_right_main=main.P_right, onset=main.onset, n_early=len(early), n_peer_tokens=len(peers))
        # copy route blocked
        rc, wc, loc = self.probs([block])
        nz = [self.probs([Edit("ablate", [z], L, peers)]) for z in nulls]
        rec.update(P_right_block=rc, P_peer_block=wc, LO_block=loc, copy_drop=c.drop(pr0, rc),
                   copy_noise_drop=max(c.drop(pr0, r) for r, _, _ in nz), copy_dLO=loc - lo0,
                   # the everyday word that moves the balance furthest toward the right letter
                   copy_noise_dLO=min(lo - lo0 for _, _, lo in nz))
        rec["copy_need"] = rec["copy_drop"] - rec["copy_noise_drop"]
        for z, (r, w, lo) in zip(nulls, nz):
            rows.append(dict(question=self.qid, condition=self.cond, peers=self.k, edit=f"block + noise@peers: {z}",
                             window="all", P_right=r, P_peer=w, LO=lo))
        rows.append(dict(question=self.qid, condition=self.cond, peers=self.k, edit="copy block", window="all",
                         P_right=rc, P_peer=wc, LO=loc))
        # both: copy block plus the bridge (or an everyday word) in each early window
        best = {}
        for label, words in [("bridge", b)] + [(f"noise: {z}", [z]) for z in nulls]:
            for i in early:
                w = c.wins[i]
                r, wr, lo = self.probs([block] + self.rm(words, w, self.allpos))
                rows.append(dict(question=self.qid, condition=self.cond, peers=self.k, edit=f"block + {label}",
                                 window=f"{w[0]}-{w[-1]}", P_right=r, P_peer=wr, LO=lo))
                cur = best.setdefault(label, dict(drop=-1.0, dLO=-1e9))
                cur["drop"] = max(cur["drop"], c.drop(rc, r))
                if lo - loc > cur["dLO"]:
                    cur.update(dLO=lo - loc, P_right=r, window=f"{w[0]}-{w[-1]}")
        bn = best.get("bridge")
        if bn:
            nd = max(v["drop"] for k_, v in best.items() if k_.startswith("noise"))
            ndl = max(v["dLO"] for k_, v in best.items() if k_.startswith("noise"))
            rec.update(both_P_right=bn["P_right"], both_window=bn["window"], bridge_drop_after_block=bn["drop"],
                       noise_drop_after_block=nd, need_after_block=bn["drop"] - nd,
                       bridge_dLO_after_block=bn["dLO"], noise_dLO_after_block=ndl, pushback_after_block=bn["dLO"] - ndl)
        if self.cond == "agree":
            cn, na = rec["copy_need"], rec.get("need_after_block", float("nan"))
            hi, lo_ = TH["need"], TH["none_need"]
            rec["route"] = ("copying carries it" if cn >= hi and na < hi else
                            "backup route" if cn < lo_ and na >= hi else
                            "both contribute" if cn >= hi and na >= hi else
                            "neither removal kills" if cn < lo_ and na < lo_ else "mixed")
        else:
            rec["restored"] = bool(loc < 0)
            rec["route"] = ("right answer back, bridge carries it" if loc < 0 and rec.get("pushback_after_block", 0) >= TH["pull_noise"] else
                            "right answer back" if loc < 0 else "still gives in")
        return rec, t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="qwen36_27b")
    ap.add_argument("--frame", default="neutral", choices=list(FRAMES))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts of each kind (testing)")
    ap.add_argument("--test", default="swap", choices=["swap", "copy"])
    ap.add_argument("--main", default=None, help="main run folder with prompts_all.csv (default: the frame's)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = Path(args.config).stem
    i, n = map(int, args.shard.split("/"))
    sel, qs = (selected_swap if args.test == "swap" else selected)(cfg, args.frame, args.main)
    hidden = [(q, "hidden", 3, None) for q in qs]
    if args.limit:
        sel, hidden = sel[: args.limit], hidden[: args.limit]
    work = [x for j, x in enumerate(sel + hidden) if j % n == i]
    out = Path(args.out or (f"results/conformity_routes/swap_{args.frame}_{cfg}" if args.test == "swap" else
                            f"results/conformity_routes/{args.frame}_{cfg}")) / f"shard{i}"
    out.mkdir(parents=True, exist_ok=True)
    h = load(args.config)
    summ, hid, rows, hrows = [], [], [], []
    for j, (q, cond, k, main) in enumerate(work):
        print(f"PROGRESS {j}/{len(work)} {q} {cond}{k}", flush=True)
        try:
            if cond == "hidden":
                s = SPECS[q]
                spec = dict(s, user=group_user(s["user"].replace(SUFFIX, ""), HIDDEN, k, args.frame))
                rec, t0 = Prompt(h, q, "hidden", k, wrong_letter(q, s), spec).run(hrows)
                hid.append(rec)
                print(f"[{q} hidden{k}] says {rec['model_said']!r} P(right) {rec['P_right']:.2f} onset {rec['onset']} "
                      f"need {rec.get('need', float('nan')):.2f} | {rec['label']} ({time.time() - t0:.0f}s)", flush=True)
            elif args.test == "swap":
                peer = main.peer_letter if cond != "alone" else wrong_letter(q, SPECS[q])
                spec = next(p[3] for p in prompts(q, args.frame) if p[0] == cond and p[1] == k)
                rec, t0 = Swap(h, q, cond, k, peer, spec).run_swap(main, rows)
                summ.append(rec)
                print(f"[{q} {cond}{k}] P(right) {rec['P_right']:.2f} (main {rec['P_right_main']:.2f}) fallback {rec['fallback']} | "
                      f"follow {rec['n_follow']}/{rec['n_targets']} shift {rec['median_shift']} | everyday through "
                      f"{rec['everyday_through']}/{rec['everyday_readable']} source {rec['source_follow']} | {rec['main_label']} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            else:
                peer = main.peer_letter
                spec = next(p[3] for p in prompts(q, args.frame) if p[0] == cond and p[1] == k)
                rec, t0 = Routes(h, q, cond, k, peer, spec).run_routes(main, rows)
                summ.append(rec)
                print(f"[{q} {cond}{k}] P(right) {rec['P_right']:.2f} (main {rec['P_right_main']:.2f}) "
                      f"block -> {rec['P_right_block']:.2f} copy_need {rec['copy_need']:.2f} dLO {rec['copy_dLO']:+.2f} | "
                      f"need after block {rec.get('need_after_block', float('nan')):.2f} pushback after "
                      f"{rec.get('pushback_after_block', float('nan')):+.2f} | {rec['route']} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as err:
            import traceback
            traceback.print_exc()
            print(f"!! {q} {cond}{k} failed: {err!r}", flush=True)
        stem = "swap" if args.test == "swap" else "routes"
        pd.DataFrame(summ).to_csv(out / f"{stem}.csv", index=False)
        pd.DataFrame(rows).to_csv(out / f"{stem}_{'moves' if args.test == 'swap' else 'windows'}.csv", index=False)
        pd.DataFrame(hid).to_csv(out / "hidden.csv", index=False)
        pd.DataFrame(hrows).to_csv(out / "hidden_windows.csv", index=False)
    print("PROGRESS ALLDONE", flush=True)


if __name__ == "__main__":
    main()
