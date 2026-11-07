"""Word ranks under the J-lens and the logit lens at every position and layer, no cutoff.

rank = number of vocabulary tokens with a strictly larger lens logit (0 = top), the paper's
definition (silent_dissent.lens_eval.token_ranks), with no top-k cutoff or clamp. A word's rank
is the best rank over its first pieces (text.word_tokens). The last layer of either lens is the
model's own output distribution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from silent_dissent.lens_eval import token_ranks
from silent_dissent.model import get_layers, layer_output

from .interventions import Edit, applied
from .text import piece_strings, word_tokens


@dataclass
class Readout:
    ranks: np.ndarray            # [lens, position, layer, word], int; -1 = word has no usable piece
    lenses: list[str]
    words: list[str]
    tokens: list[str]            # the prompt's tokens, as text
    positions: list[int]         # absolute positions read (default: all)
    pieces: dict                 # word -> its pieces (as text)

    def last(self) -> pd.DataFrame:
        """Rank per (lens, layer) at the last position: one column per word."""
        return self._grid(self.ranks[:, -1])

    def best_over_positions(self, start: int = 0) -> pd.DataFrame:
        """Best (lowest) rank over positions >= start, per (lens, layer)."""
        keep = [i for i, p in enumerate(self.positions) if p >= start]
        r = np.where(self.ranks[:, keep] < 0, np.iinfo(np.int64).max, self.ranks[:, keep]).min(1)
        return self._grid(np.where(r == np.iinfo(np.int64).max, -1, r))

    def _grid(self, r) -> pd.DataFrame:
        rows = []
        for li, lens in enumerate(self.lenses):
            for layer in range(r.shape[1]):
                rows.append({"lens": lens, "layer": layer, **{w: int(r[li, layer, k]) for k, w in enumerate(self.words)}})
        return pd.DataFrame(rows)

    def long(self) -> pd.DataFrame:
        """Tidy table: one row per (lens, position, layer, word)."""
        L, P, N, W = self.ranks.shape
        li, pi, ni, wi = np.meshgrid(range(L), range(P), range(N), range(W), indexing="ij")
        return pd.DataFrame({
            "lens": np.array(self.lenses)[li.ravel()], "pos": np.array(self.positions)[pi.ravel()],
            "token": np.array(self.tokens)[np.array(self.positions)[pi.ravel()]],
            "layer": ni.ravel(), "word": np.array(self.words)[wi.ravel()], "rank": self.ranks.ravel()})

    def summary(self, band: list[int], start: int = 0) -> pd.DataFrame:
        """Per (lens, word): at the last position, the best rank over layers (and where) and
        the band-mean log10(rank+1); the same over all positions >= start (best position per layer)."""
        rows = []
        last, allp = self.ranks[:, -1], self.best_over_positions(start)
        for li, lens in enumerate(self.lenses):
            sub = allp[allp.lens == lens]
            for k, w in enumerate(self.words):
                r = last[li, :, k]
                ra = sub[w].to_numpy()
                if (r < 0).all():
                    rows.append({"lens": lens, "word": w})
                    continue
                rows.append({"lens": lens, "word": w,
                             "last_best_rank": int(r.min()), "last_best_layer": int(r.argmin()),
                             "last_band_logrank": float(np.log10(r[band] + 1).mean()),
                             "last_final_layer_rank": int(r[-1]),
                             "allpos_best_rank": int(ra.min()), "allpos_best_layer": int(ra.argmin()),
                             "allpos_band_logrank": float(np.log10(ra[band] + 1).mean())})
        return pd.DataFrame(rows)

    def where_best(self, word: str, lens: str = "jlens", top: int = 5) -> pd.DataFrame:
        """The (position, layer) cells where `word` ranks best."""
        d = self.long()
        d = d[(d.word == word) & (d.lens == lens) & (d["rank"] >= 0)]
        return d.nsmallest(top, "rank")


@torch.no_grad()
def readout(h, text: str, words: list[str], lenses: list[str] | None = None, layers: list[int] | None = None,
            positions: list[int] | None = None, edits: list[Edit] | None = None, dedupe: bool = True,
            chunk: int = 64) -> Readout:
    """Rank every word at every position (default) and layer of `text`, under each lens.
    `edits` (ablate/steer/swap) are applied first, so the readout sees the edited model."""
    lenses = lenses or list(h.lenses)
    ids = h.encode(text)
    n = len(ids)
    positions = sorted({p % n for p in positions}) if positions is not None else list(range(n))
    if positions[-1] != n - 1:
        positions.append(n - 1)            # the last position is always read
    toks = word_tokens(h, words, dedupe)
    union = sorted({t for v in toks.values() for t in v}) or [0]
    col = {t: i for i, t in enumerate(union)}
    blocks = get_layers(h.model)
    layers = list(range(len(blocks))) if layers is None else layers
    store: dict[int, torch.Tensor] = {}
    pos_t = torch.tensor(positions, device=h.device)
    with applied(h, text, edits):
        handles = [blocks[l].register_forward_hook(
            lambda _m, _i, out, l=l: store.__setitem__(l, layer_output(out)[0, pos_t].detach())) for l in layers]
        try:
            h.model(input_ids=torch.tensor([ids], device=h.device))
        finally:
            for hd in handles:
                hd.remove()
    uid = torch.tensor(union, device=h.device)
    ranks = np.full((len(lenses), len(positions), len(blocks), len(words)), -1, dtype=np.int64)
    for li, name in enumerate(lenses):
        lens = h.lenses[name]
        for l in layers:
            H = store[l]
            for c in range(0, len(H), chunk):
                logits = lens.logits(H[c : c + chunk], l).float()
                rk = token_ranks(logits, uid.expand(len(logits), -1)).cpu().numpy()   # [chunk, U]
                for k, w in enumerate(words):
                    if toks[w]:
                        ranks[li, c : c + chunk, l, k] = rk[:, [col[t] for t in toks[w]]].min(1)
    return Readout(ranks, lenses, words, [h.tok.decode([t]) for t in ids], positions,
                   {w: piece_strings(h, toks[w]) for w in words})
