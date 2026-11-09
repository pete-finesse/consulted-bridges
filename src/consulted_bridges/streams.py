"""Read and edit a decoder block's output, for models with one residual stream or several.

Most models output [B, S, d] after each block. DeepSeek-V4 keeps hc_mult = 4 parallel residual
streams (manifold-constrained hyper-connections), so its blocks output [B, S, 4, d]. Its J-lens
(Neuronpedia, deepseek-v4-flash) was fit on the MEAN of the four streams ("block_output",
stream_reduce: mean), so:

  read   returns that mean, [B, S, d], which is what the lens decodes;
  edits  (interventions._apply) act on each stream separately. For ablate, ablate_excl, move,
         amplify and swap, which are linear in h, the mean then changes exactly as the same edit
         of the mean would, and no stream keeps the removed part, so the model's own weighted
         readout of the streams (hc_head) and the next block's input lose it too. steer adds
         alpha * |h_i| * d to stream i, so the mean moves by alpha * mean_i |h_i| instead of
         alpha * |mean h|.
"""
from __future__ import annotations

import torch

from silent_dissent.model import layer_output


def read(out) -> torch.Tensor:
    """A block's output as one residual per position: [B, S, d] (the mean over streams if several)."""
    x = layer_output(out)
    return x.mean(dim=2) if x.dim() == 4 else x
