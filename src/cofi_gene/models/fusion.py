"""Cross-view fusion modules.

M2 (Week 2) only needs the concatenation baseline (B3). The gated (B4/B7) and
source-aware attention (B5/B6) modules belong to M3/M4 and are intentionally not
implemented yet: `build_model` raises a clear error instead of silently training a
placeholder that would later invalidate the ablation table.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ConcatFusion"]


class ConcatFusion(nn.Module):
    """B3: concatenate the per-channel embeddings, then project back to ``out_dim``.

    Keeping the output width equal to a single encoder's width is what makes
    B3/B4/B5 comparable at the same downstream cost.
    """

    def __init__(self, in_dim: int, num_sources: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        if num_sources < 1:
            raise ValueError("num_sources must be >= 1")
        self.in_dim = in_dim
        self.num_sources = num_sources
        self.out_dim = out_dim
        self.projection = nn.Linear(in_dim * num_sources, out_dim, bias=bias)

    def forward(self, embeddings) -> torch.Tensor:
        if len(embeddings) != self.num_sources:
            raise ValueError(f"expected {self.num_sources} embeddings, got {len(embeddings)}")
        return self.projection(torch.cat(list(embeddings), dim=-1))
