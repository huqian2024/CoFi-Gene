"""Cross-view fusion modules.

M2 (Week 2) needs the concatenation baseline (B3). M3 (Week 3) adds the gated
fusion baseline (B4). The source-aware attention (B5) lives in
:mod:`cofi_gene.models.attention` because it is a larger module; B6/B7 (one-step
graph feedback) are still refused by `build_model`.

B3 replaces both channel embeddings with one shared vector; B4 does the same but
learns the mixing weight, and B5 keeps two source-specific vectors that the model
projects back to 128 dims. In every case the model scores the *final* embedding
with one edge head per channel (see `cofi_gene.models.UpstreamModel`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ConcatFusion", "GatedFusion"]


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


class GatedFusion(nn.Module):
    """B4: feature-wise gate between exactly two evidence channels.

    Approved Week 3 design (``docs/WEEK3_IMPLEMENTATION_REPORT.md`` §5)::

        H_cat = [H_A ; H_B]                  # [N, 2d]
        g     = sigmoid(Linear(H_cat))       # [N, d]  - one value per gene AND feature
        H     = g * H_A + (1 - g) * H_B      # [N, d]

    Deliberately the smallest possible gating network: a single ``Linear(2d -> d)``
    plus ``Sigmoid``, no hidden layer and no ReLU, so B4 stays a fair "learned
    weighting" baseline between B3 (fixed concat) and B5 (cross-graph attention).
    ``g = 0.5`` at initialisation (zero bias, zero weights) gives the plain average,
    so the gate never starts from an extreme.

    The gate is differentiable and *feature-wise*: every gene gets its own d gate
    values, which can be inspected with :meth:`gate_weight`.
    """

    def __init__(self, dim: int, num_sources: int = 2, bias: bool = True) -> None:
        super().__init__()
        if num_sources != 2:
            raise ValueError("GatedFusion implements the two-channel gate only, "
                             f"got num_sources={num_sources}")
        if dim < 1:
            raise ValueError("gate dim must be >= 1")
        self.dim = dim
        self.num_sources = num_sources
        self.gate = nn.Linear(dim * num_sources, dim, bias=bias)

    def gate_weight(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """The gate ``g`` in ``(0, 1)``, shape ``[N, d]`` (exposed for tests/analysis)."""
        first, second = self._pair(embeddings)
        return torch.sigmoid(self.gate(torch.cat([first, second], dim=-1)))

    def forward(self, embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return the single fused embedding under every input channel name.

        Both keys map to the *same* vector: B4 produces one ``H`` and the two
        graphs then differ only through their independent edge heads.
        """
        first, second = self._pair(embeddings)
        gate = torch.sigmoid(self.gate(torch.cat([first, second], dim=-1)))
        fused = gate * first + (1.0 - gate) * second
        return {name: fused for name in embeddings}

    def _pair(self, embeddings: dict[str, torch.Tensor]):
        if len(embeddings) != self.num_sources:
            raise ValueError(f"expected {self.num_sources} embeddings, got {len(embeddings)}")
        first, second = (embeddings[name] for name in list(embeddings))
        if first.shape != second.shape:
            raise ValueError(f"channel embeddings must share a shape, "
                             f"got {tuple(first.shape)} vs {tuple(second.shape)}")
        if first.shape[-1] != self.dim:
            raise ValueError(f"expected embedding width {self.dim}, got {first.shape[-1]}")
        return first, second
