"""Edge-scoring heads shared by every upstream model (B0-B7).

The head is deliberately tiny: the experiment is about the *node* representation,
so B0 (raw features) and B1-B3 (GCN/fusion) must be compared with an identical
decoder, identical negatives and identical optimisation budget.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DotProductEdgeHead", "MLPEdgeHead", "build_edge_head", "decode_pairs"]


class DotProductEdgeHead(nn.Module):
    """Symmetric dot product of the two node embeddings."""

    def forward(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        u, v = edge_index[0], edge_index[1]
        return (embeddings[u] * embeddings[v]).sum(dim=-1)


class MLPEdgeHead(nn.Module):
    """Slightly stronger decoder: MLP over ``[z_u, z_v, z_u * z_v]``."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or max(16, dim // 2)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        u, v = edge_index[0], edge_index[1]
        zu, zv = embeddings[u], embeddings[v]
        return self.mlp(torch.cat([zu, zv, zu * zv], dim=-1)).squeeze(-1)


def build_edge_head(kind: str, dim: int) -> nn.Module:
    if kind == "dot":
        return DotProductEdgeHead()
    if kind == "mlp":
        return MLPEdgeHead(dim)
    raise ValueError(f"unknown edge head: {kind!r}")


def decode_pairs(head: nn.Module, embeddings: torch.Tensor,
                 edge_index: torch.Tensor) -> torch.Tensor:
    """Edge logits for a ``[2, K]`` index array."""
    if edge_index.numel() == 0:
        return embeddings.new_zeros((0,))
    return head(embeddings, edge_index.long())
