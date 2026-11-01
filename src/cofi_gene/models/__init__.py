"""Upstream model registry for B0-B3 (Week 2 / M2 scope).

B0  raw features, no message passing, same edge head as every other model
B1  single graph A + 2-layer GCN
B2  single graph B + 2-layer GCN
B3  both graphs + per-channel GCN + concat projection

B4-B7 (gated fusion, source-aware attention, one-step feedback) are M3/M4 work and
raise :class:`NotImplementedError` on purpose, so no ablation table can ever be
produced from a placeholder implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CHANNEL_SETS
from .fusion import ConcatFusion
from .gcn import GCNEncoder, normalized_adjacency
from .heads import build_edge_head, decode_pairs

__all__ = ["UpstreamModel", "build_model", "IMPLEMENTED_MODELS", "PLANNED_MODELS"]

IMPLEMENTED_MODELS = ("b0", "b1", "b2", "b3")
PLANNED_MODELS = ("b4", "b5", "b6", "b7")

_PLANNED_REASON = {
    "b4": "M3: gated fusion baseline",
    "b5": "M3: source-aware cross-graph attention",
    "b6": "M4: B5 + one-step graph feedback",
    "b7": "M4: B4 + one-step graph feedback",
}


class UpstreamModel(nn.Module):
    """Encodes nodes per channel and exposes one shared edge-scoring head."""

    def __init__(self, name: str, feature_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 fusion_dim: int = 128, edge_head: str = "dot") -> None:
        super().__init__()
        if name not in IMPLEMENTED_MODELS:
            raise NotImplementedError(
                f"model {name!r} is not implemented yet ({_PLANNED_REASON.get(name, 'planned')}); "
                f"implemented: {IMPLEMENTED_MODELS}")
        self.name = name
        self.channels = tuple(CHANNEL_SETS[name])
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.fusion_kind = "concat" if name == "b3" else "none"

        if name == "b0":
            self.input_projection = (nn.Identity() if feature_dim == fusion_dim
                                     else nn.Linear(feature_dim, fusion_dim))
            self.output_dim = fusion_dim
        else:
            self.encoders = nn.ModuleDict({
                channel: GCNEncoder(feature_dim, hidden_dim, num_layers, dropout,
                                    out_dim=hidden_dim)
                for channel in self.channels})
            self.output_dim = hidden_dim
            if name == "b3":
                self.fusion = ConcatFusion(hidden_dim, len(self.channels), fusion_dim)
                self.output_dim = fusion_dim
        self.head = build_edge_head(edge_head, self.output_dim)

    def forward(self, x: torch.Tensor, adjacencies: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return ``channel -> node embedding`` used for that channel's edge scoring."""
        if self.name == "b0":
            embedding = self.input_projection(x)
            return {channel: embedding for channel in self.channels}
        per_channel = {}
        for channel in self.channels:
            if channel not in adjacencies:
                raise KeyError(f"model {self.name} needs adjacency for channel {channel!r}")
            per_channel[channel] = self.encoders[channel](x, adjacencies[channel])
        if self.name == "b3":
            fused = self.fusion([per_channel[channel] for channel in self.channels])
            return {channel: fused for channel in self.channels}
        return per_channel

    def score_edges(self, embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return decode_pairs(self.head, embeddings, edge_index)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def build_model(name: str, feature_dim: int, **kwargs) -> UpstreamModel:
    """Factory used by the trainer and by the tests."""
    return UpstreamModel(name=name, feature_dim=feature_dim, **kwargs)


def build_adjacencies(encode_edges: dict[str, torch.Tensor],
                      encode_weights: dict[str, torch.Tensor] | None,
                      num_nodes: int) -> dict[str, torch.Tensor]:
    """Normalised sparse adjacency per channel, built once per run."""
    adjacencies = {}
    for channel, edges in encode_edges.items():
        weights = None if encode_weights is None else encode_weights.get(channel)
        adjacencies[channel] = normalized_adjacency(edges, weights, num_nodes)
    return adjacencies
