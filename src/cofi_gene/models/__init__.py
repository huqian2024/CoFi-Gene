"""Upstream model registry for B0-B5 (Week 2 + Week 3 scope).

B0  raw features, no message passing, same edge head as every other model
B1  single graph A + 2-layer GCN
B2  single graph B + 2-layer GCN
B3  both graphs + per-channel GCN + concat projection
B4  both graphs + per-channel GCN + gated fusion              (Week 3)
B5  both graphs + per-channel GCN + cross-graph attention     (Week 3)

B6-B7 (one-step graph feedback) are M4 work and raise
:class:`NotImplementedError` on purpose, so no ablation table can ever be
produced from a placeholder implementation.

Supervision stays graph-specific for the dual-graph fusion models: B4/B5 keep two
independent edge heads (one per channel, D-005) and their own per-channel BCE term.
B4 fuses to a single ``H``; B5 keeps two source-specific ``Z_A``/``Z_B`` that are
projected back to 128 dims and are also exposed through
:meth:`UpstreamModel.fusion_representations` for analysis. Either way the head
decodes the final ``[N, 128]`` embedding, so no model hides an untrained
projection between embedding and decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CHANNEL_SETS
from .attention import CrossGraphAttention
from .fusion import ConcatFusion, GatedFusion
from .gcn import GCNEncoder, normalized_adjacency
from .heads import build_edge_head, decode_pairs

__all__ = ["UpstreamModel", "build_model", "IMPLEMENTED_MODELS", "PLANNED_MODELS"]

IMPLEMENTED_MODELS = ("b0", "b1", "b2", "b3", "b4", "b5")
PLANNED_MODELS = ("b6", "b7")

#: Models that must keep one *independent* edge head per channel (Week 3 Part 7.1).
#: B0-B3 keep the single Week 2 head so their behaviour is untouched.
PER_CHANNEL_HEAD_MODELS = ("b4", "b5")

#: Models whose fusion keeps two distinguishable per-source representations, so
#: that ``fusion_representations`` is meaningful (B4 outputs a single fused H).
SOURCE_SPECIFIC_MODELS = ("b5",)

_FUSION_KIND = {"b0": "none", "b1": "none", "b2": "none", "b3": "concat",
                "b4": "gate", "b5": "attention"}

_PLANNED_REASON = {
    "b6": "M4: B5 + one-step graph feedback",
    "b7": "M4: B4 + one-step graph feedback",
}


class UpstreamModel(nn.Module):
    """Encodes nodes per channel and exposes one or two edge-scoring heads."""

    def __init__(self, name: str, feature_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 fusion_dim: int = 128, edge_head: str = "dot",
                 attention_heads: int = 4, attention_dropout: float | None = None,
                 attention_residual: bool = False) -> None:
        super().__init__()
        if name not in IMPLEMENTED_MODELS:
            raise NotImplementedError(
                f"model {name!r} is not implemented yet ({_PLANNED_REASON.get(name, 'planned')}); "
                f"implemented: {IMPLEMENTED_MODELS}")
        self.name = name
        self.channels = tuple(CHANNEL_SETS[name])
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.fusion_kind = _FUSION_KIND[name]
        self.output_projection = None
        self.input_projection = None

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
            elif name == "b4":
                # Feature-wise gate -> single fused H, already [N, fusion_dim]-wide
                # because hidden_dim == fusion_dim (both 128). No extra projection:
                # an untrained projection would confound the downstream comparison.
                self.fusion = GatedFusion(hidden_dim, num_sources=len(self.channels))
                self.output_dim = hidden_dim
            elif name == "b5":
                self.fusion = CrossGraphAttention(
                    hidden_dim, self.channels, num_heads=attention_heads,
                    dropout=dropout if attention_dropout is None else attention_dropout,
                    residual=attention_residual)
                # Z_A / Z_B -> [Z_A ; Z_B] (256) -> 128 (AGENTS.md 4.2).
                self.output_projection = ConcatFusion(hidden_dim, len(self.channels),
                                                      fusion_dim)
                self.output_dim = fusion_dim
        # The edge head always decodes the *final* [N, output_dim] embedding, so
        # no model carries an untrained projection between embedding and decoder.
        self.head_dim = self.output_dim

        self.head = build_edge_head(edge_head, self.head_dim)
        # None for B0-B3 (one head shared by every channel, as in Week 2).
        self.channel_heads = None
        if name in PER_CHANNEL_HEAD_MODELS:
            self.channel_heads = nn.ModuleDict({
                channel: build_edge_head(edge_head, self.head_dim)
                for channel in self.channels})


    # ------------------------------------------------------------- encoding
    def forward(self, x: torch.Tensor, adjacencies: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return ``channel -> final embedding`` used for that channel's edge scoring.

        Every model hands the head one trained ``[N, output_dim]`` vector, so the
        decoder never reads an untrained intermediate. B5 additionally exposes its
        two source-specific representations through :meth:`fusion_representations`
        (AGENTS.md 6 asks for them to be recorded); B4 fuses to a single vector.
        """
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
        if self.name == "b4":
            # GatedFusion already returns the single [N, hidden_dim] fused vector.
            return self.fusion(per_channel)
        if self.name == "b5":
            sources = self.fusion(per_channel)          # Z_A, Z_B  [N, hidden_dim]
            fused = self.output_projection([sources[channel] for channel in self.channels])
            return {channel: fused for channel in self.channels}
        return per_channel

    def fusion_representations(self, x: torch.Tensor,
                               adjacencies: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Source-specific ``{channel: [N, hidden_dim]}`` representations (B5 only).

        These are the pre-projection ``Z_A``/``Z_B``: the quantities B4 collapses
        into one vector. Kept for inspection, ablations and the Week 5/6 analysis;
        the trainer scores the projected embedding.
        """
        if self.name not in SOURCE_SPECIFIC_MODELS:
            raise NotImplementedError(
                f"model {self.name!r} keeps no per-source representations "
                f"(only {SOURCE_SPECIFIC_MODELS} do)")
        missing = [channel for channel in self.channels if channel not in adjacencies]
        if missing:
            raise KeyError(f"model {self.name} needs adjacency for channel(s) {missing}")
        per_channel = {channel: self.encoders[channel](x, adjacencies[channel])
                       for channel in self.channels}
        return self.fusion(per_channel)

    def encode(self, x: torch.Tensor, adjacencies: dict[str, torch.Tensor]) -> torch.Tensor:
        """The single ``[N, output_dim]`` embedding handed to the downstream MLP.

        B4/B5 project the concatenation of their two source-specific
        representations back to ``fusion_dim`` (AGENTS.md 4.2: "concat + linear
        projection to 128"), exactly like B3, so B0-B5 stay comparable.
        """
        outputs = self.forward(x, adjacencies)
        # Every model returns one shared vector (B1/B2 simply have one channel).
        return outputs[self.channels[0]]

    # -------------------------------------------------------------- scoring
    def head_for(self, channel: str | None = None) -> nn.Module:
        """Per-channel head for B4/B5, otherwise the single shared head."""
        if channel is not None and self.channel_heads is not None:
            try:
                return self.channel_heads[channel]
            except KeyError as exc:
                raise KeyError(f"model {self.name} has no channel head for {channel!r}; "
                               f"available: {tuple(self.channel_heads)}") from exc
        return self.head

    def score_edges(self, embeddings: torch.Tensor, edge_index: torch.Tensor,
                    channel: str | None = None) -> torch.Tensor:
        return decode_pairs(self.head_for(channel), embeddings, edge_index)

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
