"""Upstream model registry for B0-B7 (Week 2 + Week 3 + Week 4 scope).

B0  raw features, no message passing, same edge head as every other model
B1  single graph A + 2-layer GCN
B2  single graph B + 2-layer GCN
B3  both graphs + per-channel GCN + concat projection
B4  both graphs + per-channel GCN + gated fusion              (Week 3)
B5  both graphs + per-channel GCN + cross-graph attention     (Week 3)
B6  B5 + one-step edge-weight feedback                        (Week 4)
B7  B4 + one-step edge-weight feedback                        (Week 4)

B6/B7 add *no* parameters: they re-use the B5/B4 encoders, per-channel edge heads
and fusion modules, and only insert one bounded edge-weight update plus a second
encoding pass (see :mod:`cofi_gene.models.feedback` and D-022). Models without
feedback keep the Week 2/3 code path bit-for-bit.

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
from .feedback import (DEFAULT_ALPHA, DEFAULT_MARGIN, EdgeWeightFeedback,
                       validate_encoder_graph)
from .fusion import ConcatFusion, GatedFusion
from .gcn import GCNEncoder, normalized_adjacency
from .heads import build_edge_head, decode_pairs

__all__ = ["UpstreamModel", "build_model", "IMPLEMENTED_MODELS", "PLANNED_MODELS",
           "FEEDBACK_MODELS", "BASE_MODEL_OF"]

IMPLEMENTED_MODELS = ("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7")
#: Nothing is refused any more; kept so callers/tests can still ask the registry.
PLANNED_MODELS: tuple[str, ...] = ()

#: Models that must keep one *independent* edge head per channel (Week 3 Part 7.1).
#: B0-B3 keep the single Week 2 head so their behaviour is untouched.
PER_CHANNEL_HEAD_MODELS = ("b4", "b5", "b6", "b7")

#: Models whose fusion keeps two distinguishable per-source representations, so
#: that ``fusion_representations`` is meaningful (B4 outputs a single fused H).
SOURCE_SPECIFIC_MODELS = ("b5", "b6")

#: Week 4: one-step edge-weight feedback. These reuse their base model exactly.
FEEDBACK_MODELS = ("b6", "b7")
BASE_MODEL_OF = {"b6": "b5", "b7": "b4"}

_FUSION_KIND = {"b0": "none", "b1": "none", "b2": "none", "b3": "concat",
                "b4": "gate", "b5": "attention",
                "b6": "attention", "b7": "gate"}


class UpstreamModel(nn.Module):
    """Encodes nodes per channel and exposes one or two edge-scoring heads."""

    def __init__(self, name: str, feature_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 fusion_dim: int = 128, edge_head: str = "dot",
                 attention_heads: int = 4, attention_dropout: float | None = None,
                 attention_residual: bool = False,
                 feedback_alpha: float = DEFAULT_ALPHA,
                 feedback_margin: float = DEFAULT_MARGIN) -> None:
        super().__init__()
        if name not in IMPLEMENTED_MODELS:
            raise NotImplementedError(
                f"model {name!r} is not implemented; implemented: {IMPLEMENTED_MODELS}")
        self.name = name
        # B6/B7 are B5/B4 plus a feedback step; everything else about them is shared.
        self.base_name = BASE_MODEL_OF.get(name, name)
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
            if self.base_name == "b3":
                self.fusion = ConcatFusion(hidden_dim, len(self.channels), fusion_dim)
                self.output_dim = fusion_dim
            elif self.base_name == "b4":
                # Feature-wise gate -> single fused H, already [N, fusion_dim]-wide
                # because hidden_dim == fusion_dim (both 128). No extra projection:
                # an untrained projection would confound the downstream comparison.
                self.fusion = GatedFusion(hidden_dim, num_sources=len(self.channels))
                self.output_dim = hidden_dim
            elif self.base_name == "b5":
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

        # Week 4: only B6/B7 re-encode after one bounded edge-weight update.
        self.feedback = (EdgeWeightFeedback(feedback_alpha, feedback_margin)
                         if name in FEEDBACK_MODELS else None)
        self._encoder_graphs: dict[str, torch.Tensor] | None = None
        self._encoder_weights: dict[str, torch.Tensor] | None = None
        self._base_adjacency: dict[str, torch.Tensor] | None = None
        self._num_graph_nodes: int | None = None

    @property
    def uses_feedback(self) -> bool:
        """True for B6/B7 (they need :meth:`set_encoding_graphs` before forward)."""
        return self.feedback is not None

    def set_encoding_graphs(self, edges: dict[str, torch.Tensor],
                            weights: dict[str, torch.Tensor] | None,
                            num_nodes: int) -> None:
        """Hand the model the **masked training** graphs it is allowed to encode.

        ``edges`` must be each channel's ``encode_edges`` (the training subset),
        never the supervision positives and never validation/test pairs. The
        caller owns that separation; this method re-checks the canonical form
        (undirected, deduplicated, no self-loops, in range) so a wrong tensor
        fails loudly instead of silently leaking.

        Call after ``model.to(device)``: the cached first-pass adjacency is built
        from these tensors and is not moved again.
        """
        if self.feedback is None:
            raise RuntimeError(
                f"model {self.name!r} has no feedback step; "
                f"only {FEEDBACK_MODELS} accept an encoding graph")
        if set(edges) != set(self.channels):
            raise ValueError(f"expected encoder graphs for channels {self.channels}, "
                             f"got {tuple(sorted(edges))}")
        num_nodes = int(num_nodes)
        graphs, scaled, base = {}, {}, {}
        for channel in self.channels:
            edge_index = validate_encoder_graph(edges[channel], num_nodes)
            weight = None
            if weights is not None and weights.get(channel) is not None:
                weight = torch.as_tensor(weights[channel]).reshape(-1).to(
                    device=edge_index.device, dtype=torch.float32)
                if weight.numel() != edge_index.shape[1]:
                    raise ValueError(f"channel {channel}: {weight.numel()} weights for "
                                     f"{edge_index.shape[1]} encoder edges")
            graphs[channel] = edge_index
            scaled[channel] = weight
            base[channel] = normalized_adjacency(edge_index, weight, num_nodes)
        self._encoder_graphs = graphs
        self._encoder_weights = scaled
        self._base_adjacency = base
        self._num_graph_nodes = num_nodes

    def encoding_graphs(self) -> dict[str, torch.Tensor]:
        """Read-only view of the tensors registered by :meth:`set_encoding_graphs`."""
        if self._encoder_graphs is None:
            raise RuntimeError("no encoding graph registered; call "
                               "set_encoding_graphs() first")
        return self._encoder_graphs


    # ------------------------------------------------------------- encoding
    def forward(self, x: torch.Tensor, adjacencies: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return ``channel -> final embedding`` used for that channel's edge scoring.

        Every model hands the head one trained ``[N, output_dim]`` vector, so the
        decoder never reads an untrained intermediate. B5/B6 additionally expose
        their two source-specific representations through
        :meth:`fusion_representations`; B4/B7 fuse to a single vector.

        B6/B7 run one feedback + re-encoding step first (``_feedback_pass``); every
        other model uses the Week 2/3 path unchanged, with ``adjacencies`` as given.
        """
        if self.name == "b0":
            embedding = self.input_projection(x)
            return {channel: embedding for channel in self.channels}
        if self.feedback is not None:
            return self._fuse(self._feedback_pass(x))
        return self._fuse(self._encode(x, adjacencies))

    def _encode(self, x: torch.Tensor,
                adjacencies: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """One independent GCN pass per channel on the given adjacency."""
        per_channel = {}
        for channel in self.channels:
            if channel not in adjacencies:
                raise KeyError(f"model {self.name} needs adjacency for channel {channel!r}")
            per_channel[channel] = self.encoders[channel](x, adjacencies[channel])
        return per_channel

    def _fuse(self, per_channel: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply this model's fusion kind; B6/B7 inherit B5/B4 respectively."""
        kind = self.fusion_kind
        if kind == "concat":
            fused = self.fusion([per_channel[channel] for channel in self.channels])
            return {channel: fused for channel in self.channels}
        if kind == "gate":
            # GatedFusion already returns the single [N, hidden_dim] fused vector.
            return self.fusion(per_channel)
        if kind == "attention":
            sources = self.fusion(per_channel)          # Z_A, Z_B  [N, hidden_dim]
            fused = self.output_projection([sources[channel] for channel in self.channels])
            return {channel: fused for channel in self.channels}
        return per_channel

    def _feedback_pass(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """B6/B7: encode -> score training edges -> reweight -> encode again.

        Both passes share the same encoders (no new parameters). The weight update
        is detached (see :mod:`cofi_gene.models.feedback`), so gradients only flow
        through the second encoding and the scoring head, never through the
        structure edit. Only the graphs registered via
        :meth:`set_encoding_graphs` -- i.e. each channel's masked training graph --
        are ever read here.
        """
        if self._base_adjacency is None or self._encoder_graphs is None \
                or self._num_graph_nodes is None:
            raise RuntimeError(
                f"model {self.name!r} needs feedback: call set_encoding_graphs("
                "encode_edges, encode_weights, num_nodes) before forward()")
        first = self._encode(x, self._base_adjacency)
        second = {}
        for channel in self.channels:
            _, adjacency = self.feedback(
                first[channel], self._encoder_graphs[channel],
                (None if self._encoder_weights is None
                 else self._encoder_weights.get(channel)),
                self.head_for(channel), self._num_graph_nodes)
            second[channel] = self.encoders[channel](x, adjacency)
        return second

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
