"""B5: symmetric bidirectional cross-graph attention over two evidence channels.

Approved Week 3 design (``docs/WEEK3_IMPLEMENTATION_REPORT.md`` §6): four heads of
32 dimensions each (``d = 128``), both channels acting as Query sources.

Why this exact shape (AGENTS.md 4.2 / Week 3 instruction 6)
----------------------------------------------------------
* Each Query must see at least two candidate Key/Value representations, otherwise
  the softmax over candidates is identically 1 and carries no information. Here
  every gene ``i`` offers **two** candidates: its own-graph representation and its
  other-graph representation. The softmax therefore has exactly two options and
  learns a genuine relative preference ``(own, other)``.
* Attention is **symmetric**: both channels act as Query sources and both are
  attended, so information flows A -> B and B -> A instead of one view merely
  reading the other.
* Keys/Values use **source-specific projections**, so the same node representation
  is not forced to play the same role in both graphs.
* The attention runs *per gene over the two evidence views*; neighbourhood
  information already entered through the GCN encoders. This is deliberately not
  all-gene Transformer attention (out of scope).

Equations (one node ``i``, query source ``s``, candidate source ``t``, head ``m``)::

    q^m_{s,i} = h_{s,i} W_q^{s,m}                    (d -> d_head = d / H)
    k^m_{t,i} = h_{t,i} W_k^{t,m}
    v^m_{t,i} = h_{t,i} W_v^{t,m}
    e^m_{s,t,i} = (q^m_{s,i} . k^m_{t,i}) / sqrt(d_head)      s, t in {A, B}
    a^m_{s,.,i} = softmax_t e^m_{s,.,i}                       (softmax over 2 candidates)
    o^m_{s,i}   = sum_t a^m_{s,t,i} v^m_{t,i}
    z^m_{s,i}   = o^m_{s,i}                                   (residual=False, default)
    z^m_{s,i}   = h^m_{s,i} + o^m_{s,i}                       (residual=True, opt-in)
    Z_{s,i}     = [ z^1_{s,i} ; ... ; z^H_{s,i} ]             d_head * H = d

``Z_A`` and ``Z_B`` are then concatenated by the model and projected back to 128
dimensions (see `cofi_gene.models.UpstreamModel`).

Tensor shapes: ``h`` is ``[N, 2, d]`` (stacked over the two channels),
``q, k, v`` are ``[N, 2, H, d_head]``, scores/weights are ``[N, 2, H, 2]``
(query channel, head, key channel) and both outputs are ``[N, d]``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["CrossGraphAttention"]


class CrossGraphAttention(nn.Module):
    """Two-token, source-aware, symmetric attention between exactly two channels."""

    def __init__(self, dim: int, channels, num_heads: int = 4, dropout: float = 0.0,
                 residual: bool = False, bias: bool = True) -> None:
        super().__init__()
        channels = tuple(channels)
        if len(channels) != 2:
            raise ValueError("CrossGraphAttention needs exactly two channels, "
                             f"got {channels}")
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dim = dim
        self.channels = channels
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.dropout = float(dropout)
        self.residual = bool(residual)
        # Source-specific projections: a node's role differs per evidence channel.
        self.query = nn.ModuleDict({channel: nn.Linear(dim, dim, bias=bias)
                                    for channel in channels})
        self.key = nn.ModuleDict({channel: nn.Linear(dim, dim, bias=bias)
                                  for channel in channels})
        self.value = nn.ModuleDict({channel: nn.Linear(dim, dim, bias=bias)
                                    for channel in channels})


    # ------------------------------------------------------------- internals
    def _stack(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """``{channel: [N, d]}`` -> ``[N, 2, d]`` in the registered channel order."""
        if set(embeddings) != set(self.channels):
            raise ValueError(f"expected embeddings for channels {self.channels}, "
                             f"got {tuple(sorted(embeddings))}")
        return torch.stack([embeddings[channel] for channel in self.channels], dim=1)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        nodes = tensor.shape[0]
        return tensor.view(nodes, len(self.channels), self.num_heads, self.head_dim)

    def _project(self, embeddings: dict[str, torch.Tensor]):
        hidden = self._stack(embeddings)
        queries = self._split_heads(
            torch.stack([self.query[c](hidden[:, i]) for i, c in enumerate(self.channels)],
                        dim=1))
        keys = self._split_heads(
            torch.stack([self.key[c](hidden[:, i]) for i, c in enumerate(self.channels)],
                        dim=1))
        values = self._split_heads(
            torch.stack([self.value[c](hidden[:, i]) for i, c in enumerate(self.channels)],
                        dim=1))
        return hidden, queries, keys, values

    def _weights(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Softmax over the two candidate sources -> ``[N, 2, H, 2]``."""
        scores = torch.einsum("nqhd,nkhd->nqhk", queries, keys) / math.sqrt(self.head_dim)
        return torch.softmax(scores, dim=-1)

    # ---------------------------------------------------------------- public
    def attention_weights(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Rank-4 attention weights ``[N, query_channel, head, key_channel]``.

        The last axis always has exactly two entries and sums to 1, so a
        degenerate "single candidate" attention (weight pinned to 1) is
        impossible by construction.
        """
        _, queries, keys, _ = self._project(embeddings)
        return self._weights(queries, keys)

    def forward(self, embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hidden, queries, keys, values = self._project(embeddings)
        weights = self._weights(queries, keys)
        context = torch.einsum("nqhk,nkhd->nqhd", weights, values)
        # concat the H heads of each direction -> [N, 2, d]
        context = context.reshape(hidden.shape[0], len(self.channels), self.dim)
        outputs = {}
        for index, channel in enumerate(self.channels):
            update = nn.functional.dropout(context[:, index], p=self.dropout,
                                          training=self.training)
            outputs[channel] = hidden[:, index] + update if self.residual else update
        return outputs


