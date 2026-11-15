"""B6/B7 (M4): one-step edge-weight feedback on the masked training graph.

Rationale (AGENTS.md 4.3, ``docs/DECISIONS.md`` D-022)
-----------------------------------------------------
Both evidence graphs are incomplete and noisy. A single feedback step lets an
already-trained edge head correct the weights of the edges it is *allowed* to
see, after which the GCN encodes once more on the corrected graph::

    H_c^(1) = GCN_c(X, A_c)                        # first pass
    s_e     = head_c(H_c^(1), e)                   # that channel's own edge head
    p_e     = sigmoid(detach(s_e))                 # stop-gradient
    w'_e    = clip(w_e * (1 + alpha * (2 p_e - 1)),
                   (1 - margin) * w_e,
                   (1 + margin) * w_e)
    A'_c    = D^-1/2 (A'_c + I) D^-1/2             # topology unchanged
    H_c^(2) = GCN_c(X, A'_c)                       # second pass, shared weights

Invariants that must not drift silently
---------------------------------------
* Only **edge weights** of the already-masked training graph change. The edge
  set is never extended (no candidate-edge insertion), and no edge is ever read
  from the validation/test splits or from another channel's graph.
* ``detach`` keeps the weight update out of the autograd graph, so the model
  cannot learn to raise a score by lowering the weight it is scored on. The
  second pass still gets gradients through its own embeddings and head.
* ``alpha == 0`` is an exact no-op, so ``feedback_alpha: 0`` reproduces the
  B5/B4 counterpart bit-for-bit.
* The module carries **no parameters**, so B6/B7 stay parameter-matched with
  B5/B4 (the ablation isolates the feedback step, not extra capacity).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gcn import normalized_adjacency
from .heads import decode_pairs

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_MARGIN",
    "feedback_weights",
    "validate_encoder_graph",
    "EdgeWeightFeedback",
]

DEFAULT_ALPHA = 0.1
DEFAULT_MARGIN = 0.5


def feedback_weights(weights, scores, alpha: float = DEFAULT_ALPHA,
                     margin: float = DEFAULT_MARGIN) -> torch.Tensor:
    """Reweight existing edges from their scores; the edge set is untouched.

    ``scores`` are the raw edge-head logits ``[E]``; they are detached here so the
    update never back-propagates into the weights. The result is clamped to
    ``[w*(1-margin), w*(1+margin)]`` (and to ``>= 0``), which keeps every update
    bounded and finite for any logit magnitude.

    ``alpha == 0`` returns the input weights unchanged (exact no-op).
    """
    alpha = float(alpha)
    margin = float(margin)
    if alpha < 0.0:
        raise ValueError(f"feedback alpha must be >= 0, got {alpha}")
    if not 0.0 <= margin <= 1.0:
        raise ValueError(f"feedback margin must be in [0, 1], got {margin}")

    scores = scores.detach().reshape(-1)
    if weights is None:
        current = torch.ones_like(scores)
    else:
        current = weights.detach() if torch.is_tensor(weights) else torch.as_tensor(weights)
        current = current.reshape(-1).to(device=scores.device, dtype=scores.dtype)
    if current.numel() != scores.numel():
        raise ValueError(f"edge weights ({current.numel()}) and scores "
                         f"({scores.numel()}) must describe the same edges")
    if current.numel() == 0 or alpha == 0.0:
        return current.clone()
    if not torch.isfinite(current).all():
        raise ValueError("edge weights must be finite before feedback")

    probabilities = torch.sigmoid(scores)
    updated = current * (1.0 + alpha * (2.0 * probabilities - 1.0))
    lower = (1.0 - margin) * current
    upper = (1.0 + margin) * current
    updated = torch.maximum(torch.minimum(updated, upper), lower)
    updated = torch.clamp(updated, min=0.0)
    if not torch.isfinite(updated).all():  # pragma: no cover - defensive
        raise ValueError("edge feedback produced non-finite weights")
    return updated


def validate_encoder_graph(edge_index, num_nodes: int) -> torch.Tensor:
    """Assert the graph handed to the feedback step is a canonical training graph.

    ``CoFiDataset`` stores each channel's encoder graph as sorted, deduplicated,
    self-loop-free pairs of the **training** split. Re-checking here (once per
    run) catches a caller that passes the supervision positives or a full graph:
    that would be a leakage bug, not a metric regression.
    """
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long()
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")
    if edge_index.numel() == 0:
        return edge_index
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= int(num_nodes):
        raise ValueError(f"edge_index outside [0, {num_nodes})")
    source, target = edge_index[0], edge_index[1]
    if bool(torch.any(source == target)):
        raise ValueError("the encoder graph must not contain self-loops")
    keys = torch.minimum(source, target) * int(num_nodes) + torch.maximum(source, target)
    if int(torch.unique(keys).numel()) != int(keys.numel()):
        raise ValueError("the encoder graph must not contain duplicate pairs "
                         "(it is a set of undirected training edges)")
    return edge_index


class EdgeWeightFeedback(nn.Module):
    """Stateless (alpha, margin) holder: score edges, reweight, rebuild adjacency."""

    def __init__(self, alpha: float = DEFAULT_ALPHA,
                 margin: float = DEFAULT_MARGIN) -> None:
        super().__init__()
        alpha = float(alpha)
        margin = float(margin)
        if alpha < 0.0:
            raise ValueError(f"feedback alpha must be >= 0, got {alpha}")
        if not 0.0 <= margin <= 1.0:
            raise ValueError(f"feedback margin must be in [0, 1], got {margin}")
        self.alpha = alpha
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, edge_index: torch.Tensor,
                edge_weight, head: nn.Module, num_nodes: int):
        """Return ``(new_weights, normalised_feedback_adjacency)``.

        ``head`` is the caller's per-channel edge head, i.e. the same decoder the
        link-prediction loss uses; no extra parameters are introduced.
        """
        scores = decode_pairs(head, embeddings, edge_index)
        updated = feedback_weights(edge_weight, scores, self.alpha, self.margin)
        adjacency = normalized_adjacency(edge_index, updated, num_nodes)
        return updated, adjacency

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return f"alpha={self.alpha}, margin={self.margin}"

