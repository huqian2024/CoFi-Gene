"""Week 7 figure 3: export a local graph-feedback case (B6/B7).

The third required figure shows *what one feedback step actually changed*. To keep
it honest, the numbers come from the same code path that trains B6/B7:

``first = GCN_channel(X, A)``  →  ``s = head_channel(first, e)``  →
``w' = clamp(w·(1+α(2σ(s)−1)), (1−m)w, (1+m)w)``

i.e. :func:`cofi_gene.models.feedback.feedback_weights`, evaluated on the
**encoder graph the model is allowed to see** (never val/test edges). This module
turns the resulting per-edge records into a JSON case that
:func:`cofi_gene.figures.graph_case` can draw, and that a report can cite instead
of "it looks like some edges changed".

No training, no data mutation: callers pass an already trained model.
"""

from __future__ import annotations

import torch

from .models.feedback import feedback_weights
from .models.gcn import normalized_adjacency

__all__ = ["feedback_edge_table", "select_case"]


def feedback_edge_table(model, x, channel: str, num_nodes: int,
                        gene_ids=None, top_k: int | None = None,
                        focus_genes=None) -> dict:
    """Per-edge ``before``/``after`` weights of one feedback step.

    ``model`` must be B6/B7 with :meth:`set_encoding_graphs` already called.
    ``gene_ids`` maps node indices back to symbols for the figure; without it the
    rows fall back to ``node<index>`` labels.
    """
    if not getattr(model, "uses_feedback", False):
        raise ValueError(f"model {getattr(model, 'name', '?')!r} has no feedback step; "
                         "only B6/B7 can export a feedback case")
    if channel not in model.channels:
        raise ValueError(f"channel {channel!r} is not used by model {model.name!r}; "
                         f"available: {model.channels}")
    edges = model.encoding_graphs()[channel]
    weight = (model.encoding_weights() or {}).get(channel)
    if edges.numel() == 0:
        raise ValueError(f"channel {channel!r} has an empty encoder graph; nothing to "
                         "report")
    adjacency = normalized_adjacency(edges, weight, int(num_nodes))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        first = model.encoders[channel](x, adjacency)
        scores = model.score_edges(first, edges, channel=channel)
        updated = feedback_weights(weight, scores, model.feedback.alpha,
                                   model.feedback.margin)
    if was_training:
        model.train()

    def label(index: int) -> str:
        if gene_ids is not None and 0 <= index < len(gene_ids):
            return str(gene_ids[index])
        return f"node{index}"

    source = edges[0].tolist()
    target = edges[1].tolist()
    before = updated.new_zeros(updated.numel()) if weight is None else \
        torch.as_tensor(weight).reshape(-1).to(updated.device)
    rows = []
    for index, (u, v) in enumerate(zip(source, target)):
        w_before = float(before[index])
        w_after = float(updated[index])
        rows.append({
            "u": int(u), "v": int(v), "u_id": label(int(u)), "v_id": label(int(v)),
            "weight_before": w_before,
            "weight_after": w_after,
            "delta": w_after - w_before,
            "score": float(scores[index]),
            "probability": float(torch.sigmoid(scores[index])),
        })
    table = {
        "channel": channel,
        "model": model.name,
        "alpha": float(model.feedback.alpha),
        "margin": float(model.feedback.margin),
        "num_nodes": int(num_nodes),
        "num_encoder_edges": len(rows),
        "edges": rows,
    }
    return select_case(table, top_k=top_k, focus_genes=focus_genes, gene_ids=gene_ids)


def select_case(table: dict, top_k: int | None = None, focus_genes=None,
                gene_ids=None) -> dict:
    """Keep the ``top_k`` largest changes and summarise the case.

    ``focus_genes`` restricts the rows to edges incident to those symbols (a
    biologically readable local case); the summary always counts the *selected*
    rows, and the number of candidate rows is kept in ``num_candidate_edges``.
    """
    rows = list(table["edges"])
    candidates = len(rows)
    focus = [str(gene) for gene in (focus_genes or [])]
    if focus:
        lookup = {}
        if gene_ids is not None:
            lookup = {str(gene): index for index, gene in enumerate(gene_ids)}
        wanted = {lookup.get(gene) for gene in focus}
        wanted = {index for index in wanted if index is not None}
        rows = [row for row in rows if row["u"] in wanted or row["v"] in wanted]
        if not rows:
            raise ValueError(f"none of the focus genes {focus} has an encoder edge in "
                             "this channel")
    rows.sort(key=lambda row: abs(row["delta"]), reverse=True)
    if top_k is not None:
        if int(top_k) < 1:
            raise ValueError("top_k must be >= 1")
        rows = rows[: int(top_k)]

    deltas = [row["delta"] for row in rows]
    nodes = sorted({row["u_id"] for row in rows} | {row["v_id"] for row in rows})
    summary = {
        "num_candidate_edges": candidates,
        "num_selected_edges": len(rows),
        "num_up": sum(1 for delta in deltas if delta > 1e-12),
        "num_down": sum(1 for delta in deltas if delta < -1e-12),
        "num_unchanged": sum(1 for delta in deltas if abs(delta) <= 1e-12),
        "max_abs_delta": max((abs(delta) for delta in deltas), default=0.0),
        "mean_abs_delta": (sum(abs(delta) for delta in deltas) / len(deltas))
        if deltas else 0.0,
    }
    return {**{key: value for key, value in table.items() if key != "edges"},
            "focus_genes": focus,
            "nodes": [{"id": node} for node in nodes],
            "edges": rows,
            "summary": summary}
