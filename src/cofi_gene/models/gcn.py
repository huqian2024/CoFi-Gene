"""Two-layer GCN encoder on torch sparse tensors (no PyG dependency).

Why not PyTorch Geometric for M2-M4: the math here is a normalised sparse
matrix product, PyG adds a build-time coupling to specific torch/CUDA wheels, and
the RTX 5090 server needs a very recent CUDA build. Keeping to ``torch.sparse``
means the same code runs on a CPU laptop (used by the tests) and on the GPU box.
This is an engineering choice, recorded as D-013; it changes no research claim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["normalized_adjacency", "GraphConv", "GCNEncoder"]


def normalized_adjacency(edge_index, edge_weight=None, num_nodes: int | None = None,
                         add_self_loops: bool = True,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Symmetric normalised adjacency ``D^-1/2 (A + I) D^-1/2`` as a sparse COO tensor.

    ``edge_index`` is ``[2, E]`` with both endpoints already canonical; the matrix
    is symmetrised here, so the caller never has to store reverse edges.
    """
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.long()
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 1
    row, col = edge_index[0], edge_index[1]
    if edge_weight is None:
        base_values = torch.ones(row.numel(), dtype=dtype)
    else:
        base_values = torch.as_tensor(edge_weight, dtype=dtype).reshape(-1)
        if base_values.numel() != row.numel():
            raise ValueError("edge_weight length must match edge_index")
    # symmetrise: every stored edge contributes (u->v) and (v->u)
    row = torch.cat([row, col])
    col = torch.cat([col, edge_index[0]])
    values = torch.cat([base_values, base_values])
    if add_self_loops:
        loops = torch.arange(num_nodes, dtype=torch.long)
        row = torch.cat([row, loops])
        col = torch.cat([col, loops])
        values = torch.cat([values, torch.ones(num_nodes, dtype=dtype)])
    indices = torch.stack([row, col])
    adjacency = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes),
                                        dtype=dtype, check_invariants=False).coalesce()
    degree = torch.zeros(num_nodes, dtype=dtype)
    degree = degree.index_add(0, adjacency.indices()[0], adjacency.values())
    degree = degree.clamp(min=1e-12)
    norm = degree.pow(-0.5)
    scaled = adjacency.values() * norm[adjacency.indices()[0]] * norm[adjacency.indices()[1]]
    return torch.sparse_coo_tensor(adjacency.indices(), scaled, (num_nodes, num_nodes),
                                   dtype=dtype, check_invariants=False).coalesce()


class GraphConv(nn.Module):
    """``X' = A_norm (X W + b)`` - one message-passing step."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adjacency, self.linear(x))


class GCNEncoder(nn.Module):
    """Stack of :class:`GraphConv` layers used by B1/B2/B3 (and later B4-B7)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.2, out_dim: int | None = None) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        out_dim = hidden_dim if out_dim is None else out_dim
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(GraphConv(dims[i], dims[i + 1])
                                    for i in range(num_layers))
        self.dropout = dropout
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, adjacency)
            if index < len(self.layers) - 1:
                hidden = F.relu(hidden)
                hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return hidden
