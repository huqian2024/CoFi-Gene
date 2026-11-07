"""Model math and the B0-B3 registry (needs the optional torch extra)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cofi_gene.models import (IMPLEMENTED_MODELS, PLANNED_MODELS, UpstreamModel,  # noqa: E402
                             build_adjacencies, build_model)
from cofi_gene.models.fusion import ConcatFusion  # noqa: E402
from cofi_gene.models.gcn import GCNEncoder, GraphConv, normalized_adjacency  # noqa: E402
from cofi_gene.models.heads import build_edge_head  # noqa: E402


def test_normalized_adjacency_is_symmetric_and_self_looped():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    adjacency = normalized_adjacency(edge_index, None, num_nodes=5)   # node 4 is isolated
    dense = adjacency.to_dense()
    assert torch.allclose(dense, dense.T), "undirected graph must give a symmetric operator"
    assert torch.all(adjacency.values() > 0)
    # degree(node0) = 1 edge + 1 self loop = 2 -> self weight 1/sqrt(2*2)
    assert dense[0, 0].item() == pytest.approx(0.5)
    # an isolated node keeps only its self loop, so the weight is exactly 1
    assert dense[4, 4].item() == pytest.approx(1.0)
    assert torch.all(dense[4, :4] == 0)
    assert dense.diagonal().sum().item() > 0


def test_normalized_adjacency_accepts_weights_and_validates_length():
    edge_index = torch.tensor([[0], [1]])
    weighted = normalized_adjacency(edge_index, torch.tensor([900.0]), num_nodes=2)
    unweighted = normalized_adjacency(edge_index, None, num_nodes=2)
    assert weighted.to_dense()[0, 1] > unweighted.to_dense()[0, 1]
    with pytest.raises(ValueError, match="length"):
        normalized_adjacency(edge_index, torch.tensor([1.0, 2.0]), num_nodes=2)
    with pytest.raises(ValueError, match="shape"):
        normalized_adjacency(torch.zeros(3), None, num_nodes=2)


def test_gcn_layers_shape_and_gradients():
    adjacency = normalized_adjacency(torch.tensor([[0, 1], [1, 2]]), None, num_nodes=4)
    x = torch.randn(4, 5, requires_grad=True)
    layer = GraphConv(5, 3)
    assert layer(x, adjacency).shape == (4, 3)
    encoder = GCNEncoder(in_dim=5, hidden_dim=8, num_layers=2, dropout=0.0, out_dim=6)
    out = encoder(x, adjacency)
    assert out.shape == (4, 6)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    with pytest.raises(ValueError, match="num_layers"):
        GCNEncoder(4, 4, num_layers=0)


def test_concat_fusion_projects_back_to_single_width():
    fusion = ConcatFusion(in_dim=6, num_sources=2, out_dim=6)
    merged = fusion([torch.randn(5, 6), torch.randn(5, 6)])
    assert merged.shape == (5, 6)
    with pytest.raises(ValueError, match="expected 2"):
        fusion([torch.randn(5, 6)])


def test_edge_heads_score_pairs():
    embeddings = torch.randn(4, 5)
    pairs = torch.tensor([[0, 1], [1, 3]])
    dot = build_edge_head("dot", 5)
    scores = dot(embeddings, pairs)
    assert scores.shape == (2,)
    assert scores[0].item() == pytest.approx(float((embeddings[0] * embeddings[1]).sum()))
    mlp = build_edge_head("mlp", 5)
    assert mlp(embeddings, pairs).shape == (2,)
    assert dot(embeddings, torch.zeros((2, 0), dtype=torch.long)).shape == (0,)
    with pytest.raises(ValueError, match="unknown edge head"):
        build_edge_head("transformer", 5)


@pytest.mark.parametrize("name", IMPLEMENTED_MODELS)
def test_registry_builds_every_implemented_model(name):
    # hidden_dim=8 is divisible by the 4 attention heads of B5 (Week 3), and
    # feature_dim=4 != fusion_dim keeps B0's input projection non-trivial. Every
    # model therefore lands on 8 wide, so the Week 2 assertions are unchanged.
    model = build_model(name, feature_dim=4, hidden_dim=8, fusion_dim=8, dropout=0.0)
    adjacency = normalized_adjacency(torch.tensor([[0, 1], [1, 2]]), None, num_nodes=4)
    x = torch.randn(4, 4)
    outputs = model(x, {channel: adjacency for channel in model.channels})
    assert set(outputs) == set(model.channels)
    for embedding in outputs.values():
        assert embedding.shape == (4, 8)
    assert model.parameter_count() > 0
    assert model.head is not None


def test_b0_ignores_adjacency_because_it_has_no_message_passing():
    model = build_model("b0", feature_dim=8, fusion_dim=8)
    x = torch.randn(4, 8)
    adjacency = normalized_adjacency(torch.tensor([[0], [1]]), None, num_nodes=4)
    other = normalized_adjacency(torch.tensor([[2], [3]]), None, num_nodes=4)
    assert torch.allclose(model(x, {"A": adjacency, "B": other})["A"],
                          model(x, {"A": other, "B": adjacency})["A"])


def test_b3_fuses_both_views_so_channels_share_the_output():
    model = build_model("b3", feature_dim=8, hidden_dim=6, fusion_dim=6, dropout=0.0)
    adjacency = normalized_adjacency(torch.tensor([[0, 1], [1, 2]]), None, num_nodes=4)
    outputs = model(torch.randn(4, 8), {"A": adjacency, "B": adjacency})
    assert torch.allclose(outputs["A"], outputs["B"])


def test_b1_requires_its_own_channel_only():
    model = build_model("b1", feature_dim=8, hidden_dim=6, dropout=0.0)
    adjacency = normalized_adjacency(torch.tensor([[0], [1]]), None, num_nodes=4)
    assert model.channels == ("A",)
    with pytest.raises(KeyError, match="adjacency for channel 'A'"):
        model(torch.randn(4, 8), {"B": adjacency})


@pytest.mark.parametrize("name", PLANNED_MODELS)
def test_planned_models_are_refused_with_a_clear_message(name):
    with pytest.raises(NotImplementedError, match=f"{name!r} is not implemented"):
        build_model(name, feature_dim=8)


def test_build_adjacencies_helper_covers_requested_channels():
    edges = {"A": torch.tensor([[0, 1], [1, 2]]), "B": torch.tensor([[0], [2]])}
    weights = {"A": torch.ones(2), "B": torch.tensor([0.5])}
    adjacencies = build_adjacencies(edges, weights, num_nodes=3)
    assert set(adjacencies) == {"A", "B"}
    assert adjacencies["A"].is_sparse
    assert isinstance(UpstreamModel("b1", feature_dim=4), UpstreamModel)
    assert np.isfinite(adjacencies["B"].to_dense().numpy()).all()
