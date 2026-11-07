"""Week 3 (M3): B4 gated fusion and B5 cross-graph attention.

The tests are deliberately written against *behaviour* (shapes, gradients, the
mathematical identity of the fusion, the number of attention candidates) rather
than against private attribute names, so the modules stay refactorable.

Small synthetic graphs keep the suite fast; nothing here reads real data or a GPU.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cofi_gene.config import load_config  # noqa: E402
from cofi_gene.models import (IMPLEMENTED_MODELS, PLANNED_MODELS,  # noqa: E402
                              build_model)
from cofi_gene.models.attention import CrossGraphAttention  # noqa: E402
from cofi_gene.models.fusion import GatedFusion  # noqa: E402
from cofi_gene.models.gcn import normalized_adjacency  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
WEEK3_MODELS = ("b4", "b5")


def _two_graphs(num_nodes=6):
    """Two *different* graphs on the same node set (A triangle-ish, B a path)."""
    return {
        "A": normalized_adjacency(torch.tensor([[0, 1, 2], [1, 2, 3]]), None,
                                  num_nodes=num_nodes),
        "B": normalized_adjacency(torch.tensor([[3, 4], [4, 5]]), None,
                                  num_nodes=num_nodes),
    }


# --------------------------------------------------------------------- shapes
@pytest.mark.parametrize("name", WEEK3_MODELS)
def test_week3_models_emit_one_128_dim_embedding_per_channel(name):
    model = build_model(name, feature_dim=32, hidden_dim=128, fusion_dim=128, dropout=0.0)
    x = torch.randn(6, 32)
    adjacency = _two_graphs()
    outputs = model(x, adjacency)
    assert set(outputs) == {"A", "B"}
    for embedding in outputs.values():
        assert embedding.shape == (6, 128), embedding.shape
    assert model.encode(x, adjacency).shape == (6, 128)
    assert model.output_dim == 128


@pytest.mark.parametrize("name", WEEK3_MODELS)
def test_week3_models_backpropagate_finite_gradients_everywhere(name):
    model = build_model(name, feature_dim=16, hidden_dim=8, fusion_dim=8, dropout=0.0)
    x = torch.randn(6, 16)
    adjacency = _two_graphs()
    outputs = model(x, adjacency)
    pairs = torch.tensor([[0, 1], [1, 2]])
    loss = sum(torch.nn.functional.binary_cross_entropy_with_logits(
        model.score_edges(outputs[channel], pairs, channel=channel),
        torch.ones(pairs.shape[1])) for channel in model.channels)
    loss.backward()
    # Every parameter must carry a gradient: a silently frozen fusion module would
    # still "train" the encoders and would silently weaken the ablation.
    missing = [name_ for name_, parameter in model.named_parameters()
               if parameter.grad is None]
    assert missing == [], missing
    for parameter in model.parameters():
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("name", WEEK3_MODELS)
def test_week3_models_are_deterministic_in_eval_mode(name):
    torch.manual_seed(0)
    model = build_model(name, feature_dim=16, hidden_dim=8, fusion_dim=8, dropout=0.0)
    model.eval()
    x = torch.randn(6, 16)
    adjacency = _two_graphs()
    with torch.no_grad():
        first = model.encode(x, adjacency)
        second = model.encode(x, adjacency)
    assert torch.allclose(first, second)
    assert torch.isfinite(first).all()


# ------------------------------------------------------------- B4 gated fusion
def _fixed_gate(fusion: GatedFusion, bias) -> GatedFusion:
    """Freeze the gate to ``sigmoid(bias)`` so the formula becomes checkable."""
    with torch.no_grad():
        fusion.gate.weight.zero_()                 # single Linear, no hidden layer
        fusion.gate.bias.copy_(torch.as_tensor(bias, dtype=fusion.gate.bias.dtype))
    return fusion


def test_gated_fusion_is_a_single_linear_layer():
    """Approved design: one Linear(2d -> d) + Sigmoid, no hidden layer / ReLU."""
    fusion = GatedFusion(8)
    assert isinstance(fusion.gate, torch.nn.Linear)
    assert fusion.gate.in_features == 16 and fusion.gate.out_features == 8


def test_gated_fusion_implements_the_documented_equation():
    dim = 4
    fusion = _fixed_gate(GatedFusion(dim), bias=[10.0, -10.0, 0.0, 0.0])  # g ~ [1, 0, .5, .5]
    h_a = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    h_b = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    gate = fusion.gate_weight({"A": h_a, "B": h_b})
    assert gate.shape == (1, dim)
    assert ((gate >= 0) & (gate <= 1)).all()
    assert gate[0, 0].item() > 0.99 and gate[0, 1].item() < 0.01
    outputs = fusion({"A": h_a, "B": h_b})
    fused = outputs["A"]
    # H = g * H_A + (1 - g) * H_B   with one shared output vector
    assert torch.allclose(fused, gate * h_a + (1 - gate) * h_b, atol=1e-5)
    assert torch.allclose(outputs["A"], outputs["B"])
    assert fused[0, 0].item() == pytest.approx(1.0, abs=1e-3)   # gate ~1: keeps A
    assert fused[0, 1].item() == pytest.approx(3.0, abs=1e-2)   # gate ~0: takes B


def test_gated_fusion_at_half_gate_is_the_plain_average():
    fusion = _fixed_gate(GatedFusion(5), bias=[0.0] * 5)
    h_a, h_b = torch.randn(3, 5), torch.randn(3, 5)
    outputs = fusion({"A": h_a, "B": h_b})
    assert torch.allclose(outputs["A"], 0.5 * (h_a + h_b))
    assert torch.allclose(outputs["A"], outputs["B"])


def test_gate_is_feature_wise_not_one_scalar_per_graph():
    """Every gene must get its own d gate values, not a single mixing weight."""
    fusion = GatedFusion(6)
    torch.manual_seed(0)
    gate = fusion.gate_weight({"A": torch.randn(5, 6), "B": torch.randn(5, 6)})
    assert gate.shape == (5, 6)
    assert gate.std().item() > 0                     # not a constant
    assert not torch.allclose(gate[0], gate[1])      # differs across genes


def test_gated_fusion_reads_both_channels():
    """Changing the *other* channel must change the fused output."""
    fusion = GatedFusion(6)
    h_a = torch.randn(4, 6)
    h_b = torch.randn(4, 6)
    baseline = fusion({"A": h_a, "B": h_b})["A"]
    moved = fusion({"A": h_a, "B": h_b + 5.0})["A"]
    assert not torch.allclose(baseline, moved)


def test_gated_fusion_reaches_both_encoders_with_gradients():
    model = build_model("b4", feature_dim=8, hidden_dim=8, fusion_dim=8, dropout=0.0)
    x = torch.randn(6, 8)
    outputs = model(x, _two_graphs())["A"]
    outputs.sum().backward()
    assert model.fusion.gate.weight.grad is not None
    assert model.fusion.gate.weight.grad.abs().sum() > 0
    for channel in model.channels:
        first_layer = model.encoders[channel].layers[0].linear
        assert first_layer.weight.grad is not None
        assert first_layer.weight.grad.abs().sum() > 0


def test_gated_fusion_validates_its_inputs():
    with pytest.raises(ValueError, match="two-channel"):
        GatedFusion(4, num_sources=3)
    fusion = GatedFusion(4)
    with pytest.raises(ValueError, match="expected 2"):
        fusion({"A": torch.randn(3, 4)})
    with pytest.raises(ValueError, match="shape"):
        fusion({"A": torch.randn(3, 4), "B": torch.randn(2, 4)})
    with pytest.raises(ValueError, match="width"):
        fusion({"A": torch.randn(3, 5), "B": torch.randn(3, 5)})


# -------------------------------------------------------- B5 attention module
def test_attention_uses_four_heads_of_32_dims_by_default():
    """Approved design: H = 4, d_head = 128 / 4 = 32."""
    attention = CrossGraphAttention(128, ("A", "B"))
    assert attention.num_heads == 4
    assert attention.head_dim == 32
    weights = attention.attention_weights({"A": torch.randn(3, 128),
                                           "B": torch.randn(3, 128)})
    assert weights.shape == (3, 2, 4, 2)


def test_attention_has_exactly_two_candidate_keys_per_query():
    attention = CrossGraphAttention(8, ("A", "B"), num_heads=2, dropout=0.0)
    embeddings = {"A": torch.randn(5, 8), "B": torch.randn(5, 8)}
    weights = attention.attention_weights(embeddings)
    # [node, query channel, head, candidate channel] -> a two-way softmax
    assert weights.shape == (5, 2, 2, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5, 2, 2), atol=1e-6)
    # Neither candidate may be structurally impossible: that is exactly the
    # "single key -> softmax == 1" degenerate design the plan forbids.
    assert (weights[..., 0] > 0).all() and (weights[..., 1] > 0).all()
    assert (weights[..., 0] < 1).all() and (weights[..., 1] < 1).all()
    assert torch.isfinite(weights).all()


def test_attention_heads_learn_different_candidate_preferences():
    torch.manual_seed(0)
    attention = CrossGraphAttention(8, ("A", "B"), num_heads=2, dropout=0.0)
    embeddings = {"A": torch.randn(6, 8), "B": torch.randn(6, 8)}
    weights = attention.attention_weights(embeddings)
    assert not torch.allclose(weights[:, :, 0], weights[:, :, 1])


def test_attention_scores_match_the_documented_equations():
    """alpha_A = softmax([Q_A.K_A, Q_A.K_B] / sqrt(d_head)), Z_A = sum alpha * V."""
    torch.manual_seed(2)
    dim, heads = 8, 2
    attention = CrossGraphAttention(dim, ("A", "B"), num_heads=heads, dropout=0.0,
                                    residual=False)
    h_a, h_b = torch.randn(4, dim), torch.randn(4, dim)
    outputs = attention({"A": h_a, "B": h_b})
    head_dim = dim // heads

    def per_head(channel, tensor):
        return attention.query[channel](tensor).view(tensor.shape[0], heads, head_dim)

    def split(channel, tensor):
        return attention.value[channel](tensor).view(tensor.shape[0], heads, head_dim)

    def keys(channel, tensor):
        return attention.key[channel](tensor).view(tensor.shape[0], heads, head_dim)

    # recompute direction A by hand
    q_a = per_head("A", h_a)
    scores = torch.stack([(q_a * keys("A", h_a)).sum(-1),
                          (q_a * keys("B", h_b)).sum(-1)], dim=-1) / math.sqrt(head_dim)
    alpha = torch.softmax(scores, dim=-1)
    v = torch.stack([split("A", h_a), split("B", h_b)], dim=2)      # [N, H, 2, d_head]
    expected = (alpha.unsqueeze(-1) * v).sum(dim=2).reshape(4, dim)
    assert torch.allclose(outputs["A"], expected, atol=1e-5)


def test_attention_mixes_the_other_graph_into_each_query():
    """Cross-graph information must actually flow in both directions."""
    torch.manual_seed(1)
    attention = CrossGraphAttention(8, ("A", "B"), num_heads=2, dropout=0.0)
    h_a, h_b = torch.randn(4, 8), torch.randn(4, 8)
    baseline = attention({"A": h_a, "B": h_b})
    other_moved = attention({"A": h_a, "B": h_b + 3.0})
    assert not torch.allclose(baseline["A"], other_moved["A"])   # B -> A
    own_moved = attention({"A": h_a + 3.0, "B": h_b})
    assert not torch.allclose(baseline["B"], own_moved["B"])     # A -> B


def _zero_value_projections(attention: CrossGraphAttention) -> None:
    with torch.no_grad():
        for channel in attention.channels:
            attention.value[channel].weight.zero_()
            attention.value[channel].bias.zero_()


def test_attention_residual_is_off_by_default_and_opt_in():
    h_a, h_b = torch.randn(3, 6), torch.randn(3, 6)
    plain = CrossGraphAttention(6, ("A", "B"), num_heads=2, dropout=0.0)
    assert plain.residual is False
    _zero_value_projections(plain)
    # no residual: zero values -> zero output
    assert torch.allclose(plain({"A": h_a, "B": h_b})["A"], torch.zeros(3, 6))

    with_residual = CrossGraphAttention(6, ("A", "B"), num_heads=2, dropout=0.0,
                                        residual=True)
    _zero_value_projections(with_residual)
    # with residual: zero values -> the own-graph representation survives
    outputs = with_residual({"A": h_a, "B": h_b})
    assert torch.allclose(outputs["A"], h_a) and torch.allclose(outputs["B"], h_b)


def test_attention_uses_source_specific_projections():
    attention = CrossGraphAttention(8, ("A", "B"), num_heads=1, dropout=0.0)
    assert attention.query["A"] is not attention.query["B"]
    assert attention.key["A"] is not attention.key["B"]
    assert attention.value["A"] is not attention.value["B"]


def test_attention_validates_its_configuration():
    with pytest.raises(ValueError, match="exactly two"):
        CrossGraphAttention(8, ("A",))
    with pytest.raises(ValueError, match="divisible"):
        CrossGraphAttention(6, ("A", "B"), num_heads=4)
    with pytest.raises(ValueError, match="num_heads"):
        CrossGraphAttention(8, ("A", "B"), num_heads=0)
    with pytest.raises(ValueError, match="dropout"):
        CrossGraphAttention(8, ("A", "B"), dropout=1.5)
    attention = CrossGraphAttention(8, ("A", "B"))
    with pytest.raises(ValueError, match="expected embeddings for channels"):
        attention({"A": torch.randn(3, 8)})


def test_attention_heads_are_configurable():
    for heads in (1, 2, 4):
        attention = CrossGraphAttention(8, ("A", "B"), num_heads=heads)
        assert attention.head_dim == 8 // heads
        weights = attention.attention_weights({"A": torch.randn(2, 8),
                                               "B": torch.randn(2, 8)})
        assert weights.shape == (2, 2, heads, 2)
        assert math.isclose(weights.sum().item(), 2 * 2 * heads, rel_tol=1e-5)


# ------------------------------------------------- registry / integration glue
def test_registry_advertises_b4_and_b5_only_as_implemented():
    for name in WEEK3_MODELS:
        assert name in IMPLEMENTED_MODELS
        assert name not in PLANNED_MODELS
    # B6/B7 stay refused until the one-step feedback module exists (M4).
    assert tuple(PLANNED_MODELS) == ("b6", "b7")
    for name in PLANNED_MODELS:
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            build_model(name, feature_dim=8)


@pytest.mark.parametrize("name,fusion_kind", [("b3", "concat"), ("b4", "gate"),
                                              ("b5", "attention")])
def test_fusion_kind_is_reported_per_model(name, fusion_kind):
    model = build_model(name, feature_dim=8, hidden_dim=8, fusion_dim=8, dropout=0.0)
    assert model.fusion_kind == fusion_kind


@pytest.mark.parametrize("name", WEEK3_MODELS)
def test_week3_models_keep_graph_specific_edge_heads(name):
    model = build_model(name, feature_dim=8, hidden_dim=8, fusion_dim=8, dropout=0.0)
    assert model.channel_heads is not None
    assert set(model.channel_heads) == set(model.channels)
    assert model.head_for("A") is not model.head_for("B")
    with pytest.raises(KeyError, match="no channel head"):
        model.head_for("C")
    # B0-B3 keep the single Week 2 head; passing a channel is harmless there.
    for week2 in ("b0", "b1", "b3"):
        other = build_model(week2, feature_dim=8, hidden_dim=8, fusion_dim=8, dropout=0.0)
        assert other.channel_heads is None
        assert other.head_for("A") is other.head


def test_week2_models_score_identically_with_or_without_a_channel_argument():
    model = build_model("b3", feature_dim=8, hidden_dim=8, fusion_dim=8, dropout=0.0)
    embedding = torch.randn(5, 8)
    pairs = torch.tensor([[0, 1], [2, 3]])
    assert torch.allclose(model.score_edges(embedding, pairs),
                          model.score_edges(embedding, pairs, channel="A"))


@pytest.mark.parametrize("name", ["b5"])
def test_week3_models_expose_source_specific_representations(name):
    model = build_model(name, feature_dim=8, hidden_dim=8, fusion_dim=8, dropout=0.0)
    x = torch.randn(6, 8)
    sources = model.fusion_representations(x, _two_graphs())
    assert set(sources) == {"A", "B"}
    for tensor in sources.values():
        assert tensor.shape == (6, 8)
    # The two sources stay distinguishable before the projection.
    assert not torch.allclose(sources["A"], sources["B"])


def test_b4_fuses_to_one_vector_so_it_has_no_per_source_representation():
    model = build_model("b4", feature_dim=8, hidden_dim=8, fusion_dim=8, dropout=0.0)
    x = torch.randn(6, 8)
    with pytest.raises(NotImplementedError, match="per-source"):
        model.fusion_representations(x, _two_graphs())
    outputs = model(x, _two_graphs())
    assert torch.allclose(outputs["A"], outputs["B"])


def test_two_graphs_are_encoded_by_independent_parameters():
    model = build_model("b5", feature_dim=8, hidden_dim=8, fusion_dim=8, dropout=0.0)
    assert model.encoders["A"] is not model.encoders["B"]
    first, second = model.encoders["A"].layers[0].linear, model.encoders["B"].layers[0].linear
    assert first.weight is not second.weight
    graphs = _two_graphs()
    swapped = {"A": graphs["B"], "B": graphs["A"]}
    x = torch.randn(6, 8)
    assert not torch.allclose(model.encode(x, graphs), model.encode(x, swapped))


@pytest.mark.parametrize("name", WEEK3_MODELS)
def test_week3_models_are_built_from_the_shipped_configs(name):
    """Mirrors the trainer wiring, so a config typo fails here and not on the GPU."""
    config = load_config(f"{REPO}/configs/{name}.yaml")
    model_cfg = config["model"]
    assert model_cfg["name"] == name
    model = build_model(model_cfg["name"], feature_dim=32,
                        hidden_dim=int(model_cfg["hidden_dim"]),
                        num_layers=int(model_cfg["num_layers"]),
                        dropout=float(model_cfg["dropout"]),
                        fusion_dim=int(model_cfg["fusion_dim"]),
                        edge_head=model_cfg["edge_head"],
                        attention_heads=int(model_cfg.get("attention_heads", 4)),
                        attention_dropout=model_cfg.get("attention_dropout"),
                        attention_residual=bool(model_cfg.get("attention_residual", False)))
    outputs = model(torch.randn(6, 32), _two_graphs())
    assert outputs["A"].shape == (6, 128)
    assert model.output_dim == 128
    if name == "b5":
        assert model.fusion.num_heads == 4 and model.fusion.head_dim == 32
        assert model.fusion.residual is False
    # Same optimisation budget as B0-B3: the ablations must stay comparable (D-018).
    assert config["optim"]["lr"] == load_config(f"{REPO}/configs/b3.yaml")["optim"]["lr"]
    assert config["optim"]["epochs"] == load_config(f"{REPO}/configs/b3.yaml")["optim"]["epochs"]


def test_b4_is_parameter_matched_with_b3():
    """B3 and B4 both hold one 256->128 projection, so only the *function* differs.

    This keeps the B3 vs B4 comparison free of a capacity confound (unlike B5,
    which adds Q/K/V projections and is reported separately).
    """
    kwargs = dict(feature_dim=16, hidden_dim=128, fusion_dim=128, dropout=0.2)
    b3 = build_model("b3", **kwargs)
    b4 = build_model("b4", **kwargs)
    assert b3.parameter_count() == b4.parameter_count()
    b5 = build_model("b5", **kwargs)
    assert b5.parameter_count() > b4.parameter_count()


@pytest.mark.parametrize("name", WEEK3_MODELS)
def test_week3_models_use_float32_by_default_and_survive_a_step(name):
    model = build_model(name, feature_dim=16, hidden_dim=8, fusion_dim=8, dropout=0.0)
    x = torch.randn(6, 16)                      # default dtype: float32
    assert x.dtype == torch.float32
    outputs = model(x, _two_graphs())
    assert all(tensor.dtype == torch.float32 for tensor in outputs.values())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    before = model.encode(x, _two_graphs()).detach().clone()
    optimizer.zero_grad()
    model.encode(x, _two_graphs()).sum().backward()
    optimizer.step()
    after = model.encode(x, _two_graphs()).detach()
    assert not torch.allclose(before, after), "optimizer must update the parameters"
    assert torch.isfinite(after).all()


@pytest.mark.parametrize("name", WEEK3_MODELS)
def test_week3_models_follow_the_requested_device(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(name, feature_dim=16, hidden_dim=8, fusion_dim=8,
                        dropout=0.0).to(device)
    x = torch.randn(6, 16, device=device)
    adjacency = {key: value.to(device) for key, value in _two_graphs().items()}
    outputs = model(x, adjacency)
    assert all(tensor.device.type == device.type for tensor in outputs.values())
    outputs["A"].sum().backward()
    assert model.encoders["A"].layers[0].linear.weight.grad is not None


@pytest.mark.parametrize("name", WEEK3_MODELS)
def test_week3_checkpoint_roundtrip_restores_embeddings(name, tmp_path):
    torch.manual_seed(2)
    model = build_model(name, feature_dim=16, hidden_dim=8, fusion_dim=8, dropout=0.0)
    x = torch.randn(6, 16)
    graphs = _two_graphs()
    model.eval()
    with torch.no_grad():
        expected = model.encode(x, graphs)
    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "model_name": name}, path)
    payload = torch.load(path, weights_only=False)
    restored = build_model(payload["model_name"], feature_dim=16, hidden_dim=8,
                           fusion_dim=8, dropout=0.0)
    restored.load_state_dict(payload["state_dict"])
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(restored.encode(x, graphs), expected)

