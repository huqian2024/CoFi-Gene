"""Week 4 (M4): B6/B7 one-step edge-weight feedback.

Behaviour-level tests only: the update formula, its bounds/no-op properties, the
detached gradient path, and that the second GCN pass really consumes the updated
weights. Small synthetic graphs keep the suite fast; nothing touches real data.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cofi_gene.config import load_config  # noqa: E402
from cofi_gene.models import (FEEDBACK_MODELS, IMPLEMENTED_MODELS,  # noqa: E402
                              build_model)
from cofi_gene.models.feedback import (DEFAULT_ALPHA, DEFAULT_MARGIN,  # noqa: E402
                                       EdgeWeightFeedback, feedback_weights,
                                       validate_encoder_graph)
from cofi_gene.models.gcn import normalized_adjacency  # noqa: E402
from cofi_gene.models.heads import build_edge_head, decode_pairs  # noqa: E402
from cofi_gene.train import train_link_prediction  # noqa: E402

REPO = __import__("pathlib").Path(__file__).resolve().parents[1]


def _graphs(num_nodes: int = 6):
    """Canonical (sorted, deduplicated, self-loop-free) training edges."""
    edges = {"A": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
             "B": torch.tensor([[0, 3], [5, 1]])}
    weights = {"A": torch.tensor([400.0, 200.0, 900.0, 150.0]),
               "B": torch.tensor([300.0, 120.0])}
    return edges, weights, num_nodes


def _model(name: str, **kwargs):
    return build_model(name, feature_dim=8, hidden_dim=8, fusion_dim=8,
                       dropout=0.0, **kwargs)


# ------------------------------------------------------------------ formula
def test_alpha_zero_is_an_exact_noop():
    weights = torch.tensor([0.0, 1.5, 400.0, 912.0])
    scores = torch.tensor([-9.0, 2.0, 4.5, -0.3])
    updated = feedback_weights(weights, scores, alpha=0.0, margin=0.5)
    assert torch.equal(updated, weights)


def test_update_is_bounded_non_negative_and_finite_for_extreme_scores():
    weights = torch.tensor([0.0, 120.0, 400.0])
    for logits in (torch.full((3,), 500.0), torch.full((3,), -500.0), torch.zeros(3)):
        updated = feedback_weights(weights, logits, alpha=DEFAULT_ALPHA,
                                   margin=DEFAULT_MARGIN)
        assert torch.isfinite(updated).all()
        assert torch.all(updated >= 0.0)
        assert torch.all(updated <= weights * (1.0 + DEFAULT_MARGIN) + 1e-4)
        assert torch.all(updated >= weights * (1.0 - DEFAULT_MARGIN) - 1e-4)


def test_higher_score_means_heavier_edge():
    weights = torch.tensor([400.0, 400.0])
    updated = feedback_weights(weights, torch.tensor([-2.0, 2.0]))
    assert updated[1] > weights[1] > updated[0]


def test_margin_zero_freezes_the_weights():
    weights = torch.tensor([400.0, 15.0])
    scores = torch.tensor([8.0, -8.0])
    assert torch.allclose(feedback_weights(weights, scores, margin=0.0), weights)


def test_mismatched_lengths_and_bad_hyperparameters_are_rejected():
    with pytest.raises(ValueError, match="same edges"):
        feedback_weights(torch.ones(3), torch.ones(2))
    with pytest.raises(ValueError, match="alpha"):
        feedback_weights(torch.ones(2), torch.ones(2), alpha=-0.1)
    with pytest.raises(ValueError, match="margin"):
        feedback_weights(torch.ones(2), torch.ones(2), margin=1.5)
    with pytest.raises(ValueError, match="alpha"):
        EdgeWeightFeedback(alpha=-1.0)


# ------------------------------------------------------- graph / detach guards
def test_encoder_graph_validation_rejects_leaky_or_non_canonical_input():
    with pytest.raises(ValueError, match="self-loops"):
        validate_encoder_graph(torch.tensor([[0, 1], [0, 2]]), 4)
    with pytest.raises(ValueError, match="duplicate"):
        validate_encoder_graph(torch.tensor([[0, 1], [1, 0]]), 4)
    with pytest.raises(ValueError, match="outside"):
        validate_encoder_graph(torch.tensor([[0], [9]]), 4)
    with pytest.raises(ValueError, match=r"\[2, E\]"):
        validate_encoder_graph(torch.tensor([0, 1, 2]), 4)
    # A canonical graph passes through unchanged.
    edges = torch.tensor([[0, 1], [1, 2]])
    assert torch.equal(validate_encoder_graph(edges, 4), edges)


def test_feedback_weights_are_detached_from_the_score_head():
    edges, _, num_nodes = _graphs()
    embeddings = torch.randn(num_nodes, 8, requires_grad=True)
    head = build_edge_head("dot", 8)
    scores = decode_pairs(head, embeddings, edges["A"])
    assert scores.requires_grad and scores.grad_fn is not None

    feedback = EdgeWeightFeedback()
    weights, adjacency = feedback(embeddings, edges["A"], torch.ones(4), head, num_nodes)
    assert weights.grad_fn is None and not weights.requires_grad
    assert not adjacency.values().requires_grad
    assert adjacency.is_sparse


def test_feedback_rebuilds_a_normalised_adjacency_from_the_new_weights():
    edges, weights, num_nodes = _graphs()
    embeddings = torch.randn(num_nodes, 8)
    head = build_edge_head("dot", 8)
    manual = feedback_weights(weights["A"], decode_pairs(head, embeddings, edges["A"]))
    _, adjacency = EdgeWeightFeedback()(embeddings, edges["A"], weights["A"], head,
                                        num_nodes)
    expected = normalized_adjacency(edges["A"], manual, num_nodes)
    # adjacency stores both directions for every edge plus N self-loops
    assert adjacency._nnz() == expected._nnz() == 2 * edges["A"].shape[1] + num_nodes
    assert torch.isfinite(adjacency.values()).all()
    assert torch.allclose(adjacency.to_dense(), expected.to_dense(), atol=1e-6)


# --------------------------------------------------------------- model wiring
def test_registry_exposes_b6_and_b7_and_keeps_b0_to_b5():
    assert set(FEEDBACK_MODELS) == {"b6", "b7"}
    for name in IMPLEMENTED_MODELS:
        assert name in ("b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7")
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_model("b8", feature_dim=8)


def test_b6_b7_reuse_their_base_fusion_and_add_no_parameters():
    b6, b7 = _model("b6"), _model("b7")
    assert (b6.fusion_kind, b6.base_name) == ("attention", "b5")
    assert (b7.fusion_kind, b7.base_name) == ("gate", "b4")
    assert b6.parameter_count() == _model("b5").parameter_count()
    assert b7.parameter_count() == _model("b4").parameter_count()
    assert list(b6.feedback.parameters()) == []
    assert b6.head_for("A") is b6.channel_heads["A"]
    assert b6.output_dim == b7.output_dim == 8


def test_feedback_models_refuse_to_run_without_a_registered_graph():
    x = torch.randn(6, 8)
    for name in FEEDBACK_MODELS:
        with pytest.raises(RuntimeError, match="set_encoding_graphs"):
            _model(name)(x, {})
    with pytest.raises(RuntimeError, match="no feedback step"):
        _model("b5").set_encoding_graphs(_graphs()[0], _graphs()[1], 6)


def test_set_encoding_graphs_stores_the_masked_training_graph_only():
    edges, weights, num_nodes = _graphs()
    model = _model("b6")
    model.set_encoding_graphs(edges, weights, num_nodes)
    stored = model.encoding_graphs()
    for channel in ("A", "B"):
        assert torch.equal(stored[channel], edges[channel])
    with pytest.raises(ValueError, match="weights"):
        model.set_encoding_graphs(edges, {"A": torch.ones(1), "B": torch.ones(2)},
                                 num_nodes)



# ------------------------------------------------------------- forward/backward
def test_feedback_changes_the_second_encoding_and_margin_zero_disables_it():
    edges, weights, num_nodes = _graphs()
    x = torch.randn(num_nodes, 8)

    active = _model("b6")
    active.set_encoding_graphs(edges, weights, num_nodes)
    active_out = active(x, {})

    disabled = _model("b6", feedback_alpha=0.0)
    disabled.load_state_dict(active.state_dict())
    disabled.set_encoding_graphs(edges, weights, num_nodes)
    disabled_out = disabled(x, {})

    frozen = _model("b6", feedback_margin=0.0)
    frozen.load_state_dict(active.state_dict())
    frozen.set_encoding_graphs(edges, weights, num_nodes)
    frozen_out = frozen(x, {})

    assert set(active_out) == {"A", "B"}
    assert active_out["A"].shape == (num_nodes, 8)
    assert not torch.allclose(active_out["A"], disabled_out["A"])
    # margin=0 clamps the update away -> same as alpha=0
    assert torch.allclose(frozen_out["A"], disabled_out["A"])

    gate = _model("b7")
    gate.set_encoding_graphs(edges, weights, num_nodes)
    assert gate(x, {})["A"].shape == (num_nodes, 8)


def test_empty_encoder_graph_is_handled():
    model = _model("b7")
    empty = {"A": torch.zeros((2, 0), dtype=torch.long),
             "B": torch.zeros((2, 0), dtype=torch.long)}
    zeros = {"A": torch.zeros(0), "B": torch.zeros(0)}
    model.set_encoding_graphs(empty, zeros, 5)
    out = model(torch.randn(5, 8), {})
    assert out["A"].shape == (5, 8) and torch.isfinite(out["A"]).all()


def test_backward_reaches_encoders_heads_and_fusion():
    edges, weights, num_nodes = _graphs()
    model = _model("b6")
    model.set_encoding_graphs(edges, weights, num_nodes)
    x = torch.randn(num_nodes, 8, requires_grad=True)
    out = model(x, {})
    loss = out["A"].pow(2).sum() + out["B"].pow(2).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    assert grads and all(g is not None for g in grads.values())
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert any(g.abs().sum() > 0 for g in grads.values())


# ------------------------------------------------------------ trainer contract
def test_trainer_runs_b6_for_two_epochs_and_records_the_feedback_settings(
        processed_dataset, tmp_path):
    config = load_config(REPO / "configs" / "b6.yaml", {
        "optim": {"epochs": 2, "eval_every": 1, "early_stop_patience": 2},
        "runtime": {"device": "cpu", "out_dir": str(tmp_path), "seed": 5},
    })
    record = train_link_prediction(processed_dataset, config, evaluate_test=False)
    assert record["model"]["name"] == "b6"
    assert record["model"]["uses_feedback"] is True
    assert record["model"]["feedback"] == {"alpha": 0.1, "margin": 0.5,
                                           "base_model": "b5"}
    assert record["test"] is None
    assert record["best_epoch"] >= 1
    assert all(np.isfinite(row["train_loss"]) for row in record["history"])

    # The trainer must hand over each channel's *encoder* graph, and that graph
    # must be disjoint from the held-out positives (defence in depth / leakage).
    dataset = processed_dataset
    for channel in ("A", "B"):
        part = dataset.channel(channel)
        encoded = {tuple(sorted(pair)) for pair in np.asarray(part.encode_edges).T.tolist()}
        held_out = {tuple(sorted(pair)) for pair in
                    np.concatenate([part.val, part.test], axis=1).T.tolist()}
        assert encoded.isdisjoint(held_out)


# ------------------------------------------------- B5/B4 equivalence at alpha=0
def test_alpha_zero_reproduces_b5_and_b4_in_eval_mode():
    """With alpha=0 the feedback weights are unchanged, so the second pass sees the
    same adjacency as the first and B6/B7 must equal B5/B4 exactly."""
    from cofi_gene.models import build_adjacencies

    edges, weights, num_nodes = _graphs()
    x = torch.randn(num_nodes, 8)
    adjacencies = build_adjacencies(edges, weights, num_nodes)

    for base, feedback_name in (("b5", "b6"), ("b4", "b7")):
        torch.manual_seed(0)
        reference = _model(base)
        candidate = _model(feedback_name, feedback_alpha=0.0)
        candidate.load_state_dict(reference.state_dict())   # same parameters
        candidate.set_encoding_graphs(edges, weights, num_nodes)
        reference.eval()
        candidate.eval()
        with torch.no_grad():
            expected = reference(x, adjacencies)
            actual = candidate(x, adjacencies)
        for channel in ("A", "B"):
            assert torch.equal(expected[channel], actual[channel]), (base, channel)


def test_alpha_zero_matches_in_train_mode_when_dropout_is_zero():
    from cofi_gene.models import build_adjacencies

    edges, weights, num_nodes = _graphs()
    x = torch.randn(num_nodes, 8)
    adjacencies = build_adjacencies(edges, weights, num_nodes)
    torch.manual_seed(3)
    reference = _model("b5")
    candidate = _model("b6", feedback_alpha=0.0)
    candidate.load_state_dict(reference.state_dict())
    candidate.set_encoding_graphs(edges, weights, num_nodes)
    with torch.no_grad():
        assert torch.equal(reference(x, adjacencies)["A"], candidate(x, {})["A"])


def test_the_second_pass_reuses_the_same_encoder_module():
    edges, weights, num_nodes = _graphs()
    model = _model("b6")
    model.set_encoding_graphs(edges, weights, num_nodes)
    calls = {"count": 0}
    handle = model.encoders["A"].register_forward_hook(
        lambda *args: calls.__setitem__("count", calls["count"] + 1))
    try:
        model(torch.randn(num_nodes, 8), {})
    finally:
        handle.remove()
    assert calls["count"] == 2   # first pass + feedback pass, shared weights
    assert model.parameter_count() == _model("b5").parameter_count()


def test_b6_b7_keep_the_week3_fusion_mathematics():
    from cofi_gene.models.attention import CrossGraphAttention
    from cofi_gene.models.fusion import GatedFusion

    b6, b7 = _model("b6"), _model("b7")
    assert isinstance(b6.fusion, CrossGraphAttention)
    assert isinstance(b7.fusion, GatedFusion)
    assert b6.fusion.num_heads == 4 and b6.fusion.residual is False
    # The gate is still the single Linear(2d -> d) + Sigmoid of D-019.
    assert isinstance(b7.fusion.gate, torch.nn.Linear)
    assert b7.fusion.gate.in_features == 2 * b7.fusion.dim
    assert b7.fusion.gate.out_features == b7.fusion.dim


# ------------------------------------------------------- extreme conditions
def test_weights_are_cast_to_the_score_dtype_and_device():
    scores = torch.zeros(3, dtype=torch.float32)
    weights = np.asarray([1, 400, 900], dtype=np.int64)   # different dtype/source
    updated = feedback_weights(weights, scores)
    assert updated.dtype == torch.float32 and updated.device == scores.device
    assert torch.allclose(updated, torch.tensor([1.0, 400.0, 900.0]))


def test_saturating_logits_and_large_weights_stay_finite():
    weights = torch.tensor([1000.0, 0.5, 0.0])
    updated = feedback_weights(weights, torch.tensor([1e4, -1e4, 1e4]))
    assert torch.isfinite(updated).all()
    assert torch.all(updated >= 0.0)
    assert torch.all(updated <= 1500.0 + 1e-3)


def test_isolated_nodes_remain_finite():
    model = _model("b7")
    single = {"A": torch.tensor([[0], [1]]), "B": torch.tensor([[0], [1]])}
    weights = {"A": torch.tensor([10.0]), "B": torch.tensor([10.0])}
    model.set_encoding_graphs(single, weights, 6)   # nodes 2..5 stay isolated
    out = model(torch.randn(6, 8), {})
    assert out["A"].shape == (6, 8)
    assert torch.isfinite(out["A"]).all()


def test_checkpoint_roundtrip_rebuilds_and_forwards(processed_dataset, tmp_path):
    config = load_config(REPO / "configs" / "b7.yaml", {
        "optim": {"epochs": 1, "eval_every": 1, "early_stop_patience": 1},
        "runtime": {"device": "cpu", "out_dir": str(tmp_path), "seed": 2},
    })
    record = train_link_prediction(processed_dataset, config, evaluate_test=False)
    checkpoint = torch.load(tmp_path / record["run_name"] / "model.pt",
                            map_location="cpu", weights_only=False)
    saved = checkpoint["config"]["model"]
    restored = build_model(checkpoint["model_name"],
                           feature_dim=processed_dataset.num_features,
                           hidden_dim=int(saved["hidden_dim"]),
                           fusion_dim=int(saved["fusion_dim"]),
                           dropout=float(saved["dropout"]),
                           feedback_alpha=float(saved["feedback_alpha"]),
                           feedback_margin=float(saved["feedback_margin"]))
    restored.load_state_dict(checkpoint["state_dict"])
    edges = {channel: torch.as_tensor(processed_dataset.channel(channel).encode_edges)
             for channel in ("A", "B")}
    weights = {channel: torch.as_tensor(
        processed_dataset.channel(channel).encode_weights, dtype=torch.float32)
        for channel in ("A", "B")}
    restored.set_encoding_graphs(edges, weights, processed_dataset.num_nodes)
    restored.eval()
    with torch.no_grad():
        out = restored(torch.as_tensor(processed_dataset.X), {})
    assert out["A"].shape == (processed_dataset.num_nodes, int(saved["fusion_dim"]))
    assert torch.isfinite(out["A"]).all()
    assert checkpoint["seed"] == 2

