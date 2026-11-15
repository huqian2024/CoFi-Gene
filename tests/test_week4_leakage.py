"""Week 4 (M4): leakage isolation for the B6/B7 feedback step.

These tests follow the real data flow and pin each boundary:

    dataset/processed -> part.encode_edges -> set_encoding_graphs -> feedback
    -> second GCN -> loss -> validation

The strongest one swaps the *hidden* val/test positives while leaving the encoder
graph untouched: training must be bit-identical, because the feedback step is not
allowed to see anything that depends on the held-out labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cofi_gene.config import load_config  # noqa: E402
from cofi_gene.dataset import ChannelSplit, CoFiDataset  # noqa: E402
from cofi_gene.models import build_adjacencies, build_model  # noqa: E402
from cofi_gene.train import evaluate_split, train_link_prediction  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FEEDBACK_MODELS = ("b6", "b7")


def _config(name: str, out_dir, epochs: int = 2, seed: int = 7):
    return load_config(REPO / "configs" / f"{name}.yaml", {
        "optim": {"epochs": epochs, "eval_every": 1, "early_stop_patience": 2},
        "runtime": {"device": "cpu", "out_dir": str(out_dir), "seed": seed},
    })


def _canonical(edge_array) -> set[tuple[int, int]]:
    array = np.asarray(edge_array)
    if array.size == 0:
        return set()
    return {tuple(sorted((int(u), int(v)))) for u, v in array.T.tolist()}


def _swap_held_out(dataset: CoFiDataset) -> CoFiDataset:
    """Same encoder graphs and negatives, val/test positives exchanged."""
    channels = {}
    for name, part in dataset.channels.items():
        channels[name] = ChannelSplit(
            train=part.train,
            val=part.test.copy(), test=part.val.copy(),
            train_weights=part.train_weights,
            val_weights=part.test_weights.copy(),
            test_weights=part.val_weights.copy(),
            neg_train=part.neg_train, neg_val=part.neg_val, neg_test=part.neg_test,
            encode_edges=part.encode_edges, encode_weights=part.encode_weights,
            stats=dict(part.stats))
    return CoFiDataset(gene_ids=dataset.gene_ids, X=dataset.X, channels=channels,
                       target_splits=dataset.target_splits, meta=dict(dataset.meta))


def _registered_model(dataset: CoFiDataset, name: str = "b6", hidden: int = 16):
    edges = {channel: torch.as_tensor(dataset.channel(channel).encode_edges)
             for channel in ("A", "B")}
    weights = {channel: torch.as_tensor(dataset.channel(channel).encode_weights,
                                       dtype=torch.float32)
               for channel in ("A", "B")}
    model = build_model(name, feature_dim=dataset.num_features, hidden_dim=hidden,
                        fusion_dim=hidden, dropout=0.0)
    model.set_encoding_graphs(edges, weights, dataset.num_nodes)
    return model, edges, weights


# ------------------------------------------------------- encoder-graph boundary
def test_feedback_adjacency_is_exactly_the_encoder_graph(processed_dataset):
    dataset = processed_dataset
    model, _, _ = _registered_model(dataset, "b6")
    x = torch.as_tensor(dataset.X)
    with torch.no_grad():
        first = model._encode(x, model._base_adjacency)
        for channel in ("A", "B"):
            _, adjacency = model.feedback(
                first[channel], model.encoding_graphs()[channel],
                model._encoder_weights[channel], model.head_for(channel),
                dataset.num_nodes)
            indices = adjacency.indices()
            pairs = {(min(int(u), int(v)), max(int(u), int(v)))
                     for u, v in indices.T.tolist() if int(u) != int(v)}
            assert pairs == _canonical(dataset.channel(channel).encode_edges)
            held_out = _canonical(dataset.channel(channel).val) | \
                _canonical(dataset.channel(channel).test)
            assert pairs.isdisjoint(held_out)


def test_negatives_and_hidden_edges_never_enter_the_encoder_graph(processed_dataset):
    for channel in ("A", "B"):
        part = processed_dataset.channel(channel)
        encoded = _canonical(part.encode_edges)
        assert encoded.isdisjoint(_canonical(part.val) | _canonical(part.test))
        assert encoded.isdisjoint(_canonical(part.neg_train) |
                                  _canonical(part.neg_val) |
                                  _canonical(part.neg_test))


# --------------------------------------------- hidden-edge invariance (core test)
@pytest.mark.parametrize("model_name", FEEDBACK_MODELS)
def test_swapping_hidden_positives_leaves_training_bit_identical(
        model_name, processed_dataset, tmp_path):
    swapped = _swap_held_out(processed_dataset)
    swapped.validate()  # still a legal dataset: exchange keeps the partitions

    original = train_link_prediction(processed_dataset,
                                     _config(model_name, tmp_path / "original"))
    alternative = train_link_prediction(swapped,
                                        _config(model_name, tmp_path / "swapped"))

    # Same encoder graph + same train positives + same negatives + same seed.
    assert [row["train_loss"] for row in original["history"]] == \
        [row["train_loss"] for row in alternative["history"]]
    first = torch.load(tmp_path / "original" / original["run_name"] / "model.pt",
                       map_location="cpu", weights_only=False)["state_dict"]
    second = torch.load(tmp_path / "swapped" / alternative["run_name"] / "model.pt",
                        map_location="cpu", weights_only=False)["state_dict"]
    assert set(first) == set(second)
    assert all(torch.equal(first[key], second[key]) for key in first)
    # ... and the held-out split really did change (otherwise the test is vacuous)
    assert original["val"] != alternative["val"]


# ------------------------------------------------- validation-side invariance
def test_validation_does_not_touch_parameters_or_the_encoder_graph(processed_dataset):
    dataset = processed_dataset
    model, _, _ = _registered_model(dataset, "b7")
    model.train()
    x = torch.as_tensor(dataset.X)
    adjacencies = build_adjacencies(
        {channel: model.encoding_graphs()[channel] for channel in ("A", "B")},
        model._encoder_weights, dataset.num_nodes)

    params = {name: value.detach().clone() for name, value in model.named_parameters()}
    adjacency_before = {channel: adjacency.values().clone()
                        for channel, adjacency in model._base_adjacency.items()}
    graph_before = {channel: edges.clone() for channel, edges in model.encoding_graphs().items()}

    metrics = evaluate_split(model, x, adjacencies, dataset, "val")
    assert set(metrics) == {"A", "B"}
    assert model.training is True                     # evaluate_split restores train mode
    for name, value in model.named_parameters():
        assert torch.equal(value, params[name]), f"{name} changed during validation"
    for channel, values in model._base_adjacency.items():
        assert torch.equal(values.values(), adjacency_before[channel])
    for channel, edges in model.encoding_graphs().items():
        assert torch.equal(edges, graph_before[channel])


def test_feedback_reads_the_registered_graph_not_the_training_positives(processed_dataset):
    """Perturbing an *encoder* edge weight must change the output; perturbing only
    the supervision positives must not (they live in a different tensor)."""
    dataset = processed_dataset
    model, edges, weights = _registered_model(dataset, "b6")
    x = torch.as_tensor(dataset.X)
    model.eval()
    with torch.no_grad():
        baseline = model(x, {})["A"]

        bumped = dict(weights)
        bumped["A"] = weights["A"] * 3.0
        model.set_encoding_graphs(edges, bumped, dataset.num_nodes)
        changed = model(x, {})["A"]
    assert not torch.allclose(baseline, changed)

    # No dataset (and therefore no supervision split) is reachable from the model.
    assert not any(value is dataset for value in vars(model).values())
    assert all(isinstance(tensor, torch.Tensor)
               for tensor in model.encoding_graphs().values())


def _feedback_pairs(dataset, model, channel: str) -> set[tuple[int, int]]:
    """Edges the feedback step actually feeds into the second GCN pass."""
    x = torch.as_tensor(dataset.X)
    with torch.no_grad():
        first = model._encode(x, model._base_adjacency)
        _, adjacency = model.feedback(first[channel], model.encoding_graphs()[channel],
                                     model._encoder_weights[channel],
                                     model.head_for(channel), dataset.num_nodes)
    indices = adjacency.indices()
    return {(min(int(u), int(v)), max(int(u), int(v)))
            for u, v in indices.T.tolist() if int(u) != int(v)}


def test_masked_protocol_is_respected_when_both_views_hide_a_pair(synthetic_raw):
    """``mask_other_view=True``: the encoder graph of one view must not contain the
    other view's held-out positives either -- and the feedback step inherits it."""
    from cofi_gene.pipeline import PipelineConfig, build_dataset

    config = PipelineConfig(
        dataset_name="masked_synthetic",
        string_links=str(synthetic_raw["string_links"]),
        string_info=str(synthetic_raw["string_info"]),
        gene_text=str(synthetic_raw["gene_text"]),
        gene_order=str(synthetic_raw["gene_order"]),
        de_tables={context: str(path)
                   for context, path in synthetic_raw["de_tables"].items()},
        n_components=16, split_seed=42, mask_other_view=True)
    dataset, report = build_dataset(config)
    dataset.validate()
    assert dataset.channels["A"].stats["num_encode_edges"] > 0

    model, _, _ = _registered_model(dataset, "b6")
    for channel, other in (("A", "B"), ("B", "A")):
        part = dataset.channel(channel)
        encoded = _canonical(part.encode_edges)
        assert encoded <= _canonical(part.train)                 # visible training edges only
        assert encoded.isdisjoint(_canonical(part.val) | _canonical(part.test))
        other_held = _canonical(dataset.channel(other).val) | \
            _canonical(dataset.channel(other).test)
        assert encoded.isdisjoint(other_held)                    # both-view masking
        assert _feedback_pairs(dataset, model, channel) == encoded

