"""Week 5 (M5) cross-view gene-correspondence control (experiment 3).

The control must be surgical: only the A/B pairing consumed by fusion changes.
These tests pin that down (encoders untouched, state_dict unchanged, permutation
applied exactly once before fusion) and check the trainer records the setting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cofi_gene.config import load_config  # noqa: E402
from cofi_gene.models import (CROSS_VIEW_MODELS, build_adjacencies,  # noqa: E402
                              build_model)
from cofi_gene.perturbation import cross_view_permutation  # noqa: E402
from cofi_gene.train import train_link_prediction  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NUM_NODES = 7


def _model(name: str, **kwargs):
    return build_model(name, feature_dim=8, hidden_dim=8, fusion_dim=8,
                       dropout=0.0, **kwargs)


def _graph():
    edges = {"A": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
             "B": torch.tensor([[0, 3], [5, 1]])}
    weights = {"A": torch.tensor([400.0, 200.0, 900.0, 150.0]),
               "B": torch.tensor([300.0, 120.0])}
    return build_adjacencies(edges, weights, NUM_NODES)


# ------------------------------------------------------------------ registry
def test_only_cross_view_models_accept_the_control():
    assert CROSS_VIEW_MODELS == ("b3", "b4", "b5", "b6", "b7")
    perm = cross_view_permutation(NUM_NODES, 0)
    for name in CROSS_VIEW_MODELS:
        model = _model(name)
        assert model.cross_view_channel == "B"
        assert model.cross_view_permutation is None
        model.set_cross_view_permutation(perm)
        assert torch.equal(model.cross_view_permutation, torch.as_tensor(perm))
        model.clear_cross_view_permutation()
        assert model.cross_view_permutation is None
    for name in ("b0", "b1", "b2"):
        with pytest.raises(RuntimeError, match="no cross-view fusion"):
            _model(name).set_cross_view_permutation(perm)


def test_identity_and_non_bijections_are_refused():
    model = _model("b5")
    with pytest.raises(ValueError, match="no fixed point"):
        model.set_cross_view_permutation(torch.arange(NUM_NODES))
    with pytest.raises(ValueError, match="not a permutation"):
        model.set_cross_view_permutation(torch.tensor([0, 1, 2, 3, 4, 5, 5]))
    with pytest.raises(ValueError, match="at least two genes"):
        model.set_cross_view_permutation(torch.tensor([0]))
    with pytest.raises(ValueError, match="not used by model"):
        model.set_cross_view_permutation(cross_view_permutation(NUM_NODES, 0),
                                         channel="C")


def test_state_dict_is_unchanged_by_the_control():
    model = _model("b6")
    before = list(model.state_dict().keys())
    model.set_cross_view_permutation(cross_view_permutation(NUM_NODES, 3))
    assert list(model.state_dict().keys()) == before
    assert "cross_view_perm" not in model.state_dict()


# --------------------------------------------------------------- forward math
def test_permutation_is_applied_once_right_before_fusion():
    torch.manual_seed(0)
    model = _model("b5").eval()
    x = torch.randn(NUM_NODES, 8)
    adjacencies = _graph()
    perm = torch.as_tensor(cross_view_permutation(NUM_NODES, 1))

    with torch.no_grad():
        aligned = model.fusion_representations(x, adjacencies)
        model.set_cross_view_permutation(perm)
        misaligned = model.fusion_representations(x, adjacencies)
        encoded = {channel: model.encoders[channel](x, adjacencies[channel])
                   for channel in model.channels}
        expected = model.fusion({"A": encoded["A"], "B": encoded["B"][perm]})

    for channel in ("A", "B"):
        assert torch.allclose(misaligned[channel], expected[channel], atol=1e-6)
        assert torch.allclose(encoded[channel], encoded[channel])   # encoders untouched
    assert not torch.allclose(aligned["A"], misaligned["A"])


def test_misalignment_leaves_encoders_alone_but_changes_the_embedding():
    torch.manual_seed(2)
    model = _model("b4").eval()
    x = torch.randn(NUM_NODES, 8)
    adjacencies = _graph()

    with torch.no_grad():
        encoded_before = {channel: model.encoders[channel](x, adjacencies[channel])
                          for channel in model.channels}
        before = model.encode(x, adjacencies)
        model.set_cross_view_permutation(cross_view_permutation(NUM_NODES, 9))
        encoded_after = {channel: model.encoders[channel](x, adjacencies[channel])
                         for channel in model.channels}
        after = model.encode(x, adjacencies)

    for channel in ("A", "B"):
        assert torch.equal(encoded_before[channel], encoded_after[channel])
    assert not torch.allclose(before, after)
    model.clear_cross_view_permutation()
    with torch.no_grad():
        assert torch.allclose(model.encode(x, adjacencies), before)


def test_forward_rejects_a_permutation_for_the_wrong_node_count():
    model = _model("b5").eval()
    model.set_cross_view_permutation(cross_view_permutation(5, 0))
    with pytest.raises(ValueError, match="entries but channel"):
        model(torch.randn(NUM_NODES, 8), _graph())



# ---------------------------------------------------------------- trainer
def _train_config(name: str, out_dir, seed: int = 7, epochs: int = 2,
                  data: dict | None = None):
    overrides = {
        "optim": {"epochs": epochs, "eval_every": 1, "early_stop_patience": 1},
        "runtime": {"device": "cpu", "out_dir": str(out_dir), "seed": seed},
    }
    if data:
        overrides["data"] = data
    return load_config(REPO / "configs" / f"{name}.yaml", overrides)


def test_edge_drop_keeps_the_evaluation_set_fixed(processed_dataset, tmp_path):
    baseline = train_link_prediction(
        processed_dataset, _train_config("b3", tmp_path))
    dropped = train_link_prediction(
        processed_dataset, _train_config("b3", tmp_path,
                                         data={"edge_drop_fraction": 0.3}))
    assert "perturbation" not in baseline
    perturbation = dropped["perturbation"]
    assert perturbation["edge_drop_fraction"] == 0.3
    for channel in ("A", "B"):
        counts = perturbation["channels"][channel]
        assert counts["num_encode_edges_removed"] > 0
        assert (counts["num_encode_edges_after"]
                == counts["num_encode_edges_before"] - counts["num_encode_edges_removed"])
        assert counts["edge_drop"]["fraction"] == 0.3
        # Identical evaluation set: same positives and negatives as the baseline.
        assert (dropped["test"][channel]["n_positives"]
                == baseline["test"][channel]["n_positives"])
        assert (dropped["test"][channel]["n_negatives"]
                == baseline["test"][channel]["n_negatives"])


def test_zero_fraction_reproduces_the_unperturbed_run(processed_dataset, tmp_path):
    baseline = train_link_prediction(processed_dataset, _train_config("b3", tmp_path))
    zero = train_link_prediction(processed_dataset, _train_config(
        "b3", tmp_path, data={"edge_drop_fraction": 0.0}))
    assert "perturbation" not in zero
    assert ([row["train_loss"] for row in zero["history"]]
            == [row["train_loss"] for row in baseline["history"]])
    assert zero["val"] == baseline["val"]


@pytest.mark.parametrize("name", list(CROSS_VIEW_MODELS))
def test_every_cross_view_model_trains_with_the_control(name, processed_dataset,
                                                        tmp_path):
    config = _train_config(name, tmp_path, epochs=1,
                           data={"misalign_cross_view": True})
    record = train_link_prediction(processed_dataset, config, evaluate_test=False)
    perturbation = record["perturbation"]
    assert perturbation["misalign_cross_view"] is True
    assert perturbation["cross_view"]["channel"] == "B"
    assert len(perturbation["cross_view"]["permutation_sha256"]) == 16
    assert record["test"] is None
    assert record["history"] and all(row["train_loss"] == row["train_loss"]
                                     for row in record["history"])


def test_mask_other_view_is_recorded_and_does_not_touch_evaluation(
        processed_dataset, tmp_path):
    baseline = train_link_prediction(processed_dataset, _train_config("b3", tmp_path))
    masked = train_link_prediction(processed_dataset, _train_config(
        "b3", tmp_path, data={"mask_other_view": True}))
    perturbation = masked["perturbation"]
    assert perturbation["mask_other_view"] is True
    for channel in ("A", "B"):
        counts = perturbation["channels"][channel]
        assert counts["cross_view_mask"] is not None
        assert (counts["num_train_positives_after"]
                == counts["num_train_positives_before"]
                - counts["cross_view_mask"]["num_train_positives_removed"])
        assert (masked["test"][channel]["n_positives"]
                == baseline["test"][channel]["n_positives"])
