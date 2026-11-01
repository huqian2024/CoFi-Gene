"""Upstream training smoke tests on the synthetic dataset (needs torch)."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cofi_gene.config import load_config  # noqa: E402
from cofi_gene.dataset import CoFiDataset  # noqa: E402
from cofi_gene.train import evaluate_split, main as train_main, set_seed, train_link_prediction  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _config(model: str, out_dir, epochs: int = 4):
    return load_config(REPO / "configs" / f"{model}.yaml", {
        "optim": {"epochs": epochs, "eval_every": 2, "early_stop_patience": 4},
        "runtime": {"device": "cpu", "out_dir": str(out_dir), "seed": 7},
    })


@pytest.mark.parametrize("model", ["b0", "b1", "b2", "b3"])
def test_every_week2_model_trains_end_to_end(model, processed_dataset, tmp_path):
    record = train_link_prediction(processed_dataset, _config(model, tmp_path))
    assert record["best_epoch"] >= 1
    assert np.isfinite(record["best_val_mean_ap"])
    for channel, metrics in record["test"].items():
        assert metrics["n_positives"] > 0 and metrics["n_negatives"] > 0
        assert 0.0 <= metrics["average_precision"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert set(record["test"]) == set(CHANNEL_KEYS[model])
    run_dir = tmp_path / record["run_name"]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "model.pt").exists()
    saved = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["model"]["name"] == model
    assert saved["notes"]


CHANNEL_KEYS = {"b0": ("A", "B"), "b1": ("A",), "b2": ("B",), "b3": ("A", "B")}


def test_training_is_deterministic_for_a_fixed_seed(processed_dataset, tmp_path):
    config = _config("b3", tmp_path, epochs=2)
    first = train_link_prediction(processed_dataset, config)
    second = train_link_prediction(processed_dataset, config)
    assert [h["train_loss"] for h in first["history"]] == \
        [h["train_loss"] for h in second["history"]]
    assert first["test"] == second["test"]


def test_set_seed_makes_torch_and_numpy_reproducible():
    set_seed(3)
    first = (torch.rand(3).tolist(), np.random.rand(2).tolist())
    set_seed(3)
    second = (torch.rand(3).tolist(), np.random.rand(2).tolist())
    assert first == second


def test_model_learns_above_the_prevalence_baseline(processed_dataset, tmp_path):
    """A plumbing test that cannot detect a broken trainer is worthless.

    With 1:1 negatives the AUPRC of a random predictor equals 0.5, and the
    synthetic features/edges share a planted community structure, so a working
    B1 must clearly exceed that baseline.
    """
    record = train_link_prediction(processed_dataset, _config("b1", tmp_path, epochs=30))
    assert record["best_val_mean_ap"] > 0.65, record["best_val_mean_ap"]
    assert record["test"]["A"]["average_precision"] > 0.5


def test_evaluate_split_uses_held_out_edges_only(processed_dataset):
    from cofi_gene.models import build_adjacencies, build_model

    model = build_model("b1", feature_dim=processed_dataset.num_features, dropout=0.0)
    x = torch.as_tensor(processed_dataset.X)
    part = processed_dataset.channel("A")
    adjacencies = build_adjacencies(
        {"A": torch.as_tensor(part.encode_edges)},
        {"A": torch.as_tensor(part.encode_weights)}, processed_dataset.num_nodes)
    val = evaluate_split(model, x, adjacencies, processed_dataset, "val")["A"]
    test = evaluate_split(model, x, adjacencies, processed_dataset, "test")["A"]
    assert val["n_positives"] == part.val.shape[1]
    assert test["n_positives"] == part.test.shape[1]


def test_cli_dry_run_and_missing_dataset(processed_dir, tmp_path):
    assert train_main(["--config", "configs/b3.yaml", "--dataset", str(processed_dir),
                       "--dry-run"]) == 0
    with pytest.raises(FileNotFoundError):
        train_main(["--config", "configs/b3.yaml",
                    "--dataset", str(tmp_path / "does_not_exist"), "--dry-run"])


def test_cuda_request_without_gpu_fails_loudly(processed_dir, tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    config = _config("b1", tmp_path, epochs=1)
    config["runtime"]["device"] = "cuda"
    dataset = CoFiDataset.load(processed_dir)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        train_link_prediction(dataset, config)
