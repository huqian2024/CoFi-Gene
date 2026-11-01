"""Dataset persistence: save/load must round-trip exactly and reject bad files."""

import json

import numpy as np
import pytest

from cofi_gene.dataset import DATASET_VERSION, CoFiDataset


def test_roundtrip_preserves_every_array(processed_dataset, tmp_path):
    out_dir = tmp_path / "dataset"
    processed_dataset.save(out_dir)
    loaded = CoFiDataset.load(out_dir)
    assert loaded.gene_ids == processed_dataset.gene_ids
    assert np.array_equal(loaded.X, processed_dataset.X)
    assert loaded.num_nodes == processed_dataset.num_nodes
    assert loaded.num_features == processed_dataset.num_features
    for channel in processed_dataset.channel_names:
        original, restored = processed_dataset.channel(channel), loaded.channel(channel)
        for name in ("train", "val", "test", "neg_train", "neg_val", "neg_test",
                     "encode_edges", "train_weights", "encode_weights"):
            assert np.array_equal(getattr(original, name), getattr(restored, name)), name
        assert restored.stats == original.stats
    assert loaded.target_splits == processed_dataset.target_splits
    loaded.validate()


def test_version_mismatch_is_rejected(processed_dataset, tmp_path):
    out_dir = tmp_path / "dataset"
    processed_dataset.save(out_dir)
    payload = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
    payload["dataset_version"] = DATASET_VERSION + 1
    (out_dir / "dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported dataset version"):
        CoFiDataset.load(out_dir)


def test_missing_channel_array_is_reported(processed_dataset, tmp_path):
    out_dir = tmp_path / "dataset"
    processed_dataset.save(out_dir)
    payload = json.loads((out_dir / "dataset.json").read_text(encoding="utf-8"))
    payload["channels"]["A"]["num_train"] = 1
    with np.load(out_dir / "dataset.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays.pop("A_train")
    np.savez_compressed(out_dir / "dataset.npz", **arrays)
    (out_dir / "dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing arrays"):
        CoFiDataset.load(out_dir)


def test_summary_is_json_serialisable(processed_dataset):
    import json as _json

    assert _json.dumps(processed_dataset.summary())
