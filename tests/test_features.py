"""Shared feature reducer: determinism, sign convention, persistence, errors."""

import numpy as np
import pytest

from cofi_gene.features import (FeatureReducer, fit_feature_reducer, load_feature_matrix,
                               load_gene_order)


@pytest.fixture()
def random_matrix():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(40, 3))
    return (latent @ rng.normal(size=(3, 30)) + rng.normal(scale=0.1, size=(40, 30))).astype(np.float32)


def test_reducer_is_deterministic_and_sign_fixed(random_matrix):
    first = fit_feature_reducer(random_matrix, n_components=8)
    second = fit_feature_reducer(random_matrix, n_components=8)
    assert np.allclose(first.components, second.components)
    # largest-magnitude loading of every component is positive by convention
    assert np.all(first.components[np.arange(8), np.abs(first.components).argmax(axis=1)] > 0)
    projected = first.transform(random_matrix)
    assert projected.shape == (40, 8)
    assert projected.dtype == np.float32


def test_reducer_preserves_low_rank_structure():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(40, 3))
    exactly_rank_three = (latent @ rng.normal(size=(3, 30))).astype(np.float32)
    reducer = fit_feature_reducer(exactly_rank_three, n_components=3)
    assert reducer.explained_variance_ratio.sum() > 0.999
    projected = reducer.transform(exactly_rank_three)
    reconstructed = projected @ reducer.components
    centred = exactly_rank_three - reducer.mean
    assert np.allclose(reconstructed, centred, atol=1e-3)


def test_reducer_roundtrips_through_disk(random_matrix, tmp_path):
    reducer = fit_feature_reducer(random_matrix, n_components=5, standardize=True)
    path = tmp_path / "reducer.npz"
    reducer.save(path)
    loaded = FeatureReducer.load(path)
    assert np.allclose(loaded.transform(random_matrix), reducer.transform(random_matrix))


def test_invalid_reducer_arguments(random_matrix):
    with pytest.raises(ValueError, match="exceeds"):
        fit_feature_reducer(random_matrix, n_components=1000)
    with pytest.raises(ValueError, match="positive"):
        fit_feature_reducer(random_matrix, n_components=0)
    reducer = fit_feature_reducer(random_matrix, n_components=4)
    with pytest.raises(ValueError, match="expected"):
        reducer.transform(np.zeros((5, 7)))


def test_feature_matrix_and_gene_order_loading(random_matrix, tmp_path):
    npy = tmp_path / "gene_text.npy"
    np.save(npy, random_matrix)
    assert load_feature_matrix(npy).shape == random_matrix.shape
    npz = tmp_path / "gene_text.npz"
    np.savez(npz, gene_text=random_matrix)
    assert load_feature_matrix(npz).shape == random_matrix.shape
    with pytest.raises(ValueError, match="no feature array"):
        bad = tmp_path / "bad.npz"
        np.savez(bad, other=random_matrix)
        load_feature_matrix(bad)
    unsupported = tmp_path / "genes.tsv"
    unsupported.write_text("gene\tvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_feature_matrix(unsupported)

    order = tmp_path / "genes.txt"
    order.write_text("A\nB\nC\n", encoding="utf-8")
    assert load_gene_order(order) == ("A", "B", "C")
    order.write_text("A\nA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_gene_order(order)


def test_non_finite_features_are_rejected(tmp_path):
    bad = np.zeros((3, 2), dtype=np.float32)
    bad[1, 1] = np.nan
    path = tmp_path / "bad.npy"
    np.save(path, bad)
    with pytest.raises(ValueError, match="NaN"):
        load_feature_matrix(path)
