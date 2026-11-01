"""Gene prior features: loading and the shared 3072 -> 128 compression.

B0-B7 must consume the *same* node features, so the reducer is fitted once, on
public gene priors only, and persisted. Fitting it on downstream response labels
(or refitting per model) would be leakage and would also break the comparison,
which is why this module never sees PIE DE tables.

The pending choice is *which* reducer and how many components (D-012): the
implementation below is a deterministic, sign-fixed PCA via SVD, with an
optional z-scoring step that is off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "FeatureReducer",
    "fit_feature_reducer",
    "load_feature_matrix",
    "load_gene_order",
]

_KEYS = ("X", "gene_text", "features", "arr_0")


def load_feature_matrix(path: str | Path) -> np.ndarray:
    """Load a gene-prior feature matrix from ``.npy`` / ``.npz``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".npy":
        matrix = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = next((k for k in _KEYS if k in archive), None)
            if key is None:
                raise ValueError(f"no feature array among {_KEYS} in {path}")
            matrix = archive[key]
    else:
        raise ValueError(f"unsupported feature file type: {path.suffix}")
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a [N, F] matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature matrix contains NaN/inf")
    return matrix


def load_gene_order(path: str | Path) -> tuple[str, ...]:
    """Read the row order that must accompany a feature matrix (one id per line)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(genes)) != len(genes):
        raise ValueError("gene order file contains duplicates")
    return tuple(genes)


@dataclass
class FeatureReducer:
    """Affine projection fitted once on public priors and reused by all models."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray          # [n_components, F]
    explained_variance_ratio: np.ndarray

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def transform(self, X) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != self.components.shape[1]:
            raise ValueError(f"expected [N, {self.components.shape[1]}], got {array.shape}")
        centered = (array - self.mean) / self.scale
        return (centered @ self.components.T).astype(np.float32)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(Path(path), mean=self.mean, scale=self.scale,
                            components=self.components,
                            explained_variance_ratio=self.explained_variance_ratio)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureReducer":
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(mean=archive["mean"], scale=archive["scale"],
                       components=archive["components"],
                       explained_variance_ratio=archive["explained_variance_ratio"])


def fit_feature_reducer(X, n_components: int = 128, standardize: bool = False) -> FeatureReducer:
    """Deterministic PCA (SVD) reducer.

    Sign convention: each component is flipped so that its largest-magnitude
    loading is positive, which makes the saved matrix reproducible across BLAS
    builds and therefore keeps checkpoints comparable.
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected a [N, F] matrix, got shape {matrix.shape}")
    n_samples, n_features = matrix.shape
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if n_components > min(n_samples, n_features):
        raise ValueError(
            f"n_components={n_components} exceeds min(N, F)=min({n_samples}, {n_features})")
    mean = matrix.mean(axis=0)
    if standardize:
        scale = matrix.std(axis=0)
        scale[scale == 0] = 1.0
    else:
        scale = np.ones(n_features, dtype=np.float64)
    centered = (matrix - mean) / scale
    # economy SVD: centered = U S Vt, principal axes are rows of Vt
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].copy()
    signs = np.sign(components[np.arange(n_components), np.abs(components).argmax(axis=1)])
    signs[signs == 0] = 1.0
    components *= signs[:, None]
    total_variance = float(np.sum(singular_values ** 2))
    ratio = (singular_values[:n_components] ** 2 / total_variance) if total_variance else \
        np.zeros(n_components)
    return FeatureReducer(mean=mean, scale=scale, components=components,
                          explained_variance_ratio=ratio.astype(np.float64))
