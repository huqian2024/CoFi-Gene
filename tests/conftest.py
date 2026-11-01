"""Shared fixtures: a synthetic raw-data tree and its processed dataset.

Everything is built from :mod:`cofi_gene.synthetic`, so the tests never touch
real STRING/PIE files, the network or a GPU. Building once per session keeps the
suite fast while still exercising the real pipeline code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cofi_gene.dataset import CoFiDataset
from cofi_gene.pipeline import PipelineConfig, build_dataset
from cofi_gene.synthetic import SyntheticSpec, write_synthetic_raw

SPEC = SyntheticSpec(n_genes=60, n_raw_features=64, n_downstream_targets=24,
                     n_contexts=2, n_effects_per_target=12, seed=0)


@pytest.fixture(scope="session")
def synthetic_raw(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("synthetic_raw")
    return write_synthetic_raw(root, SPEC)


@pytest.fixture(scope="session")
def pipeline_config(synthetic_raw, tmp_path_factory) -> PipelineConfig:
    out_dir = tmp_path_factory.mktemp("processed")
    return PipelineConfig(
        dataset_name="synthetic_test",
        string_links=str(synthetic_raw["string_links"]),
        string_info=str(synthetic_raw["string_info"]),
        gene_text=str(synthetic_raw["gene_text"]),
        gene_order=str(synthetic_raw["gene_order"]),
        de_tables={context: str(path) for context, path in synthetic_raw["de_tables"].items()},
        n_components=16,
        split_seed=42,
    )


@pytest.fixture(scope="session")
def processed(synthetic_raw, pipeline_config) -> tuple[CoFiDataset, dict, Path]:
    dataset, report = build_dataset(pipeline_config)
    out_dir = Path(synthetic_raw["root"]).parent / "processed_dataset"
    dataset.save(out_dir)
    return dataset, report.as_dict(), out_dir


@pytest.fixture(scope="session")
def processed_dataset(processed) -> CoFiDataset:
    return processed[0]


@pytest.fixture(scope="session")
def processed_dir(processed) -> Path:
    return processed[2]
