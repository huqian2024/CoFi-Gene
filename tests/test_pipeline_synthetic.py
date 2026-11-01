"""End-to-end data build on synthetic raw files.

This is the Week-2 "does the pipeline actually work" test: raw STRING-like rows go
in, a validated dataset comes out, and the leakage invariants are asserted on the
real build rather than on hand-made objects.
"""

import json

import numpy as np

from cofi_gene.graph_build import undirected_pair_keys
from cofi_gene.pipeline import PipelineConfig, build_dataset, split_decisions


def test_build_reports_gene_selection_and_channel_statistics(processed):
    dataset, report, _ = processed
    assert dataset.num_nodes == len(dataset.gene_ids) == 60
    assert dataset.num_features == 16
    assert set(dataset.channels) == {"A", "B"}
    assert report["gene_selection"]["n_genes_used"] == 60
    for channel in ("A", "B"):
        assert report["channels"][channel]["num_edges"] > 20
        assert report["splits"][channel]["total"] == report["channels"][channel]["num_edges"]


def test_channels_are_not_identical_and_overlap_is_reported(processed):
    dataset, report, _ = processed
    a = dataset.channel("A").known_positives()
    b = dataset.channel("B").known_positives()
    assert a != b, "channels must differ: they are separate evidence views"
    overlap = report["overlap"]
    assert 0.0 < overlap["jaccard"] < 1.0
    assert overlap["n_shared_edges"] < min(overlap["n_edges_a"], overlap["n_edges_b"])


def test_unmapped_protein_is_dropped(processed):
    """Trap: rows pointing at a protein missing from protein.info."""
    dataset, report, _ = processed
    assert report["channels"]["A"]["n_edges_unknown_protein"] >= 1
    assert report["gene_selection"]["n_proteins_mapped"] >= dataset.num_nodes


def test_features_are_shared_and_reduced(processed):
    dataset, report, _ = processed
    assert dataset.X.dtype == np.float32
    assert report["features"]["n_components"] == 16
    assert 0.5 < report["features"]["explained_variance_ratio_sum"] <= 1.0
    assert "never on PIE DE labels" in report["features"]["note"]


def test_downstream_targets_are_group_split_and_cover_the_graph(processed):
    dataset, report, _ = processed
    downstream = report["downstream"]
    assert downstream["coverage"] == 1.0
    assert downstream["label_rule_is_final"] is False
    partitions = [set(dataset.target_splits.get(name, ())) for name in ("train", "val", "test")]
    assert all(len(part) > 0 for part in partitions)
    assert sum(len(part) for part in partitions) == downstream["n_targets_usable"]
    for i in range(3):
        for j in range(i + 1, 3):
            assert partitions[i].isdisjoint(partitions[j])


def test_split_decisions_are_recorded_without_downstream_tables(synthetic_raw):
    config = PipelineConfig(
        dataset_name="no_downstream",
        string_links=str(synthetic_raw["string_links"]),
        string_info=str(synthetic_raw["string_info"]),
        gene_text=str(synthetic_raw["gene_text"]),
        gene_order=str(synthetic_raw["gene_order"]),
        n_components=8,
    )
    _, report = build_dataset(config)
    assert report.downstream["status"] == "no_de_tables_provided"
    decisions = split_decisions(config)
    assert decisions["edge_split_seed"] == config.split_seed
    assert report.config["n_components"] == 8
    assert json.dumps(report.as_dict())  # report must stay JSON serialisable


def test_build_is_reproducible_from_identical_inputs(synthetic_raw):
    config = PipelineConfig(
        dataset_name="repeat",
        string_links=str(synthetic_raw["string_links"]),
        string_info=str(synthetic_raw["string_info"]),
        gene_text=str(synthetic_raw["gene_text"]),
        gene_order=str(synthetic_raw["gene_order"]),
        n_components=8,
    )
    first, _ = build_dataset(config)
    second, _ = build_dataset(config)
    assert first.gene_ids == second.gene_ids
    assert np.array_equal(first.X, second.X)
    for channel in ("A", "B"):
        assert np.array_equal(first.channel(channel).train, second.channel(channel).train)
        assert np.array_equal(first.channel(channel).neg_train, second.channel(channel).neg_train)


def _tiny_raw(tmp_path, rows):
    """Write a minimal STRING-like triple: links, protein.info and gene_text."""
    header = "\t".join(["protein1", "protein2", "neighborhood",
                        "neighborhood_transferred", "fusion", "cooccurrence",
                        "homology", "coexpression", "experimental", "combined_score"])
    links = tmp_path / "links.txt"
    links.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    info = tmp_path / "info.txt"
    lines = ["#string_protein_id\tpreferred_name\tprotein_size\tannotation"]
    lines += [f"p{i}\tG{i}\t100\tx" for i in range(8)]
    info.write_text("\n".join(lines) + "\n", encoding="utf-8")
    matrix = np.random.default_rng(0).normal(size=(8, 10)).astype(np.float32)
    gene_text = tmp_path / "gene_text.npy"
    np.save(gene_text, matrix)
    order = tmp_path / "genes.txt"
    order.write_text("\n".join(f"G{i}" for i in range(8)) + "\n", encoding="utf-8")
    return links, info, gene_text, order


def test_combined_only_score_row_never_becomes_an_edge(tmp_path):
    """A pair whose only evidence is combined_score must stay out of both graphs."""
    real = ["p0\tp1", "p1\tp2", "p2\tp3", "p3\tp4", "p4\tp5", "p0\tp2", "p0\tp3", "p0\tp4"]
    real = [f"{pair}\t0\t0\t0\t0\t0\t500\t500\t900" for pair in real]
    # (p3, p5) is *not* planted through any channel, only via combined_score
    trap = "p3\tp5\t0\t0\t0\t0\t0\t0\t0\t999"
    links, info, gene_text, order = _tiny_raw(tmp_path, [*real, trap])
    config = PipelineConfig(dataset_name="trap", string_links=str(links),
                            string_info=str(info), gene_text=str(gene_text),
                            gene_order=str(order), n_components=4,
                            train_fraction=0.6, val_fraction=0.2)
    dataset, _ = build_dataset(config)
    index = {gene: position for position, gene in enumerate(dataset.gene_ids)}
    assert {"G3", "G5"}.issubset(index), "both trap endpoints must be active genes"
    trap_pair = (min(index["G3"], index["G5"]), max(index["G3"], index["G5"]))
    for channel in ("A", "B"):
        part = dataset.channel(channel)
        assert trap_pair not in part.known_positives()
        assert trap_pair not in undirected_pair_keys(part.encode_edges)


