"""Assemble a :class:`~cofi_gene.dataset.CoFiDataset` from raw STRING/PIE inputs.

Two passes over the STRING link file keep memory bounded:

1. collect the gene symbols that actually carry evidence above the threshold
   (intersected with the feature/PIE gene universe when those are provided);
2. re-stream and materialise ``edge_index``/weights for the final node order.

Every step records counts into a report that the caller writes to the manifest,
so "N genes, E_A edges, E_B edges, X% overlap, Y% coverage" is always answered
from a file rather than from memory of a chat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .dataset import ChannelSplit, CoFiDataset
from .features import fit_feature_reducer, load_feature_matrix, load_gene_order
from .gene_ids import align_string_to_symbols, normalize_symbol, parse_string_protein_info
from .graph_build import (build_channel_graph, canonicalize_pairs, index_array_to_pairs,
                          overlap_stats, pairs_to_index_array)
from .negative_sampling import negative_pair_count, sample_negative_pairs
from .splits import (THREE_WAY, remove_edge_pairs, split_edges, split_perturbation_targets,
                     split_summary)
from .string_io import iter_string_channel_edges, open_text

__all__ = ["PipelineConfig", "BuildReport", "build_dataset", "split_decisions"]

CANONICAL_CHANNELS = ("A", "B")


@dataclass
class PipelineConfig:
    """All knobs of the data build. Defaults are deliberately explicit."""

    dataset_name: str = "unnamed"
    string_links: str | None = None
    string_info: str | None = None
    gene_text: str | None = None
    gene_order: str | None = None
    de_tables: dict[str, str] = field(default_factory=dict)  # context -> parquet
    channel_column_a: str = "experimental"
    channel_column_b: str = "coexpression"
    min_score_a: int = 0
    min_score_b: int = 0
    weight_mode: str = "none"
    n_components: int = 128
    standardize_features: bool = False
    p_adj_threshold: float = 0.05
    min_abs_log2fc: float | None = None
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    split_seed: int = 42
    target_train_fraction: float = 0.7
    target_val_fraction: float = 0.15
    negatives_ratio: float = 1.0
    negative_seed: int = 42
    mask_other_view: bool = False
    max_link_rows: int | None = None
    max_genes: int | None = None


@dataclass
class BuildReport:
    """Machine-readable record of what the build actually did."""

    dataset_name: str
    config: dict
    gene_selection: dict = field(default_factory=dict)
    alignment: dict = field(default_factory=dict)
    channels: dict = field(default_factory=dict)
    overlap: dict = field(default_factory=dict)
    splits: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    downstream: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def split_decisions(config: PipelineConfig) -> dict:
    """The split seed/ratios that any reproduction must reuse verbatim."""
    return {
        "edge_split_seed": config.split_seed,
        "train_fraction": config.train_fraction,
        "val_fraction": config.val_fraction,
        "target_split_seed": config.split_seed,
        "target_train_fraction": config.target_train_fraction,
        "target_val_fraction": config.target_val_fraction,
        "negative_seed": config.negative_seed,
        "negatives_ratio": config.negatives_ratio,
        "mask_other_view": config.mask_other_view,
    }


def _load_features(config: PipelineConfig):
    """Return ``(symbols, raw_matrix)`` for the shared node features."""
    if not config.gene_text:
        raise ValueError("gene_text is required: B0-B7 must share one node feature matrix")
    if not config.gene_order:
        raise ValueError("gene_order is required: features cannot be aligned to symbols by position alone")
    matrix = load_feature_matrix(config.gene_text)
    symbols = load_gene_order(config.gene_order)
    if len(symbols) != matrix.shape[0]:
        raise ValueError(f"gene order has {len(symbols)} rows but feature matrix has {matrix.shape[0]}")
    canonical = [normalize_symbol(s) for s in symbols]
    if any(not s for s in canonical):
        raise ValueError("gene order contains an unusable symbol")
    if len(set(canonical)) != len(canonical):
        raise ValueError("gene order collapses to duplicate symbols after normalisation")
    return tuple(canonical), matrix


def _scan_active_symbols(links: str, columns: dict[str, tuple[str, int]],
                         protein_to_symbol: dict[str, str], allowed: set[str],
                         max_rows: int | None = None) -> dict[str, set[str]]:
    """Streaming pass recording which symbols carry evidence per channel.

    Run once per channel (one scan each); memory stays bounded because only the
    symbol universe is kept, never the edge rows themselves.
    """
    import itertools

    from .string_io import iter_string_channel_edges

    active: dict[str, set[str]] = {}
    for name, (column, min_score) in columns.items():
        stream = iter_string_channel_edges(links, channel=column, min_score=min_score)
        if max_rows is not None:
            stream = itertools.islice(stream, max_rows)
        bucket: set[str] = set()
        for p1, p2, _score in stream:
            symbol1 = protein_to_symbol.get(p1)
            symbol2 = protein_to_symbol.get(p2)
            if symbol1 is not None and symbol1 in allowed:
                bucket.add(symbol1)
            if symbol2 is not None and symbol2 in allowed:
                bucket.add(symbol2)
        active[name] = bucket
    return active


def _weights_lookup(edge_index, weights) -> dict[tuple[int, int], float]:
    """Canonical pair -> weight, used to attach weights to split subsets."""
    pairs = zip(np.asarray(edge_index)[0].tolist(), np.asarray(edge_index)[1].tolist(),
                np.asarray(weights, dtype=float).tolist())
    return canonicalize_pairs(pairs)


def _weights_for(pairs, weight_of: dict[tuple[int, int], float]) -> np.ndarray:
    return np.asarray([weight_of.get(pair, 1.0) for pair in pairs], dtype=np.float32)


def _build_channel_splits(graphs: dict, edge_splits: dict, config: PipelineConfig,
                          num_nodes: int) -> dict[str, ChannelSplit]:
    """Combine per-channel positive splits, encoder graphs and negatives."""
    held_out = {name: set(split.val) | set(split.test) for name, split in edge_splits.items()}
    channels: dict[str, ChannelSplit] = {}
    for channel_index, (name, graph) in enumerate(sorted(graphs.items())):
        split = edge_splits[name]
        weight_of = _weights_lookup(graph.edge_index, graph.weights)
        encode_pairs = split.train
        if config.mask_other_view:
            other = [n for n in edge_splits if n != name]
            if other:
                encode_pairs = remove_edge_pairs(encode_pairs, held_out[other[0]], num_nodes)
        known = set(split.train) | set(split.val) | set(split.test)
        negatives = {}
        for split_index, split_name in enumerate(("train", "val", "test")):
            positives = getattr(split, split_name)
            count = negative_pair_count(len(positives), config.negatives_ratio)
            seed = config.negative_seed + 1000 * channel_index + split_index
            negatives[split_name] = sample_negative_pairs(num_nodes, count, known, seed=seed)
        channels[name] = ChannelSplit(
            train=pairs_to_index_array(split.train, num_nodes),
            val=pairs_to_index_array(split.val, num_nodes),
            test=pairs_to_index_array(split.test, num_nodes),
            train_weights=_weights_for(split.train, weight_of),
            val_weights=_weights_for(split.val, weight_of),
            test_weights=_weights_for(split.test, weight_of),
            neg_train=pairs_to_index_array(negatives["train"], num_nodes),
            neg_val=pairs_to_index_array(negatives["val"], num_nodes),
            neg_test=pairs_to_index_array(negatives["test"], num_nodes),
            encode_edges=pairs_to_index_array(encode_pairs, num_nodes),
            encode_weights=_weights_for(encode_pairs, weight_of),
            stats={"column": graph.stats.get("column"), "graph": graph.stats,
                   "num_encode_edges": len(encode_pairs),
                   "num_train": len(split.train), "num_val": len(split.val),
                   "num_test": len(split.test),
                   "num_neg_train": len(negatives["train"]),
                   "num_neg_val": len(negatives["val"]),
                   "num_neg_test": len(negatives["test"]),
                   "mask_other_view_applied": bool(config.mask_other_view)})
    return channels


def _build_downstream(config: PipelineConfig, gene_ids: tuple[str, ...], de_loader=None) -> tuple[dict, dict]:
    """Group-split the perturbation targets; never split response rows."""
    from .pie_io import de_labels, load_de_table, unique_targets

    if not config.de_tables:
        return {}, {"status": "no_de_tables_provided",
                    "note": "downstream targets unavailable; upstream experiments can still run"}
    loader = de_loader or load_de_table
    gene_set = set(gene_ids)
    per_context: dict[str, dict] = {}
    all_targets: set[str] = set()
    total_rows = 0
    total_positive = 0
    for context, path in sorted(config.de_tables.items()):
        table = loader(path, context=context)
        labels = de_labels(table, config.p_adj_threshold, config.min_abs_log2fc)
        total_rows += int(labels.size)
        total_positive += int(labels.sum())
        targets = {normalize_symbol(t) for t in unique_targets(table)}
        targets.discard("")
        all_targets |= targets
        covered = len(targets & gene_set)
        per_context[context] = {
            "path": str(path),
            "n_rows": int(labels.size),
            "n_positive": int(labels.sum()),
            "positive_rate": float(labels.mean()) if labels.size else None,
            "n_targets": len(targets),
            "n_targets_in_graph": covered,
            "target_coverage": (covered / len(targets)) if targets else 0.0,
        }
    usable = sorted(target for target in all_targets if target in gene_set)
    target_splits: dict[str, tuple[str, ...]] = {}
    if len(usable) >= 3:
        split = split_perturbation_targets(usable, config.target_train_fraction,
                                           config.target_val_fraction, config.split_seed)
        target_splits = {name: getattr(split, name) for name in THREE_WAY}
    report = {
        "p_adj_threshold": config.p_adj_threshold,
        "min_abs_log2fc": config.min_abs_log2fc,
        "label_rule_is_final": False,
        "contexts": per_context,
        "n_rows_total": total_rows,
        "n_positive_total": total_positive,
        "positive_rate_total": (total_positive / total_rows) if total_rows else None,
        "n_targets_total": len(all_targets),
        "n_targets_usable": len(usable),
        "coverage": (len(usable) / len(all_targets)) if all_targets else 0.0,
        "n_targets_per_split": {name: len(target_splits.get(name, ())) for name in THREE_WAY},
    }
    return target_splits, report


def build_dataset(config: PipelineConfig, de_loader=None) -> tuple[CoFiDataset, BuildReport]:
    """Run the full raw->processed build and return the dataset plus evidence."""
    import itertools

    warnings: list[str] = []
    report = BuildReport(dataset_name=config.dataset_name, config=asdict(config))

    feature_symbols, raw_features = _load_features(config)
    feature_index = {symbol: position for position, symbol in enumerate(feature_symbols)}
    report.features = {
        "source": str(config.gene_text),
        "gene_order": str(config.gene_order),
        "n_genes_available": len(feature_symbols),
        "n_raw_features": int(raw_features.shape[1]),
    }

    if not config.string_links or not config.string_info:
        raise ValueError("string_links and string_info are required")
    with open_text(config.string_info) as handle:
        protein_to_symbol = parse_string_protein_info(handle)
    # Auditable alignment: how many STRING proteins survive symbol normalisation
    # and the feature/PIE gene universe, and how many collide on one symbol.
    _, alignment_report = align_string_to_symbols(list(protein_to_symbol),
                                                  protein_to_symbol,
                                                  allowed_symbols=set(feature_symbols))
    report.alignment = alignment_report.as_dict()

    allowed = set(feature_symbols)
    columns = {"A": (config.channel_column_a, config.min_score_a),
               "B": (config.channel_column_b, config.min_score_b)}
    active = _scan_active_symbols(config.string_links, columns, protein_to_symbol,
                                 allowed, config.max_link_rows)
    union = sorted(active["A"] | active["B"])
    if config.max_genes is not None:
        union = union[:config.max_genes]
        warnings.append(f"gene list truncated to max_genes={config.max_genes} (smoke mode)")
    if len(union) < 3:
        raise ValueError("fewer than 3 genes carry evidence above threshold; check inputs/thresholds")
    gene_ids = tuple(union)
    index_of_symbol = {symbol: position for position, symbol in enumerate(gene_ids)}
    protein_to_index = {protein: index_of_symbol[symbol]
                        for protein, symbol in protein_to_symbol.items()
                        if symbol in index_of_symbol}
    report.gene_selection = {
        "n_feature_symbols": len(feature_symbols),
        "n_active_A": len(active["A"]),
        "n_active_B": len(active["B"]),
        "n_union_active": len(active["A"] | active["B"]),
        "n_genes_used": len(gene_ids),
        "n_proteins_mapped": len(protein_to_index),
    }

    rows = [feature_index[symbol] for symbol in gene_ids]
    raw_selected = raw_features[rows]
    reducer = fit_feature_reducer(raw_selected, config.n_components, config.standardize_features)
    X = reducer.transform(raw_selected)
    report.features.update({
        "n_genes_used": len(gene_ids),
        "n_components": int(X.shape[1]),
        "standardize": bool(config.standardize_features),
        "explained_variance_ratio_sum": float(reducer.explained_variance_ratio.sum()),
        "note": "reducer fitted on public gene priors only, never on PIE DE labels",
    })

    graphs = {}
    edge_splits = {}
    for name, (column, min_score) in columns.items():
        stream = iter_string_channel_edges(config.string_links, column, min_score)
        if config.max_link_rows is not None:
            stream = itertools.islice(stream, config.max_link_rows)
        graph = build_channel_graph(stream, protein_to_index, len(gene_ids),
                                    min_score=min_score, weight_mode=config.weight_mode)
        graph.stats["column"] = column
        graphs[name] = graph
        report.channels[name] = dict(graph.stats)
        edge_splits[name] = split_edges(index_array_to_pairs(graph.edge_index), len(gene_ids),
                                        config.train_fraction, config.val_fraction,
                                        config.split_seed)
        report.splits[name] = split_summary(edge_splits[name])
    report.overlap = overlap_stats(graphs["A"].edge_index, graphs["B"].edge_index)

    channels = _build_channel_splits(graphs, edge_splits, config, len(gene_ids))
    target_splits, downstream_report = _build_downstream(config, gene_ids, de_loader)
    report.downstream = downstream_report
    report.warnings = warnings

    meta = {
        "dataset_name": config.dataset_name,
        "channels": sorted(channels),
        "channel_columns": {name: column for name, (column, _) in columns.items()},
        "feature_reducer": "pca_svd_sign_fixed",
        "split": split_decisions(config),
        "provenance": "built by cofi_gene.pipeline.build_dataset",
    }
    dataset = CoFiDataset(gene_ids=gene_ids, X=X, channels=channels,
                          target_splits=target_splits, meta=meta)
    dataset.validate()
    return dataset, report




