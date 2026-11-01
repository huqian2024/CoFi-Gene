"""Synthetic raw inputs that mirror the real STRING / PIE file layout.

The point is not realism but *contract fidelity*: the generator writes files whose
columns, ids and quirks match the published ones closely enough that the whole
build -> split -> train path can be exercised on a CPU laptop, with no data
download and no GPU. It also plants deliberate traps so the tests can prove the
pipeline rejects them:

- a row with ``combined_score`` high but both channel columns zero (must be
  filtered out: the combined score is never used as a channel);
- a gene with several STRING proteins mapping to one symbol (conflict counting);
- rows referencing proteins absent from ``protein.info`` (must be dropped);
- feature rows stored in a *different* order from the gene order used downstream
  (must be aligned by id, not by position).
"""

from __future__ import annotations

import gzip
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import numpy as np

__all__ = ["SyntheticSpec", "write_synthetic_raw"]

_LINK_COLUMNS = ("protein1", "protein2", "neighborhood", "neighborhood_transferred",
                 "fusion", "cooccurrence", "homology", "coexpression", "experimental",
                 "combined_score")


@dataclass
class SyntheticSpec:
    """Sizes and knobs of the fake dataset (small enough for fast tests)."""

    n_genes: int = 60
    n_raw_features: int = 64
    n_shared_edges: int = 60
    n_only_a_edges: int = 70
    n_only_b_edges: int = 55
    n_contexts: int = 2
    n_effects_per_target: int = 12
    n_downstream_targets: int = 24
    positive_rate: float = 0.3
    seed: int = 0
    species: str = "9606"
    version: str = "12.0"
    # Genes are assigned to communities; edges and features both key off the same
    # assignment, so link prediction is genuinely learnable (a plumbing test that
    # cannot detect a broken trainer would be worthless).
    n_communities: int = 4
    within_community_rate: float = 0.85

    @property
    def gene_symbols(self) -> tuple[str, ...]:
        return tuple(f"GENE{i:04d}" for i in range(self.n_genes))

    @property
    def communities(self) -> np.ndarray:
        return np.arange(self.n_genes) % max(2, self.n_communities)

    @property
    def protein_ids(self) -> tuple[str, ...]:
        return tuple(f"{self.species}.ENSP{i:011d}" for i in range(self.n_genes))

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(["K562", "RPE1", "HepG2", "Jurkat"][: self.n_contexts])



def _write_links(path: Path, spec: SyntheticSpec, rng: Random) -> dict:
    """Write a STRING ``links.detailed`` file and return the planted edge counts."""
    proteins = spec.protein_ids
    communities = spec.communities
    by_community = {c: [i for i in range(spec.n_genes) if communities[i] == c]
                    for c in set(communities.tolist())}
    pairs: dict[tuple[str, str], tuple[int, int]] = {}

    def pick_pair() -> tuple[int, int]:
        u = rng.randrange(spec.n_genes)
        pool = by_community[int(communities[u])]
        if len(pool) > 1 and rng.random() < spec.within_community_rate:
            v = rng.choice([candidate for candidate in pool if candidate != u])
        else:
            v = rng.randrange(spec.n_genes)
        return (u, v) if u < v else (v, u)

    def add(count: int, experimental: bool, coexpression: bool) -> None:
        for _ in range(count):
            u, v = pick_pair()
            if u == v:
                continue
            key = (proteins[u], proteins[v])
            exp_score = rng.randint(400, 999) if experimental else 0
            coexp_score = rng.randint(400, 999) if coexpression else 0
            pairs[key] = (exp_score, coexp_score)


    add(spec.n_shared_edges, True, True)
    add(spec.n_only_a_edges, True, False)
    add(spec.n_only_b_edges, False, True)

    lines = ["\t".join(_LINK_COLUMNS)]
    for (p1, p2), (exp_score, coexp_score) in sorted(pairs.items()):
        row = {
            "protein1": p1, "protein2": p2, "neighborhood": 0,
            "neighborhood_transferred": 0, "fusion": 0, "cooccurrence": 0,
            "homology": 0, "coexpression": coexp_score, "experimental": exp_score,
            "combined_score": max(exp_score, coexp_score, 800),
        }
        lines.append("\t".join(str(row[c]) for c in _LINK_COLUMNS))
    # Trap 1: only combined_score is high -> must never become an A/B edge.
    lines.append("\t".join(str({"protein1": proteins[0], "protein2": proteins[1],
                                "neighborhood": 0, "neighborhood_transferred": 0,
                                "fusion": 0, "cooccurrence": 0, "homology": 0,
                                "coexpression": 0, "experimental": 0,
                                "combined_score": 999}[c]) for c in _LINK_COLUMNS))
    # Trap 2: one row references a protein that is missing from protein.info.
    lines.append("\t".join(str({"protein1": proteins[2],
                                "protein2": f"{spec.species}.ENSP99999999999",
                                "neighborhood": 0, "neighborhood_transferred": 0,
                                "fusion": 0, "cooccurrence": 0, "homology": 0,
                                "coexpression": 700, "experimental": 700,
                                "combined_score": 900}[c]) for c in _LINK_COLUMNS))
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")
    return {
        "n_planted_pairs": len(pairs),
        "n_planted_shared": spec.n_shared_edges,
        "n_planted_only_a": spec.n_only_a_edges,
        "n_planted_only_b": spec.n_only_b_edges,
    }


def _write_info(path: Path, spec: SyntheticSpec) -> dict:
    """Write ``protein.info``; includes a two-proteins-one-symbol conflict."""
    provenances = [
        ("#string_protein_id", "preferred_name", "protein_size", "annotation"),
    ]
    rows = [(protein, symbol, 500, f"synthetic {symbol}")
            for protein, symbol in zip(spec.protein_ids, spec.gene_symbols)]
    # Trap 3: a second protein id for GENE0000 -> alignment conflict to be counted.
    rows.append((f"{spec.species}.ENSP00000000099", spec.gene_symbols[0], 480,
                 "synthetic duplicate symbol"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(provenances[0]) + "\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")
    return {"n_proteins_in_info": len(rows),
            "n_duplicate_symbol_entries": 1}


def _write_features(path_matrix: Path, path_order: Path, spec: SyntheticSpec) -> np.ndarray:
    """Write a ``gene_text``-like matrix plus an id order in a *permuted* row order.

    The features carry the community structure that also drives the edges, so a
    working trainer can beat the prevalence baseline on held-out edges.
    """
    rng = np.random.default_rng(spec.seed)
    communities = spec.communities
    n_communities = int(communities.max()) + 1
    onehot = np.zeros((spec.n_genes, n_communities), dtype=np.float32)
    onehot[np.arange(spec.n_genes), communities] = 1.0
    signal = onehot @ rng.normal(scale=2.0, size=(n_communities, spec.n_raw_features))
    noise = rng.normal(scale=0.4, size=(spec.n_genes, spec.n_raw_features))
    matrix = (signal + noise).astype(np.float32)
    permutation = np.random.default_rng(spec.seed + 1).permutation(spec.n_genes)
    path_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_matrix, matrix[permutation])
    symbols = spec.gene_symbols
    with path_order.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(symbols[i] for i in permutation) + "\n")
    return matrix


def _write_pie(path: Path, spec: SyntheticSpec, context: str) -> dict:
    """Write a PIE ``de/<cell_line>.parquet``-like table (needs pyarrow)."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError("synthetic PIE tables need pyarrow: pip install -e '.[data]'") from exc
    rng = np.random.default_rng(spec.seed + zlib.crc32(context.encode("utf-8")) % 1000)
    symbols = spec.gene_symbols
    targets = symbols[: spec.n_downstream_targets]
    target_col, feature_col, ref_col, lfc_col, p_col = [], [], [], [], []
    n_positive = 0
    for target in targets:
        features = rng.choice(len(symbols), size=spec.n_effects_per_target, replace=False)
        for position in features:
            is_positive = bool(rng.random() < spec.positive_rate)
            p_adj = float(rng.uniform(0.0001, 0.04)) if is_positive else \
                float(rng.uniform(0.06, 0.99))
            log2fc = float(rng.normal(0.0, 1.0)) if is_positive else \
                float(rng.normal(0.0, 0.15))
            target_col.append(target)
            feature_col.append(symbols[position])
            ref_col.append(float(rng.uniform(0.0, 12.0)))
            lfc_col.append(log2fc)
            p_col.append(p_adj)
            n_positive += int(is_positive)
    table = pa.table({
        "target": pa.array(target_col, type=pa.string()),
        "feature": pa.array(feature_col, type=pa.string()),
        "ref_mean": pa.array(ref_col, type=pa.float32()),
        "log2_fold_change": pa.array(lfc_col, type=pa.float32()),
        "p_adj": pa.array(p_col, type=pa.float64()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return {"context": context, "n_rows": len(target_col),
            "n_positive": n_positive,
            "positive_rate": (n_positive / len(target_col)) if target_col else 0.0,
            "n_targets": len(targets)}


def write_synthetic_raw(root: str | Path, spec: SyntheticSpec | None = None) -> dict:
    """Materialise a full fake raw-data tree and return its paths plus ground truth."""
    spec = spec or SyntheticSpec()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    links = root / f"string_{spec.species}.protein.links.detailed.v{spec.version}.txt.gz"
    info = root / f"string_{spec.species}.protein.info.v{spec.version}.txt"
    features = root / "gene_text.npy"
    order = root / "gene_text.ids.txt"
    pie_dir = root / "pie" / "de"

    rng = Random(spec.seed)
    link_truth = _write_links(links, spec, rng)
    info_truth = _write_info(info, spec)
    _write_features(features, order, spec)
    de_tables = {}
    pie_truth = []
    for context in spec.contexts:
        de_tables[context] = pie_dir / f"{context}.parquet"
    # parquet needs pyarrow; keep the whole build optional so CPU-only split tests
    # can still generate the STRING/feature part of the tree.
    pie_written = True
    try:
        for context, path in de_tables.items():
            pie_truth.append(_write_pie(path, spec, context))
    except ImportError:
        pie_written = False
        de_tables = {}

    truth = {
        "spec": asdict(spec),
        "genes": list(spec.gene_symbols),
        "links": link_truth,
        "info": info_truth,
        "downstream": pie_truth,
        "pie_written": pie_written,
    }
    (root / "synthetic_truth.json").write_text(
        json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "root": root,
        "string_links": links,
        "string_info": info,
        "gene_text": features,
        "gene_order": order,
        "de_tables": de_tables,
        "truth": truth,
    }

