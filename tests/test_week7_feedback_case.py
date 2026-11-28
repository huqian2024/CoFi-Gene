"""Week 7 feedback case: per-edge before/after weights from a real B6/B7 model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cofi_gene.feedback_report import feedback_edge_table, select_case  # noqa: E402
from cofi_gene.models import build_model  # noqa: E402


def _tiny_b7(processed_dataset, hidden: int = 8):
    model = build_model("b7", feature_dim=processed_dataset.num_features,
                        hidden_dim=hidden, fusion_dim=hidden, dropout=0.0)
    edges = {channel: torch.as_tensor(processed_dataset.channel(channel).encode_edges)
             for channel in model.channels}
    weights = {channel: torch.as_tensor(processed_dataset.channel(channel).encode_weights)
               for channel in model.channels}
    model.set_encoding_graphs(edges, weights, processed_dataset.num_nodes)
    x = torch.as_tensor(processed_dataset.X)
    return model, x


def test_edge_table_reports_bounded_before_after_weights(processed_dataset):
    model, x = _tiny_b7(processed_dataset)
    table = feedback_edge_table(model, x, "A", processed_dataset.num_nodes,
                                gene_ids=processed_dataset.gene_ids)
    n_edges = int(processed_dataset.channel("A").encode_edges.shape[1])
    assert table["num_encoder_edges"] == n_edges == len(table["edges"])
    assert table["alpha"] == pytest.approx(0.1)
    assert table["margin"] == pytest.approx(0.5)
    rows = table["edges"]
    assert len(rows) == n_edges
    for row in rows:
        assert row["delta"] == pytest.approx(row["weight_after"] - row["weight_before"])
        assert 0.0 <= row["probability"] <= 1.0
        # the feedback formula can only move an edge inside the margin band
        assert row["weight_after"] >= (1 - 0.5) * row["weight_before"] - 1e-6
        assert row["weight_after"] <= (1 + 0.5) * row["weight_before"] + 1e-6
    assert table["summary"]["num_up"] + table["summary"]["num_down"] + \
        table["summary"]["num_unchanged"] == n_edges


def test_top_k_keeps_the_largest_absolute_changes(processed_dataset):
    model, x = _tiny_b7(processed_dataset)
    full = feedback_edge_table(model, x, "A", processed_dataset.num_nodes)
    top = feedback_edge_table(model, x, "A", processed_dataset.num_nodes, top_k=3)
    assert len(top["edges"]) == 3
    assert top["summary"]["num_candidate_edges"] == len(full["edges"])
    deltas = [abs(row["delta"]) for row in top["edges"]]
    assert deltas == sorted(deltas, reverse=True)
    assert top["summary"]["max_abs_delta"] == pytest.approx(
        max(abs(row["delta"]) for row in full["edges"]))


def test_focus_genes_restrict_the_case_to_that_neighbourhood(processed_dataset):
    model, x = _tiny_b7(processed_dataset)
    first = int(processed_dataset.channel("A").encode_edges[0, 0])
    gene = processed_dataset.gene_ids[first]
    table = feedback_edge_table(model, x, "A", processed_dataset.num_nodes,
                                gene_ids=processed_dataset.gene_ids,
                                focus_genes=[gene], top_k=50)
    assert table["focus_genes"] == [gene]
    assert table["edges"], "the gene must have at least one encoder edge"
    for row in table["edges"]:
        assert gene in (row["u_id"], row["v_id"])
    assert table["summary"]["num_candidate_edges"] >= len(table["edges"])


def test_models_without_feedback_and_bad_channels_are_rejected(processed_dataset):
    plain = build_model("b1", feature_dim=processed_dataset.num_features,
                        hidden_dim=8, fusion_dim=8, dropout=0.0)
    with pytest.raises(ValueError, match="no feedback step"):
        feedback_edge_table(plain, torch.zeros(processed_dataset.num_nodes, 2),
                            "A", processed_dataset.num_nodes)
    model, x = _tiny_b7(processed_dataset)
    with pytest.raises(ValueError, match="not used by model"):
        feedback_edge_table(model, x, "Z", processed_dataset.num_nodes)


def test_empty_encoder_graph_is_reported(processed_dataset):
    model, x = _tiny_b7(processed_dataset)
    model.set_encoding_graphs(
        {"A": torch.zeros((2, 0), dtype=torch.long),
         "B": torch.as_tensor(processed_dataset.channel("B").encode_edges)},
        {"A": torch.zeros((0,)), "B": torch.as_tensor(
            processed_dataset.channel("B").encode_weights)},
        processed_dataset.num_nodes)
    with pytest.raises(ValueError, match="empty encoder graph"):
        feedback_edge_table(model, x, "A", processed_dataset.num_nodes)


def test_select_case_is_deterministic_and_validates_inputs(processed_dataset):
    model, x = _tiny_b7(processed_dataset)
    first = select_case({"edges": feedback_edge_table(
        model, x, "A", processed_dataset.num_nodes)["edges"], "channel": "A"}, top_k=5)
    second = select_case({"edges": feedback_edge_table(
        model, x, "A", processed_dataset.num_nodes)["edges"], "channel": "A"}, top_k=5)
    assert first["edges"] == second["edges"]
    with pytest.raises(ValueError, match="top_k"):
        select_case(first, top_k=0)
    with pytest.raises(ValueError, match="none of the focus genes"):
        select_case(first, focus_genes=["NOT_A_GENE"], gene_ids=processed_dataset.gene_ids)
