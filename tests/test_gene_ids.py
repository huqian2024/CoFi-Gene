"""Gene identifier normalisation and STRING -> symbol alignment."""

import pytest

from cofi_gene.gene_ids import (align_string_to_symbols, normalize_symbol,
                               parse_string_protein_info, strip_version_suffix)


def test_symbol_normalisation_rules():
    assert normalize_symbol(" tp53 ") == "TP53"
    assert normalize_symbol("ENSG00000141510.17") == "ENSG00000141510"
    assert strip_version_suffix("9606.ENSP00000269305") == "9606.ENSP00000269305"
    for placeholder in ("", "  ", "nan", "None", "-", "N/A"):
        assert normalize_symbol(placeholder) == ""


def test_protein_info_parsing_skips_comments_and_unusable_symbols():
    lines = [
        "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n",
        "9606.ENSP00000000001\tTP53\t393\t tumour protein\n",
        "9606.ENSP00000000002\tnan\t100\tbroken\n",
        "\n",
        "9606.ENSP00000000003\tEGFR\t1210\t\n",
    ]
    mapping = parse_string_protein_info(lines)
    assert mapping == {"9606.ENSP00000000001": "TP53", "9606.ENSP00000000003": "EGFR"}


def test_alignment_reports_conflicts_and_unmapped_proteins():
    protein_to_symbol = {
        "p1": "TP53", "p2": "TP53", "p3": "EGFR", "p4": "NOTINUNIVERSE"}
    chosen, report = align_string_to_symbols(
        ["p1", "p2", "p3", "p4", "p_missing"], protein_to_symbol,
        allowed_symbols={"TP53", "EGFR"})
    # deterministic winner among colliding proteins: lexicographically smallest id
    assert chosen == {"p1": "TP53", "p3": "EGFR"}
    assert report.n_symbol_conflicts == 1
    assert report.conflicts["TP53"] == ("p1", "p2")
    assert report.n_unmapped_proteins == 2
    assert report.n_unique_symbols == 2
    payload = report.as_dict()
    assert payload["n_mapped_proteins"] == 2
    assert "p_missing" in payload["examples_unmapped"]


def test_alignment_without_allowed_universe_keeps_every_mapped_symbol():
    chosen, report = align_string_to_symbols(["a", "b"], {"a": "X", "b": "Y"})
    assert chosen == {"a": "X", "b": "Y"}
    assert report.n_unmapped_proteins == 0
    with pytest.raises(TypeError):
        align_string_to_symbols(None, {})
