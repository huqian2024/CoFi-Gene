"""STRING link reader: channel isolation, zero-score handling, gzip support."""

import gzip

import pytest

from cofi_gene.string_io import (CHANNELS, iter_string_channel_edges, open_text,
                                read_string_header)

HEADER = "\t".join(["protein1", "protein2", "neighborhood", "neighborhood_transferred",
                    "fusion", "cooccurrence", "homology", "coexpression",
                    "experimental", "combined_score"])


def _row(p1, p2, experimental, coexpression, combined=900):
    return "\t".join([p1, p2, "0", "0", "0", "0", "0", str(coexpression),
                      str(experimental), str(combined)])


@pytest.fixture()
def links_file(tmp_path):
    path = tmp_path / "9606.protein.links.detailed.v12.0.txt"
    path.write_text("\n".join([
        HEADER,
        _row("p1", "p2", 700, 300),      # both channels have evidence
        _row("p1", "p3", 500, 0),        # experimental only
        _row("p2", "p3", 0, 800),        # co-expression only
        _row("p3", "p4", 0, 0, 999),     # combined-only trap: must never appear
    ]) + "\n", encoding="utf-8")
    return path


def test_header_and_gzip_open(tmp_path, links_file):
    assert read_string_header(links_file) == HEADER.split("\t")
    gz = tmp_path / "links.txt.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as handle:
        handle.write(links_file.read_text(encoding="utf-8"))
    assert read_string_header(gz) == HEADER.split("\t")
    with open_text(gz) as handle:
        assert handle.readline().startswith("protein1")


def test_channels_are_independent_and_zero_scores_are_never_edges(links_file):
    experimental = list(iter_string_channel_edges(links_file, "experimental"))
    coexpression = list(iter_string_channel_edges(links_file, "coexpression"))
    assert [tuple(row) for row in experimental] == [("p1", "p2", 700), ("p1", "p3", 500)]
    assert [tuple(row) for row in coexpression] == [("p1", "p2", 300), ("p2", "p3", 800)]
    # the combined-only row must not leak into either channel
    assert all(row[:2] != ("p3", "p4") for row in experimental + coexpression)


def test_min_score_threshold_applies_to_the_single_channel_column(links_file):
    assert list(iter_string_channel_edges(links_file, "experimental", min_score=600)) == \
        [("p1", "p2", 700)]
    assert list(iter_string_channel_edges(links_file, "coexpression", min_score=500)) == \
        [("p2", "p3", 800)]


def test_invalid_arguments_are_rejected(links_file, tmp_path):
    with pytest.raises(ValueError, match="channel must be"):
        list(iter_string_channel_edges(links_file, "combined_score"))
    with pytest.raises(ValueError, match="0-1000"):
        list(iter_string_channel_edges(links_file, "experimental", min_score=5000))
    with pytest.raises(FileNotFoundError):
        open_text(tmp_path / "nope.txt")
    bad = tmp_path / "bad.txt"
    bad.write_text("protein1\tprotein2\np1\tp2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        list(iter_string_channel_edges(bad, "experimental"))
    assert set(CHANNELS) == {"experimental", "coexpression"}
