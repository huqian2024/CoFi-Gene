"""Streaming readers for the published STRING link files.

CoFi-Gene needs exactly two evidence channels from
``<species>.protein.links.detailed.<version>.txt[.gz]``:

- column ``experimental``      -> graph A
- column ``coexpression``      -> graph B

``combined_score`` is deliberately never used as a substitute: it mixes the two
channels plus homology-based transfer, which would make the "two views" claim
circular. Reading is streaming so that the full human link file (~10^7 rows)
never has to sit in memory as Python objects.
"""

from __future__ import annotations

import gzip
from pathlib import Path

__all__ = ["CHANNELS", "read_string_header", "iter_string_channel_edges", "open_text"]

CHANNELS = ("experimental", "coexpression")


def open_text(path: str | Path):
    """Open a plain or gzip-compressed text file transparently."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_string_header(path: str | Path) -> list[str]:
    """Return the column names of a STRING links/aliases/info table."""
    with open_text(path) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped.split("\t") if "\t" in stripped else stripped.split()
    raise ValueError(f"no header line found in {path}")


def iter_string_channel_edges(path: str | Path, channel: str = "experimental",
                              min_score: int = 0):
    """Yield ``(protein1, protein2, score)`` triples for one evidence channel.

    ``min_score`` filters on that single channel's own score column, never on the
    combined score. A channel score of exactly 0 means "no evidence of this type"
    and is always skipped - otherwise a pair that has only *combined* evidence
    would silently become an edge in both graphs.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS}, got {channel!r}")
    if not 0 <= int(min_score) <= 1000:
        raise ValueError("min_score must be within STRING's 0-1000 range")
    with open_text(path) as handle:
        columns: list[str] | None = None
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            fields = stripped.split("\t" if "\t" in stripped else None)
            if columns is None:
                columns = [c.strip() for c in fields]
                for required in ("protein1", "protein2", channel):
                    if required not in columns:
                        raise ValueError(f"column {required!r} missing in {path}")
                i1 = columns.index("protein1")
                i2 = columns.index("protein2")
                isc = columns.index(channel)
                continue
            score = int(float(fields[isc]))
            if score <= 0 or score < min_score:
                continue
            yield fields[i1].strip(), fields[i2].strip(), score

