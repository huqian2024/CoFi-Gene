"""Gene identifier normalisation and STRING -> symbol alignment.

The two graphs come from STRING protein links (``9606.ENSP...`` ids) while the
downstream PIE tables use gene symbols. Alignment therefore has to be explicit
and auditable: nothing may rely on dictionary iteration order, and every dropped
or ambiguous identifier is counted so the manifest can report coverage.

Policy implemented here (flagged as decision D-009 in ``docs/DECISIONS.md``,
pending team confirmation, overridable by the caller):

- Symbols are upper-cased and whitespace-stripped.
- A trailing Ensembl version suffix (``ENSG00000141510.17``) is removed.
- Placeholder strings (``nan``, ``none``, ``null``, ``-``, empty) are treated as
  missing rather than as a gene called "NAN".
- When several STRING proteins map to the same symbol, the lexicographically
  smallest protein id wins and the collision is counted as a conflict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "MISSING_TOKENS",
    "normalize_symbol",
    "strip_version_suffix",
    "parse_string_protein_info",
    "AlignmentReport",
    "align_string_to_symbols",
]

MISSING_TOKENS = frozenset({"", "-", "nan", "none", "null", "na", "n/a"})


def strip_version_suffix(identifier: str) -> str:
    """``ENSG00000141510.17`` -> ``ENSG00000141510``; other ids untouched."""
    identifier = identifier.strip()
    if "." in identifier:
        head, _, tail = identifier.rpartition(".")
        if head and tail.isdigit():
            return head
    return identifier


def normalize_symbol(symbol: str) -> str:
    """Canonical gene symbol, or ``""`` when the value is a placeholder."""
    if symbol is None:
        return ""
    text = str(symbol).strip()
    if text.lower() in MISSING_TOKENS:
        return ""
    text = strip_version_suffix(text)
    return text.upper()


def parse_string_protein_info(lines) -> dict[str, str]:
    """Parse STRING ``<species>.protein.info.txt`` into ``protein_id -> symbol``.

    The published file starts with ``#string_protein_id  preferred_name ...``;
    comment and blank lines are skipped, and rows with an unusable symbol are
    dropped (they are reported later as unmapped by :func:`align_string_to_symbols`).
    """
    mapping: dict[str, str] = {}
    header_checked = False
    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if not header_checked:
            header_checked = True
            # Some releases publish the header without a leading '#', in which
            # case the first data row is lost; detect that and fall back.
            if fields[0].lower().startswith("string_protein"):
                continue
        if len(fields) < 2:
            continue
        symbol = normalize_symbol(fields[1])
        if not symbol:
            continue
        mapping[fields[0].strip()] = symbol
    return mapping


@dataclass
class AlignmentReport:
    """Auditable counts for one STRING -> symbol alignment."""

    n_string_proteins: int = 0
    n_mapped_proteins: int = 0
    n_unique_symbols: int = 0
    n_symbol_conflicts: int = 0
    n_unmapped_proteins: int = 0
    examples_unmapped: tuple[str, ...] = ()
    conflicts: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_string_proteins": self.n_string_proteins,
            "n_mapped_proteins": self.n_mapped_proteins,
            "n_unique_symbols": self.n_unique_symbols,
            "n_symbol_conflicts": self.n_symbol_conflicts,
            "n_unmapped_proteins": self.n_unmapped_proteins,
            "examples_unmapped": list(self.examples_unmapped),
            "conflict_examples": {k: list(v) for k, v in list(self.conflicts.items())[:20]},
        }


def align_string_to_symbols(protein_ids, protein_to_symbol: dict[str, str],
                            allowed_symbols=None) -> tuple[dict[str, str], AlignmentReport]:
    """Build ``protein_id -> symbol`` restricted to ``allowed_symbols``.

    ``allowed_symbols`` (optional) is normally the set of symbols available in the
    gene-feature source or in the downstream PIE tables; restricting early keeps
    the two evidence graphs on a shared node set.
    """
    allowed = None
    if allowed_symbols is not None:
        allowed = {normalize_symbol(s) for s in allowed_symbols}
        allowed.discard("")
    report = AlignmentReport()
    chosen: dict[str, str] = {}
    per_symbol: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for protein in protein_ids:
        report.n_string_proteins += 1
        symbol = protein_to_symbol.get(protein, "")
        if not symbol or (allowed is not None and symbol not in allowed):
            report.n_unmapped_proteins += 1
            if len(unmapped) < 5:
                unmapped.append(protein)
            continue
        per_symbol.setdefault(symbol, []).append(protein)
    for symbol, proteins in per_symbol.items():
        proteins.sort()
        chosen[proteins[0]] = symbol
        if len(proteins) > 1:
            report.n_symbol_conflicts += 1
            report.conflicts[symbol] = tuple(proteins)
    report.n_mapped_proteins = len(chosen)
    report.n_unique_symbols = len(per_symbol)
    report.examples_unmapped = tuple(unmapped)
    return chosen, report
