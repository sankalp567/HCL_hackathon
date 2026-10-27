"""Metadata filter construction and application.

Semantics:
  * hard filter on field F with values V: a chunk passes iff its F value is present and in V.
  * a chunk MISSING F fails a hard filter on F (we cannot confirm it satisfies the constraint)...
  * ...unless F is absent from the entire corpus: then the filter is *unavailable*, is not applied,
    and the limitation is reported (never silently "pretend it filtered").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .contracts import FieldMap, FilterSpec

# Priority-ordered alternatives (regex alternation takes the first that matches at a position):
#   A "FY2025-26"/"fiscal year 2025-26"/"2025-26"  B "FY25-26"  C "FY2026" (year the FY ends)  D "FY26"
#   E "year ended (31) March 2026" (Indian FY ends 31 March)  F "fiscal 2026" / "fiscal year 2026"
FY_PATTERN = (r"(?:\b(?:fy|fiscal(?:\s+year)?)\s*)?'?(20\d{2})\s*[-\u2013/]\s*(20\d{2}|\d{2})\b"
              r"|\bfy\s*'?(\d{2})\s*[-\u2013/]\s*(\d{2})\b"
              r"|\bfy\s*'?(20\d{2})\b"
              r"|\bfy\s*'?(\d{2})\b"
              r"|\b(?:fiscal\s+|financial\s+)?year\s+ended\s+(?:(?:31|31st)\s+)?march,?\s+(20\d{2})\b"
              r"|\bfiscal(?:\s+year)?\s+'?(20\d{2})\b(?![-\u2013/]\d)")
_FY_RE = re.compile(FY_PATTERN, re.I)


def _fy_from_match(m) -> str | None:
    g = m.groups()
    if g[0]:                                   # A
        y1, y2 = int(g[0]), int(g[1]) % 100
    elif g[2]:                                 # B
        y1, y2 = 2000 + int(g[2]), int(g[3])
    elif g[4] or g[6] or g[7]:                 # C / E / F: the calendar year in which the fiscal year ENDS
        end = int(g[4] or g[6] or g[7])
        return f"FY{end - 1}-{end % 100:02d}"
    else:                                      # D: FY26 -> FY2025-26
        y2 = int(g[5])
        return f"FY{2000 + y2 - 1}-{y2:02d}"
    return f"FY{y1}-{y2:02d}" if (y1 + 1) % 100 == y2 else None


def find_fiscal_year_spans(text: str) -> list[tuple[str, str]]:
    """[(normalized FY, literal text matched)] for every fiscal year written in `text` (in order, de-duplicated by FY)."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _FY_RE.finditer(text):
        fy = _fy_from_match(m)
        if fy and fy not in seen:
            seen.add(fy)
            out.append((fy, m.group(0)))
    return out


def find_fiscal_years(text: str) -> list[str]:
    """All fiscal years explicitly written in `text`, normalized, in order, de-duplicated."""
    return [fy for fy, _ in find_fiscal_year_spans(text)]


def normalize_fy(value: Any) -> str | None:
    """'2025-26','FY2025-26','FY25-26','fy 2025-2026','FY2026','FY26' -> 'FY2025-26'. None if not a fiscal year."""
    m = _FY_RE.fullmatch(str(value).strip())
    return _fy_from_match(m) if m else None


def norm_value(fld: str, v: Any) -> str:
    if fld in ("fiscal_year",):
        return normalize_fy(v) or str(v).strip().lower()
    return str(v).strip().lower()


def matches(meta: Mapping[str, Any], hard: Mapping[str, tuple]) -> bool:
    for fld, allowed in hard.items():
        if not allowed:          # empty tuple = no constraint (same as to_chroma_where)
            continue
        if fld not in meta or meta[fld] in (None, ""):
            return False
        have = norm_value(fld, meta[fld])
        if have not in {norm_value(fld, a) for a in allowed}:
            return False
    return True


def to_chroma_where(hard: Mapping[str, tuple], field_map: FieldMap | None = None) -> dict | None:
    """Build a ChromaDB `where` dict. One condition -> bare clause; several -> {"$and": [...]}."""
    fm = field_map or FieldMap()
    clauses = []
    for fld, vals in hard.items():
        key = fm.source(fld)
        # fiscal_year is compared in normalized form locally; send the same form (assumes the index stores "FY2025-26").
        # Other fields are sent verbatim and are case-sensitive in Chroma: confirm the stored spelling with Member 2.
        vals = [normalize_fy(v) or v for v in vals] if fld == "fiscal_year" else list(vals)
        if len(vals) == 1:
            clauses.append({key: {"$eq": vals[0]}})
        elif vals:
            clauses.append({key: {"$in": vals}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


@dataclass
class FilterReport:
    applied: dict[str, tuple] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)   # field -> reason
    limitations: list[str] = field(default_factory=list)


def resolve_filters(spec: FilterSpec, corpus_values: Mapping[str, set[str]]) -> tuple[FilterSpec, FilterReport]:
    """Drop hard filters that cannot be applied reliably; report why.

    corpus_values: canonical field -> set of normalized values observed in the corpus
    (empty/missing set => field has no metadata at all).
    """
    rep = FilterReport()
    keep: dict[str, tuple] = {}
    for fld, vals in spec.hard.items():
        seen = corpus_values.get(fld, set())
        if not seen:
            rep.unavailable[fld] = "metadata field absent from corpus"
            rep.limitations.append(f"filter '{fld}' not applied: metadata field absent from indexed chunks")
            continue
        wanted = {norm_value(fld, v) for v in vals}
        if not (wanted & seen):
            rep.unavailable[fld] = f"requested value(s) {sorted(wanted)} not present in corpus ({sorted(seen)[:6]})"
            rep.limitations.append(
                f"filter '{fld}'={sorted(wanted)} not applied: no indexed chunk has that value "
                f"(may appear only as a comparative column, not as a report year)")
            continue
        keep[fld] = tuple(vals)
        rep.applied[fld] = tuple(vals)
    return FilterSpec(keep, dict(spec.soft)), rep
