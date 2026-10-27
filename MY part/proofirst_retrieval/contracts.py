"""Shared data contracts.

Canonical metadata keys (the contract Member 2's index must map onto via FieldMap):
  company, fiscal_year (REPORT year, e.g. "FY2025-26"), doc_id, pdf_page_index (0-based),
  printed_page (nullable), content_type ("table"|"narrative"|...), scope ("consolidated"|"standalone"),
  table_id, table_title, headers ("a|b|c"), row_label, row_index, unit, periods ("FY2025-26|FY2024-25").
All keys are optional; code must degrade (and report) when a key is absent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

CANONICAL_FIELDS = (
    "company", "fiscal_year", "doc_id", "pdf_page_index", "printed_page", "content_type", "scope",
    "table_id", "table_title", "headers", "row_label", "row_index", "unit", "periods",
)


@dataclass(frozen=True)
class FieldMap:
    """canonical field -> key used in the vector store's metadata. Identity by default."""
    mapping: Mapping[str, str] = field(default_factory=dict)

    def source(self, canonical: str) -> str:
        return self.mapping.get(canonical, canonical)

    def to_canonical(self, raw: Mapping[str, Any] | None) -> dict[str, Any]:
        raw = raw or {}
        inv = {self.source(c): c for c in CANONICAL_FIELDS}
        out = {inv.get(k, k): v for k, v in raw.items()}
        return out


@dataclass
class Candidate:
    """One retrieved chunk. Ranks are 1-based; scores are retriever-native (not comparable across retrievers)."""
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ranks: dict[str, int] = field(default_factory=dict)       # retriever/list name -> 1-based rank
    scores: dict[str, float] = field(default_factory=dict)    # retriever/list name -> native score
    fusion_score: float | None = None
    rerank_score: float | None = None
    soft_boost: float = 0.0

    @property
    def doc_id(self) -> str | None:
        return self.metadata.get("doc_id")

    def copy(self) -> "Candidate":
        return Candidate(self.chunk_id, self.text, dict(self.metadata), dict(self.ranks), dict(self.scores),
                         self.fusion_score, self.rerank_score, self.soft_boost)

    def index_text(self) -> str:
        """Text used for lexical/dense scoring: chunk text + table context from metadata."""
        m = self.metadata
        parts = [self.text]
        for k in ("table_title", "headers", "row_label"):
            v = m.get(k)
            if v:
                parts.append(str(v).replace("|", " "))
        return "\n".join(parts)


@dataclass
class QueryAnalysis:
    original: str
    companies: list[str] = field(default_factory=list)       # only if explicitly mentioned
    periods: list[str] = field(default_factory=list)         # normalized "FY2025-26"; only if explicit
    relative_period: str | None = None                       # "previous"/"latest": noted, never turned into a filter
    metrics: list[str] = field(default_factory=list)         # canonical metric ids
    ambiguous_terms: list[str] = field(default_factory=list)
    scope: str | None = None                                 # only if explicit
    intent: str = "unknown"                                  # numeric_lookup|comparison|trend|qualitative|unknown
    expansion_terms: list[str] = field(default_factory=list) # controlled alias expansions (lexical only)
    notes: list[str] = field(default_factory=list)
    narrative_cue: bool = False                              # wording like policy/strategy/risk/why: asks for narrative, not a table
    cross_company: bool = False                              # wording asks about companies collectively ("which company", "all three")
    surface: dict[str, list[str]] = field(default_factory=dict)   # field -> literal spans in the question that express the constraint
    enforced: dict[str, tuple] = field(default_factory=dict)      # hard filters actually applied by the pipeline (set at run time)


@dataclass(frozen=True)
class FilterSpec:
    """hard: user-specified constraints (field -> allowed values). soft: preferences (boost only)."""
    hard: Mapping[str, tuple] = field(default_factory=dict)
    soft: Mapping[str, tuple] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.hard

    def without(self, fld: str) -> "FilterSpec":
        return FilterSpec({k: v for k, v in self.hard.items() if k != fld}, dict(self.soft))

    def with_hard(self, fld: str, values: tuple) -> "FilterSpec":
        h = dict(self.hard); h[fld] = tuple(values)
        return FilterSpec(h, dict(self.soft))


@dataclass
class EvidenceRecord:
    company: str | None
    doc_id: str | None
    pdf_page_index: int | None
    printed_page: Any
    chunk_id: str
    source_text: str
    content_type: str | None
    scope: str | None
    table_id: str | None
    table_title: str | None
    headers: str | None
    row_label: str | None
    unit: str | None
    periods: str | None
    fiscal_year: str | None
    table_context: str | None          # full table text (all rows) when the hit is a table row
    matched_chunk_ids: list[str]
    scores: dict[str, float]
    ranks: dict[str, int]
    fusion_score: float | None
    rerank_score: float | None
    # NOTE: scores are retrieval relevance only; they never validate a numeric value.
