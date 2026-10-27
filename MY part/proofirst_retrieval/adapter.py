"""Adapter boundary: RetrievalResult -> `POST /api/evidence/search`-shaped dict.

STATUS: PROVISIONAL. Field names follow docs/TEAM_ARCHITECTURE_AND_RETRIEVAL.md (sections 5.3, 10, 13),
NOT a confirmed API from Member 2. Nothing is invented: values the retrieval layer does not carry
(bbox, pdf_page_number, row_idx, col_idx) are emitted as None.

Integration requirements still to be confirmed with Member 2 (this module assumes none of them):
  1. ChromaDB metadata field names and stored spellings (map via contracts.FieldMap, not here).
  2. Distance metric and score direction of the dense channel (channel_scores are copied as-is).
  3. Embedding model revision used at index time.
  4. Collection name(s).
  5. pdf_page_index base: contracts.py documents 0-based, the design doc says 0-based, but this is
     UNCONFIRMED against the real index. The value is copied through unchanged; pdf_page_number is None
     rather than guessed as index+1.
  6. Bounding boxes (design: normalized TOPLEFT + page size) are not retrieved today -> bbox None.

Semantics: absence_is_proof is ALWAYS False. A retrieval outcome of no_evidence / low_relevance / error
is never proof that a fact is absent from the reports; `refusal_hint` names the matching section 13.3
refusal code (or "retrieval_error") for the downstream orchestrator to use.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any

from .contracts import EvidenceRecord
from .pipeline import RetrievalResult

_NOT_IN_EVIDENCE = "R_NOT_IN_RETRIEVED_EVIDENCE"  # design doc 13.3, verbatim
_REFUSAL_HINTS = {
    "no_evidence": _NOT_IN_EVIDENCE,
    "low_relevance": _NOT_IN_EVIDENCE,
    "error": "retrieval_error",  # not a 13.3 code: a retrieval failure, not a refusal verdict
}


def _jsonable(v: Any) -> Any:
    """Recursively coerce to JSON-safe values (tuples/sets -> lists, dataclasses -> dicts, rest -> str)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, enum.Enum):
        return _jsonable(v.value)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return _jsonable(dataclasses.asdict(v))
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_jsonable(x) for x in v]
    return str(v)


def _kind(e: EvidenceRecord) -> str:
    if e.row_label:
        return "row"
    return "table" if e.table_id else "paragraph"


def _hit(e: EvidenceRecord) -> dict[str, Any]:
    return {
        "evidence_id": e.chunk_id,
        "kind": _kind(e),
        "locator": {
            "document_id": e.doc_id,
            "pdf_page_index": e.pdf_page_index,   # copied unchanged; base convention unconfirmed
            "pdf_page_number": None,              # not carried by retrieval; never derived from index
            "printed_page_label": None if e.printed_page is None else str(e.printed_page),
            "table_id": e.table_id,
            "row_idx": None,
            "col_idx": None,
            "bbox": None,
        },
        "text": e.source_text,
        "context": {
            "table_context": e.table_context or None,
            "table_title": e.table_title,
            "headers": e.headers,
            "row_label": e.row_label,
            "unit": e.unit,
            "periods": e.periods,
            "scope": e.scope,
            "fiscal_year": e.fiscal_year,
            "company": e.company,
            "content_type": e.content_type,
        },
        "matched_chunk_ids": list(e.matched_chunk_ids or []),
        "channel_scores": {
            "scores": dict(e.scores or {}),   # retriever-native, not comparable across channels
            "ranks": dict(e.ranks or {}),
        },
        "fusion_score": e.fusion_score,
        "rerank_score": e.rerank_score,
    }


def to_evidence_search_response(result: RetrievalResult, *, query: str) -> dict[str, Any]:
    """Map a RetrievalResult to a plain JSON-serialisable evidence-search response (PROVISIONAL)."""
    status = getattr(result, "status", "ok")
    return _jsonable({
        "query": query,
        "status": status,
        "status_note": getattr(result, "status_note", ""),
        "refusal_hint": _REFUSAL_HINTS.get(status),
        "absence_is_proof": False,  # hard-wired; never read from the result
        "hits": [_hit(e) for e in result.evidence],
        "coverage": dict(result.coverage or {}),
        "limitations": list(result.limitations or []),
        "retrieval_errors": list(getattr(result, "retrieval_errors", []) or []),
        "routing": dict(getattr(result, "routing", {}) or {}),
        "top_scores": dict(getattr(result, "top_scores", {}) or {}),
        "below_threshold": list(getattr(result, "below_threshold", []) or []),
        "receipt": {
            "n_candidates": len(result.candidates),
            "counts": dict(result.counts or {}),
            "reranker_status": result.reranker_status,
            "config": dict(result.config or {}),
            "absence_is_proof": False,
        },
    })
