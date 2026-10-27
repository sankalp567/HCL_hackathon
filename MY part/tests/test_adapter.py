import json

import pytest

from proofirst_retrieval.adapter import to_evidence_search_response
from proofirst_retrieval.contracts import EvidenceRecord, QueryAnalysis
from proofirst_retrieval.pipeline import RetrievalResult


def ev(**kw):
    base = dict(company="HCLTech", doc_id="hcl-ar", pdf_page_index=114, printed_page="112", chunk_id="c1",
                source_text="Revenue 100", content_type="table", scope="consolidated", table_id="t1",
                table_title="P&L", headers="a|b", row_label="Revenue", unit="INR crore",
                periods="FY2025-26|FY2024-25", fiscal_year="FY2025-26", table_context="full table",
                matched_chunk_ids=["c1", "c2"], scores={"bm25": 3.2, "dense": 0.8}, ranks={"bm25": 1, "dense": 2},
                fusion_score=0.03, rerank_score=0.9)
    base.update(kw)
    return EvidenceRecord(**base)


def res(evidence=None, **kw):
    base = dict(analysis=QueryAnalysis(original="q"), candidates=[], evidence=evidence or [], timings_ms={},
                counts={"bm25": 5}, filter_report=None, limitations=["lim"], coverage={"HCLTech": True, "TCS": False},
                reranker_status="ok", config={"k": 10})
    base.update(kw)
    return RetrievalResult(**base)


def test_field_mapping():
    out = to_evidence_search_response(res([ev()]), query="rev")
    h = out["hits"][0]
    assert out["query"] == "rev"
    assert h["evidence_id"] == "c1" and h["text"] == "Revenue 100"
    loc = h["locator"]
    assert loc["document_id"] == "hcl-ar" and loc["pdf_page_index"] == 114
    assert loc["printed_page_label"] == "112" and loc["table_id"] == "t1"
    assert loc["bbox"] is None and loc["pdf_page_number"] is None and loc["row_idx"] is None
    c = h["context"]
    assert c["table_context"] == "full table" and c["table_title"] == "P&L" and c["headers"] == "a|b"
    assert c["row_label"] == "Revenue" and c["unit"] == "INR crore" and c["scope"] == "consolidated"
    assert c["fiscal_year"] == "FY2025-26" and c["company"] == "HCLTech" and c["periods"] == "FY2025-26|FY2024-25"
    assert h["matched_chunk_ids"] == ["c1", "c2"]
    assert h["channel_scores"] == {"scores": {"bm25": 3.2, "dense": 0.8}, "ranks": {"bm25": 1, "dense": 2}}
    assert h["fusion_score"] == 0.03 and h["rerank_score"] == 0.9


def test_page_index_copied_unchanged():
    assert to_evidence_search_response(res([ev(pdf_page_index=0)]), query="q")["hits"][0]["locator"]["pdf_page_index"] == 0


@pytest.mark.parametrize("kw,kind", [
    ({}, "row"),
    ({"row_label": None}, "table"),
    ({"row_label": None, "table_id": None}, "paragraph"),
])
def test_kind(kw, kind):
    assert to_evidence_search_response(res([ev(**kw)]), query="q")["hits"][0]["kind"] == kind


def test_coverage_limitations_receipt_preserved():
    out = to_evidence_search_response(res([ev()], routing={"company": "explicit"}, top_scores={"bm25": 3.2},
                                          below_threshold=[{"chunk_id": "x"}]), query="q")
    assert out["coverage"] == {"HCLTech": True, "TCS": False}
    assert out["limitations"] == ["lim"]
    assert out["routing"] == {"company": "explicit"} and out["top_scores"] == {"bm25": 3.2}
    assert out["below_threshold"] == [{"chunk_id": "x"}]
    r = out["receipt"]
    assert r["n_candidates"] == 0 and r["counts"] == {"bm25": 5} and r["reranker_status"] == "ok"
    assert r["config"] == {"k": 10}


def test_status_passthrough_and_errors():
    out = to_evidence_search_response(res(status="degraded", status_note="n", retrieval_errors=["dense: boom"]), query="q")
    assert out["status"] == "degraded" and out["status_note"] == "n" and out["retrieval_errors"] == ["dense: boom"]


def test_json_round_trip_with_non_json_config():
    out = to_evidence_search_response(res([ev()], config={"fields": ("a", "b"), "s": {1, 2}}), query="q")
    assert json.loads(json.dumps(out)) == out
    assert out["receipt"]["config"]["fields"] == ["a", "b"]


@pytest.mark.parametrize("status,hint", [
    ("ok", None), ("degraded", None), ("partial_coverage", None),
    ("low_relevance", "R_NOT_IN_RETRIEVED_EVIDENCE"),
    ("no_evidence", "R_NOT_IN_RETRIEVED_EVIDENCE"),
    ("error", "retrieval_error"),
])
def test_absence_never_proof_and_refusal_hint(status, hint):
    # even if a caller (wrongly) sets absence_is_proof=True, the adapter never propagates it
    out = to_evidence_search_response(res(status=status, absence_is_proof=True), query="q")
    assert out["absence_is_proof"] is False and out["receipt"]["absence_is_proof"] is False
    assert out["refusal_hint"] == hint


def test_missing_optional_metadata_is_none():
    e = ev(doc_id=None, pdf_page_index=None, printed_page=None, table_id=None, table_title=None, headers=None,
           row_label=None, unit=None, periods=None, fiscal_year=None, table_context=None, company=None,
           scope=None, content_type=None, scores={}, ranks={}, fusion_score=None, rerank_score=None,
           matched_chunk_ids=[])
    h = to_evidence_search_response(res([e]), query="q")["hits"][0]
    assert h["kind"] == "paragraph"
    assert h["locator"]["document_id"] is None and h["locator"]["printed_page_label"] is None
    assert h["context"]["table_context"] is None and h["context"]["unit"] is None
    assert h["fusion_score"] is None and h["rerank_score"] is None


def test_empty_result():
    out = to_evidence_search_response(res(status="no_evidence"), query="q")
    assert out["hits"] == [] and out["absence_is_proof"] is False
