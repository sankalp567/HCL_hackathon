import math
import pytest
from proofirst_retrieval import metrics as M


Q = {"a": 3, "b": 1}


def test_recall_precision():
    assert M.recall_at_k(["x", "a", "y", "b"], Q, 2) == 0.5
    assert M.recall_at_k(["x", "a", "y", "b"], Q, 4) == 1.0
    assert M.precision_at_k(["a", "x", "y", "b", "z"], Q, 5) == pytest.approx(2 / 5)
    assert M.precision_at_k(["a"], Q, 5) == pytest.approx(1 / 5)  # denominator is K, not len(results)


def test_mrr():
    assert M.mrr(["x", "y", "b"], Q) == pytest.approx(1 / 3)
    assert M.mrr(["x", "y"], Q) == 0.0
    assert M.mrr(["a"], Q) == 1.0


def test_ndcg_known_values():
    # ideal order a(3),b(1): DCG=(7/1)+(1/log2(3)); ranking b,a gives (1/1)+(7/log2(3))
    ideal = 7 + 1 / math.log2(3)
    assert M.ndcg_at_k(["a", "b"], Q, 10) == pytest.approx(1.0)
    assert M.ndcg_at_k(["b", "a"], Q, 10) == pytest.approx((1 + 7 / math.log2(3)) / ideal)
    assert M.ndcg_at_k(["x", "y"], Q, 10) == 0.0
    assert M.ndcg_at_k(["x", "a"], Q, 1) == 0.0


def test_no_relevant_returns_none_and_is_excluded_from_mean():
    for f in (lambda: M.recall_at_k(["a"], {}, 5), lambda: M.precision_at_k(["a"], {}, 5),
              lambda: M.mrr(["a"], {}), lambda: M.ndcg_at_k(["a"], {}, 5)):
        assert f() is None
    assert M.mean_ignoring_none([1.0, None, 0.0]) == (0.5, 2)
    assert M.mean_ignoring_none([None]) == (None, 0)
    assert all(v is None for v in M.query_metrics(["a"], {}).values())


def test_query_metrics_keys():
    assert set(M.query_metrics(["a"], Q)) == {"recall@5", "recall@10", "precision@5", "mrr", "ndcg@5", "ndcg@10"}


def test_numeric_metrics():
    top = [{"chunk_id": "c1", "company": "Acme", "fiscal_year": "FY2025-26", "doc_id": "d1", "pdf_page_index": 12,
            "table_id": "t1", "scope": "consolidated"},
           {"chunk_id": "c2", "company": "Borealis", "doc_id": "d2", "pdf_page_index": 3}]
    e = {"companies": ["Acme"], "fiscal_year": "FY2025-26", "scope": "consolidated", "table_ids": ["t1"],
         "row_chunk_ids": ["c1"], "pages": [["d1", 12]]}
    m = M.numeric_evidence_metrics(top, e, 5)
    assert m["source_page_hit"] == m["company_year_hit"] == m["metric_table_hit"] == m["row_evidence_hit"] == 1.0
    assert m["scope_match@1"] == 1.0 and m["company_coverage"] is None
    miss = M.numeric_evidence_metrics(top[1:], e, 5)
    assert miss["source_page_hit"] == 0.0 and miss["scope_match@1"] == 0.0


def test_numeric_metrics_coverage_and_missing_labels():
    top = [{"chunk_id": "a", "company": "Acme"}, {"chunk_id": "b", "company": "Cygnus"}]
    m = M.numeric_evidence_metrics(top, {"companies": ["Acme", "Borealis"]}, 5)
    assert m["company_coverage"] == 0.5 and m["company_full_coverage"] == 0.0
    none = M.numeric_evidence_metrics(top, {}, 5)
    assert all(v is None for v in none.values())


def test_percentile_requires_sample():
    assert M.percentile([1.0] * 5, 50) is None
    assert M.percentile(list(range(100)), 50) == pytest.approx(49.5)


def test_duplicate_ranked_ids_do_not_inflate_ndcg():
    assert M.ndcg_at_k(["a", "a"], {"a": 1}, 5) == pytest.approx(1.0)
    assert M.mrr(["x", "x", "a"], {"a": 1}) == pytest.approx(1 / 2)
