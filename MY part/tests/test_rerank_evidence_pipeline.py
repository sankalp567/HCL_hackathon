import pytest
from conftest import mk
from proofirst_retrieval.contracts import QueryAnalysis
from proofirst_retrieval.evidence import CorpusIndex, assemble, company_coverage, select_with_company_quota
from proofirst_retrieval.pipeline import PipelineConfig
from proofirst_retrieval.rerank import (CrossEncoderReranker, LexicalFeatureReranker, NoReranker, RerankerUnavailable,
                                        cross_encoder_status)


A = QueryAnalysis(original="consolidated revenue from operations")


def cands():
    return [mk("a", "unrelated words", company="X"), mk("b", "Revenue from operations", company="Y", scope="consolidated")]


def test_no_reranker_identity():
    assert [c.chunk_id for c in NoReranker().rerank(A, cands())] == ["a", "b"]


def test_lexical_reranker_structure_and_inputs_untouched():
    inp = cands()
    out = LexicalFeatureReranker().rerank(A, inp)
    assert [c.chunk_id for c in out] == ["b", "a"]
    assert all(c.rerank_score is not None for c in out) and inp[0].rerank_score is None
    assert out[0].metadata == {"company": "Y", "scope": "consolidated"}


def test_cross_encoder_with_injected_scorer_preserves_metadata():
    ce = CrossEncoderReranker(scorer=lambda pairs: [0.1, 0.9])
    out = ce.rerank(A, cands())
    assert [c.chunk_id for c in out] == ["b", "a"] and out[0].rerank_score == 0.9
    assert out[0].metadata["company"] == "Y"
    assert ce.rerank(A, []) == []


def test_cross_encoder_bad_scorer_output_raises():
    with pytest.raises(ValueError):
        CrossEncoderReranker(scorer=lambda pairs: [1.0]).rerank(A, cands())


def test_cross_encoder_unavailable_is_explicit(monkeypatch):
    import builtins
    real = builtins.__import__
    def fake(name, *a, **k):
        if name.startswith("sentence_transformers"):
            raise ImportError("no st")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RerankerUnavailable):
        CrossEncoderReranker().rerank(A, cands())
    ok, why = cross_encoder_status()
    assert isinstance(ok, bool) and why


# ---- evidence assembly ----
TABLE = [mk(f"r{i}", f"row{i} | {i}", company="Acme", doc_id="d", pdf_page_index=5, printed_page=2, table_id="t1",
            table_title="P&L", headers="Particulars|FY2025-26", row_label=f"row{i}", row_index=i, unit="INR crore",
            content_type="table", scope="consolidated", fiscal_year="FY2025-26") for i in range(3)]


def test_table_context_preserved_and_provenance_kept():
    idx = CorpusIndex(TABLE)
    ev = assemble([TABLE[1]], idx)
    assert len(ev) == 1
    e = ev[0]
    assert "row0 | 0" in e.table_context and "row2 | 2" in e.table_context and "P&L" in e.table_context and "INR crore" in e.table_context
    assert (e.doc_id, e.pdf_page_index, e.printed_page, e.chunk_id, e.unit, e.row_label) == ("d", 5, 2, "r1", "INR crore", "row1")
    assert e.source_text == "row1 | 1"


def test_same_table_hits_collapse_into_one_record():
    ev = assemble([TABLE[2], TABLE[0]], CorpusIndex(TABLE))
    assert len(ev) == 1 and ev[0].chunk_id == "r2" and ev[0].matched_chunk_ids == ["r2", "r0"]
    assert len(assemble([TABLE[2], TABLE[0]], CorpusIndex(TABLE), collapse_tables=False)) == 2


def test_narrative_has_no_table_context():
    ev = assemble([mk("n", "text", company="Acme")], CorpusIndex(TABLE))
    assert ev[0].table_context is None


def test_company_quota_and_coverage():
    c = [mk(f"a{i}", company="A") for i in range(6)] + [mk("b0", company="B")]
    top = select_with_company_quota(c, 4, ["A", "B"])
    assert "b0" in [x.chunk_id for x in top] and len(top) == 4
    assert [x.chunk_id for x in top] == [x.chunk_id for x in sorted(top, key=lambda x: [y.chunk_id for y in c].index(x.chunk_id))]
    assert select_with_company_quota(c, 4, ["A"]) == c[:4]       # single company: plain top-k
    assert company_coverage(top, ["A", "B", "Z"]) == {"A": True, "B": True, "Z": False}


# ---- pipeline on the synthetic fixture ----
def run(fixture_pipe, q, **kw):
    pipe, _ = fixture_pipe
    return pipe.run(q, PipelineConfig(**kw))


def test_pipeline_filters_restrict_to_requested_company_year(fixture_pipe):
    r = run(fixture_pipe, "What was Acme's consolidated revenue from operations in FY2025-26?", use_filters=True, per_entity=True)
    assert r.candidates and all(c.metadata["company"] == "Acme" and c.metadata["fiscal_year"] == "FY2025-26" for c in r.candidates)
    assert r.candidates[0].chunk_id == "acme-fy2025-cons-rev"
    assert r.filter_report.applied.keys() == {"company", "fiscal_year", "scope"}


def test_comparative_query_covers_each_company(fixture_pipe):
    r = run(fixture_pipe, "Compare consolidated revenue from operations of Acme and Borealis in FY2025-26",
            use_filters=True, per_entity=True, final_k=6)
    assert r.coverage == {"Acme": True, "Borealis": True}
    ids = [c.chunk_id for c in r.candidates]
    assert "acme-fy2025-cons-rev" in ids and "borealis-fy2025-cons-rev" in ids


def test_filter_not_applied_when_value_not_in_corpus_and_reported(fixture_pipe):
    r = run(fixture_pipe, "Acme revenue from operations in FY2019-20", use_filters=True)
    assert "fiscal_year" in r.filter_report.unavailable
    assert any("not applied" in x for x in r.limitations)
    assert r.candidates and all(c.metadata["company"] == "Acme" for c in r.candidates)  # company filter still applied


def test_no_filters_when_user_gave_none(fixture_pipe):
    r = run(fixture_pipe, "What is the dividend per share?", use_filters=True)
    assert r.filter_report.applied == {} and len({c.metadata["company"] for c in r.candidates}) > 1


def test_dense_only_bm25_only_and_hybrid_counts(fixture_pipe):
    q = "Acme total income consolidated FY2025-26"
    d = run(fixture_pipe, q, use_bm25=False)
    b = run(fixture_pipe, q, use_dense=False)
    h = run(fixture_pipe, q)
    assert d.counts["bm25_hits"] == 0 and b.counts["dense_hits"] == 0
    assert h.counts["union_before_dedupe"] == h.counts["dense_hits"] + h.counts["bm25_hits"]
    assert h.counts["after_dedupe"] <= h.counts["union_before_dedupe"]
    assert all(c.fusion_score is not None for c in h.candidates) and all(c.fusion_score is None for c in d.candidates)
    assert set(h.timings_ms) >= {"analysis", "dense", "bm25", "fusion", "rerank", "total"}


def test_rerank_pool_limited_and_failure_falls_back(fixture_pipe):
    r = run(fixture_pipe, "Acme revenue from operations", reranker=LexicalFeatureReranker(), rerank_pool=7)
    assert r.counts["reranked_pool"] == 7 and r.reranker_status.startswith("ok")
    boom = CrossEncoderReranker(scorer=lambda p: (_ for _ in ()).throw(RuntimeError("model crashed")))
    base = run(fixture_pipe, "Acme revenue from operations")
    f = run(fixture_pipe, "Acme revenue from operations", reranker=boom)
    assert f.reranker_status.startswith("failed") and any("reranker failed" in x for x in f.limitations)
    assert [c.chunk_id for c in f.candidates] == [c.chunk_id for c in base.candidates]


def test_evidence_includes_full_table_for_numeric_hit(fixture_pipe):
    r = run(fixture_pipe, "What was Acme's consolidated revenue from operations in FY2025-26?", use_filters=True)
    e = r.evidence[0]
    assert e.table_context and "Total income" in e.table_context and "Profit for the year" in e.table_context
    assert e.pdf_page_index is not None and e.doc_id == "acme_FY2025-26_ar" and e.unit == "INR crore"


def test_dense_missing_is_reported():
    from proofirst_retrieval.bm25 import BM25Retriever
    from proofirst_retrieval.pipeline import RetrievalPipeline
    ch = [mk("a", "revenue", company="X")]
    r = RetrievalPipeline(BM25Retriever(ch), None, ch).run("revenue", PipelineConfig())
    assert any("dense retriever not configured" in x for x in r.limitations) and r.candidates


def test_failing_reranker_leaks_no_partial_scores(fixture_pipe):
    class Leaky:
        name = "leaky"
        def rerank(self, analysis, cands):
            for c in cands:
                c.rerank_score = 9.0
            raise RuntimeError("boom")
    r = run(fixture_pipe, "Acme revenue from operations", reranker=Leaky())
    assert r.reranker_status.startswith("failed") and all(c.rerank_score is None for c in r.candidates)


def test_company_quota_uses_ceiling():
    c = [mk(f"a{i}", company="A") for i in range(10)] + [mk(f"b{i}", company="B") for i in range(10)]
    top = select_with_company_quota(c, 10, ["A", "B"])
    assert sum(x.metadata["company"] == "B" for x in top) >= 3     # ceil(10/4)=3, not floor=2
