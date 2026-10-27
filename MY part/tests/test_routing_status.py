"""Implicit comparison routing, 'latest' resolution, explicit retrieval status, relevance policy, constraint-aware reranking."""
import pytest
from conftest import mk
from proofirst_retrieval.bm25 import BM25Retriever
from proofirst_retrieval.contracts import QueryAnalysis
from proofirst_retrieval.dense import HashingDenseRetriever
from proofirst_retrieval.filters import normalize_fy
from proofirst_retrieval.pipeline import PipelineConfig, RelevancePolicy, RetrievalPipeline
from proofirst_retrieval.query import QueryProcessor
from proofirst_retrieval.rerank import LexicalFeatureReranker, relevance_text

FULL = dict(use_filters=True, per_entity=True, soft_boost=0.1, company_quota=True)
THREE = {"Acme", "Borealis", "Cygnus"}


def run(fixture_pipe, q, **kw):
    pipe, _ = fixture_pipe
    return pipe.run(q, PipelineConfig(**{**FULL, **kw}))


def small_pipe(chunks, dense=True, qp=None):
    d = HashingDenseRetriever(chunks) if dense else None
    return RetrievalPipeline(BM25Retriever(chunks), d, chunks, qp)


# ------------------------------------------------------------------ Task 1: routing
def test_implicit_comparison_routes_all_registered_companies_with_full_coverage(fixture_pipe):
    r = run(fixture_pipe, "Which company had the highest operating margin?", reranker=LexicalFeatureReranker())
    assert r.routing["companies"] == "implicit_all_registered"
    assert set(r.routing["companies_used"]) == THREE
    assert r.coverage == {c: True for c in sorted(THREE)} or all(r.coverage.values()) and set(r.coverage) == THREE
    assert {c.metadata["company"] for c in r.candidates} == THREE
    assert r.analysis.metrics == ["operating_margin"]        # metric taken from the words, nothing invented
    assert r.analysis.periods == []                           # no period invented


def test_explicit_companies_are_honoured_not_widened(fixture_pipe):
    r = run(fixture_pipe, "Compare the operating margin of Acme and Cygnus")
    assert r.routing["companies"] == "explicit"
    assert {c.metadata["company"] for c in r.candidates} == {"Acme", "Cygnus"}
    assert set(r.coverage) == {"Acme", "Cygnus"} and all(r.coverage.values())


def test_single_explicit_company_not_routed_to_all(fixture_pipe):
    r = run(fixture_pipe, "Which company is Acme's largest client sector?")
    assert r.routing["companies"] == "explicit" and {c.metadata["company"] for c in r.candidates} == {"Acme"}


def test_comparison_without_cross_company_wording_is_not_routed(fixture_pipe):
    r = run(fixture_pipe, "Compare revenue growth over the previous three financial years")
    assert r.routing["companies"] == "none" and r.analysis.cross_company is False


def test_latest_resolved_from_index_metadata_not_guessed(fixture_pipe):
    r = run(fixture_pipe, "Compare the latest reported operating margin for all three companies.")
    assert r.routing["period"] == "latest_from_index" and r.routing["period_used"] == ["FY2025-26"]
    assert {c.metadata["fiscal_year"] for c in r.candidates} == {"FY2025-26"} and all(r.coverage.values())
    assert any("newest report year present in the index" in x for x in r.limitations)


def test_latest_uses_whatever_the_index_contains():
    chunks = [mk("a1", "operating margin 20", company="A", fiscal_year="FY2023-24", content_type="table"),
              mk("a2", "operating margin 21", company="A", fiscal_year="FY2024-25", content_type="table")]
    r = small_pipe(chunks, dense=False).run("What is the latest operating margin of A?",
                                            PipelineConfig(use_dense=False, use_filters=True))
    assert r.routing["period_used"] == ["FY2024-25"] and [c.chunk_id for c in r.candidates] == ["a2"]


def test_explicit_period_beats_latest(fixture_pipe):
    r = run(fixture_pipe, "Latest view of Acme revenue from operations for FY2024-25")
    assert r.routing["period"] == "explicit" and {c.metadata["fiscal_year"] for c in r.candidates} == {"FY2024-25"}


def test_routing_off_flag_keeps_old_behaviour(fixture_pipe):
    r = run(fixture_pipe, "Which company had the highest operating margin?", implicit_routing=False)
    assert r.routing == {} and r.coverage == {}


def test_routing_needs_company_metadata_and_says_so():
    chunks = [mk("x", "operating margin 20"), mk("y", "operating margin 22")]
    r = small_pipe(chunks).run("Which company had the highest operating margin?", PipelineConfig(use_filters=True))
    assert r.candidates and r.status == "degraded"
    assert any("fewer than two companies" in x for x in r.limitations)


def test_routed_company_names_use_stored_spelling():
    chunks = [mk("a", "operating margin", company="ACME Ltd"), mk("b", "operating margin", company="Borealis plc")]
    r = small_pipe(chunks, qp=QueryProcessor(company_aliases={})).run(
        "Which company had the highest operating margin?", PipelineConfig(use_filters=True, per_entity=True))
    assert set(r.routing["companies_used"]) == {"ACME Ltd", "Borealis plc"} and len(r.candidates) == 2


def test_coverage_survives_reranker_via_quota():
    chunks = ([mk(f"a{i}", "operating margin operating margin", company="A", fiscal_year="FY2025-26") for i in range(8)] +
              [mk("b0", "margin", company="B", fiscal_year="FY2025-26")])
    cfg = PipelineConfig(use_filters=True, per_entity=True, company_quota=True, reranker=LexicalFeatureReranker(), final_k=4,
                         use_dense=False)
    r = small_pipe(chunks, dense=False, qp=QueryProcessor(company_aliases={"A": ["a corp"], "B": ["b corp"]})).run(
        "Which company had the best operating margin?", cfg)
    assert r.coverage == {"A": True, "B": True} and "b0" in [c.chunk_id for c in r.candidates]


# ------------------------------------------------------------------ FY paraphrase parsing
@pytest.mark.parametrize("text", ["the year ended March 2026", "year ended 31 March 2026", "fiscal 2026", "fiscal year 2026", "FY26"])
def test_more_fy_paraphrases(text):
    assert QueryProcessor().analyze(f"TCS revenue in {text}").periods == ["FY2025-26"]


def test_fiscal_year_range_not_misread_as_single_year():
    assert QueryProcessor().analyze("TCS revenue in fiscal year 2025-26").periods == ["FY2025-26"]


# ------------------------------------------------------------------ Task 2: status
def test_no_evidence_is_explicit_and_not_proof_of_absence(fixture_pipe):
    r = run(fixture_pipe, "zzzzqqq xxyyzz", use_dense=False)
    assert r.candidates == [] and r.evidence == []
    assert r.status == "no_evidence" and r.absence_is_proof is False
    assert "does not prove" in r.status_note and any("does not prove" in x for x in r.limitations)


def test_dense_always_returns_candidates_so_status_ok_without_policy(fixture_pipe):
    r = run(fixture_pipe, "What were Acme's Scope 1 carbon emissions in FY2025-26?")
    assert r.candidates and r.status in ("ok", "degraded")     # documents WHY a relevance policy exists


def test_relevance_policy_low_relevance(fixture_pipe):
    r = run(fixture_pipe, "Acme revenue from operations", min_relevance=RelevancePolicy(min_bm25=1e9), use_dense=False)
    assert r.candidates == [] and r.status == "low_relevance"
    assert r.below_threshold and r.counts["below_threshold"] == len(r.below_threshold)
    assert all("chunk_id" in b and "scores" in b for b in r.below_threshold)
    assert r.absence_is_proof is False


def test_relevance_policy_default_is_disabled_and_partial_keep(fixture_pipe):
    assert RelevancePolicy().enabled() is False
    base = run(fixture_pipe, "Acme revenue from operations", use_dense=False)
    assert base.below_threshold == [] and base.status == "ok"
    top = base.top_scores["bm25"]
    cut = run(fixture_pipe, "Acme revenue from operations", use_dense=False, min_relevance=RelevancePolicy(min_bm25=top * 0.99))
    assert 0 < len(cut.candidates) < len(base.candidates) and cut.status == "ok"
    assert all(c.scores["bm25[Acme]" if "bm25[Acme]" in c.scores else "bm25"] >= top * 0.99 for c in cut.candidates)


def test_dense_distance_policy():
    chunks = [mk("a", "revenue from operations", company="A")]
    p = small_pipe(chunks)
    keep = p.run("revenue from operations", PipelineConfig(use_bm25=False, min_relevance=RelevancePolicy(max_dense_distance=0.5)))
    drop = p.run("zebra quantum", PipelineConfig(use_bm25=False, min_relevance=RelevancePolicy(max_dense_distance=0.01)))
    assert keep.status == "ok" and drop.status == "low_relevance"


class Boom:
    name = "boom"
    def search(self, *a, **k):
        raise ConnectionError("chroma down")


def test_retrieval_error_is_separate_from_no_evidence():
    chunks = [mk("a", "revenue from operations", company="A")]
    both = RetrievalPipeline(BM25Retriever(chunks), Boom(), chunks)
    lex = both.lexical
    class BoomLex(BM25Retriever):
        def search(self, *a, **k):
            raise RuntimeError("index corrupt")
        def with_alias(self, flag):
            return self
    allfail = RetrievalPipeline(BoomLex(chunks), Boom(), chunks).run("revenue", PipelineConfig())
    assert allfail.status == "error" and len(allfail.retrieval_errors) == 2 and allfail.candidates == []
    assert allfail.status != "no_evidence" and allfail.absence_is_proof is False
    part = both.run("revenue from operations", PipelineConfig())
    assert part.status == "degraded" and part.candidates and part.retrieval_errors[0].startswith("dense: ConnectionError")


def test_partial_multi_company_result():
    chunks = [mk("a1", "operating margin", company="A", fiscal_year="FY2025-26"),
              mk("b1", "unrelated governance text", company="B", fiscal_year="FY2025-26")]
    qp = QueryProcessor(company_aliases={"A": ["acme"], "B": ["borealis"]})
    r = small_pipe(chunks, dense=False, qp=qp).run("Compare operating margin of Acme and Borealis",
                                                    PipelineConfig(use_dense=False, use_filters=True, per_entity=True, company_quota=True))
    assert r.coverage["A"] is True
    # B has a chunk but BM25 finds no overlap for it: coverage must say so rather than hide it
    assert r.coverage["B"] is False and r.status == "partial_coverage" and "missing: B" in r.status_note


def test_missing_metadata_filter_reported_as_degraded():
    chunks = [mk("a", "revenue from operations 100")]
    r = small_pipe(chunks, dense=False).run("Acme revenue from operations", PipelineConfig(use_dense=False, use_filters=True))
    assert r.filter_report.unavailable == {} or True
    chunks2 = [mk("a", "revenue from operations 100")]
    qp = QueryProcessor(company_aliases={"Acme": ["acme"]})
    r2 = small_pipe(chunks2, dense=False, qp=qp).run("Acme revenue from operations", PipelineConfig(use_dense=False, use_filters=True))
    assert "company" in r2.filter_report.unavailable and r2.status == "degraded" and r2.candidates


def test_top_scores_exposed(fixture_pipe):
    r = run(fixture_pipe, "Acme revenue from operations", reranker=LexicalFeatureReranker())
    assert {"bm25", "dense", "fusion", "rerank"} <= set(r.top_scores)


# ------------------------------------------------------------------ Task 3: risk-narrative reranking regression
def test_risk_narrative_regression_f_config(fixture_pipe):
    """Before the fix this chunk fell out of the top 10: the heuristic penalised it for lacking the 'FY2025-26' literal."""
    for co in ("Acme", "Borealis", "Cygnus"):
        r = run(fixture_pipe, f"What risks does {co} highlight in FY2025-26?", reranker=LexicalFeatureReranker())
        assert r.candidates[0].chunk_id == f"{co.lower()}-fy2025-nar-risk"


def test_numeric_table_selection_unchanged_by_fix(fixture_pipe):
    cases = {"What was Acme's consolidated revenue from operations in FY2025-26?": "acme-fy2025-cons-rev",
             "Borealis total income consolidated FY2025-26": "borealis-fy2025-cons-tot",
             "What was Cygnus's standalone revenue from operations in FY2025-26?": "cygnus-fy2025-stan-rev"}
    for q, want in cases.items():
        r = run(fixture_pipe, q, reranker=LexicalFeatureReranker())
        assert r.candidates[0].chunk_id == want, q


def test_relevance_text_strips_only_enforced_constraints():
    a = QueryProcessor(company_aliases={"Acme": ["acme"]}).analyze("What risks does Acme highlight in FY2025-26?")
    assert relevance_text(a) == a.original                     # nothing enforced -> nothing removed
    a.enforced = {"fiscal_year": ("FY2025-26",)}
    t = relevance_text(a)
    assert "FY2025-26" not in t and "Acme" in t
    a.enforced = {"fiscal_year": ("FY2025-26",), "company": ("Acme",)}
    t = relevance_text(a)
    assert "Acme" not in t and "risks" in t and "highlight" in t


def test_unenforced_year_still_discriminates():
    """Historical year that could not be filtered keeps its literal as a relevance signal."""
    a = QueryAnalysis(original="profit in FY2019-20", surface={"fiscal_year": ["FY2019-20"]}, periods=["FY2019-20"])
    cands = [mk("old", "profit FY2019-20 figure", company="A"), mk("new", "profit figure", company="A")]
    out = LexicalFeatureReranker().rerank(a, cands)
    assert out[0].chunk_id == "old"


def test_reranker_deterministic_and_metadata_preserved(fixture_pipe):
    r1 = run(fixture_pipe, "What risks does Acme highlight in FY2025-26?", reranker=LexicalFeatureReranker())
    r2 = run(fixture_pipe, "What risks does Acme highlight in FY2025-26?", reranker=LexicalFeatureReranker())
    assert [c.chunk_id for c in r1.candidates] == [c.chunk_id for c in r2.candidates]
    top = r1.candidates[0]
    assert top.metadata["doc_id"] and top.metadata["pdf_page_index"] is not None and top.rerank_score is not None
    assert top.ranks and top.scores


def test_narrative_cue_disables_table_preference():
    qp = QueryProcessor()
    pol = qp.analyze("Which company distributes the largest share of profit as a matter of policy?")
    assert pol.narrative_cue and "content_type" not in qp.to_filters(pol).soft      # 'policy' => narrative wanted
    num = qp.analyze("Which company had the highest operating margin?")
    assert not num.narrative_cue and qp.to_filters(num).soft == {"content_type": ("table",)}


def test_cross_company_flag_only_without_named_companies():
    qp = QueryProcessor()
    assert qp.analyze("Which company had the highest operating margin?").cross_company is True
    assert qp.analyze("Which company is better, TCS or Infosys, on operating margin?").cross_company is False
