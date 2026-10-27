import pytest
from conftest import mk
from proofirst_retrieval.contracts import FieldMap, FilterSpec
from proofirst_retrieval.filters import matches, normalize_fy, to_chroma_where, resolve_filters
from proofirst_retrieval.query import QueryProcessor


def test_normalize_fy():
    assert normalize_fy("FY2025-26") == "FY2025-26"
    assert normalize_fy("fy 2025-2026") == "FY2025-26"
    assert normalize_fy("2025-26") == "FY2025-26"
    assert normalize_fy("FY26") == "FY2025-26"
    assert normalize_fy("2025-27") is None and normalize_fy("hello") is None


def test_matches_hard_and_missing_metadata():
    assert matches({"company": "Acme", "scope": "consolidated"}, {"company": ("acme",)})
    assert not matches({"company": "Borealis"}, {"company": ("Acme",)})
    assert not matches({}, {"company": ("Acme",)})            # missing metadata cannot be confirmed
    assert not matches({"company": ""}, {"company": ("Acme",)})
    assert matches({"fiscal_year": "2025-26"}, {"fiscal_year": ("FY2025-26",)})


def test_chroma_where_construction():
    assert to_chroma_where({}) is None
    assert to_chroma_where({"company": ("Acme",)}) == {"company": {"$eq": "Acme"}}
    w = to_chroma_where({"company": ("A", "B"), "fiscal_year": ("FY2025-26",)})
    assert w == {"$and": [{"company": {"$in": ["A", "B"]}}, {"fiscal_year": {"$eq": "FY2025-26"}}]}
    fm = FieldMap({"company": "co_name"})
    assert to_chroma_where({"company": ("A",)}, fm) == {"co_name": {"$eq": "A"}}


def test_field_map_to_canonical():
    fm = FieldMap({"company": "co_name", "pdf_page_index": "page"})
    assert fm.to_canonical({"co_name": "Acme", "page": 4, "other": 1}) == {"company": "Acme", "pdf_page_index": 4, "other": 1}


def test_resolve_filters_reports_unavailable():
    spec = FilterSpec({"company": ("Acme",), "scope": ("consolidated",), "fiscal_year": ("FY2023-24",)})
    corpus = {"company": {"acme", "borealis"}, "scope": set(), "fiscal_year": {"fy2025-26"}}
    kept, rep = resolve_filters(spec, corpus)
    assert set(kept.hard) == {"company"}
    assert set(rep.unavailable) == {"scope", "fiscal_year"}
    assert len(rep.limitations) == 2 and "absent" in rep.limitations[0]


QP = QueryProcessor(company_aliases={"Acme": ["acme"], "Borealis": ["borealis"]})


def test_query_extracts_only_explicit_attributes():
    a = QP.analyze("What was the revenue from operations?")
    assert a.companies == [] and a.periods == [] and a.scope is None
    assert QP.to_filters(a).hard == {}


def test_query_company_period_scope_metric():
    a = QP.analyze("What was Acme's consolidated revenue from operations in FY2025-26?")
    assert a.companies == ["Acme"] and a.periods == ["FY2025-26"] and a.scope == "consolidated"
    assert a.metrics == ["revenue_from_operations"] and a.intent == "numeric_lookup"
    f = QP.to_filters(a)
    assert f.hard == {"company": ("Acme",), "fiscal_year": ("FY2025-26",), "scope": ("consolidated",)}
    assert f.soft == {"content_type": ("table",)}   # soft preference, not a hard constraint


def test_revenue_vs_total_income_are_distinct():
    a = QP.analyze("Acme total income FY2025-26")
    assert a.metrics == ["total_income"]
    assert "revenue from operations" not in a.expansion_terms
    b = QP.analyze("Acme revenue from operations")
    assert b.metrics == ["revenue_from_operations"] and "total income" not in b.expansion_terms
    both = QP.analyze("revenue from operations versus total income")
    assert set(both.metrics) == {"revenue_from_operations", "total_income"}


def test_generic_revenue_flagged_ambiguous():
    a = QP.analyze("Acme revenue")
    assert a.metrics == ["revenue_from_operations"] and a.ambiguous_terms == ["revenue"]


def test_relative_period_not_turned_into_filter():
    a = QP.analyze("How did Acme's attrition change compared with last year?")
    assert a.relative_period == "previous" and a.periods == []
    assert "fiscal_year" not in QP.to_filters(a).hard


def test_intents():
    assert QP.analyze("Compare revenue of Acme and Borealis").intent == "comparison"
    assert QP.analyze("What are Acme's key risks?").intent == "qualitative"
    assert QP.analyze("Acme revenue growth year-on-year").intent == "trend"
    assert QP.analyze("hello there").intent == "unknown"


def test_both_scopes_no_constraint():
    a = QP.analyze("Acme consolidated and standalone revenue")
    assert a.scope is None


@pytest.mark.parametrize("text,fy", [("FY25-26", "FY2025-26"), ("FY 24-25", "FY2024-25"), ("FY2026", "FY2025-26"),
                                     ("FY26", "FY2025-26"), ("2025-26", "FY2025-26"), ("fy2025-2026", "FY2025-26")])
def test_fy_variants(text, fy):
    assert normalize_fy(text) == fy
    assert QP.analyze(f"Acme revenue in {text}").periods == [fy]


def test_fy_bad_and_multiple():
    assert normalize_fy("FY25-27") is None
    assert QP.analyze("Acme revenue FY2025-26 versus FY24-25").periods == ["FY2025-26", "FY2024-25"]


@pytest.mark.parametrize("q,intent", [("What is Acme's growth strategy?", "qualitative"),
                                      ("What is the dividend policy of Acme?", "qualitative"),
                                      ("What was Acme revenue last year", "numeric_lookup"),
                                      ("Acme better hiring plans", "qualitative"),
                                      ("How did Acme's attrition change compared with the previous year?", "trend")])
def test_intent_fixes(q, intent):
    assert QP.analyze(q).intent == intent


def test_empty_hard_filter_consistent_local_and_chroma():
    assert matches({"company": "A"}, {"company": ()}) is True
    assert to_chroma_where({"company": ()}) is None


def test_chroma_where_normalizes_fiscal_year():
    assert to_chroma_where({"fiscal_year": ("fy 2025-26",)}) == {"fiscal_year": {"$eq": "FY2025-26"}}
