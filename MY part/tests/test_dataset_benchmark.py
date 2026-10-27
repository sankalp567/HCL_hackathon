import csv, json
from pathlib import Path
import pytest
from proofirst_retrieval.dataset import load_corpus, load_queries, validate_query
from proofirst_retrieval import benchmark

ROOT = Path(__file__).resolve().parent.parent


def test_fixture_loads_and_labels_reference_real_chunks():
    corpus = load_corpus(ROOT / "data" / "synthetic_corpus.jsonl")
    qs = load_queries(ROOT / "data" / "synthetic_queries.jsonl", {c.chunk_id for c in corpus})
    assert len(qs) >= 50 and {q["provenance"] for q in qs} == {"synthetic"}
    assert {"unanswerable", "multi_company", "scope", "paraphrase", "historical", "qualitative"} <= {q["category"] for q in qs}


def test_validation_rules():
    assert validate_query({"qid": "x", "question": "q", "category": "numeric_lookup", "provenance": "manual:me", "qrels": {"a": 3}}) == []
    bad = validate_query({"qid": "x", "question": "q", "category": "nope", "provenance": "p", "qrels": {"a": 7}})
    assert any("category" in e for e in bad) and any("grade" in e for e in bad)
    assert validate_query({"qid": "x", "question": "q", "category": "unanswerable", "provenance": "p", "qrels": {"a": 1}})
    assert validate_query({"qid": "x", "question": "q", "category": "scope", "provenance": "p", "qrels": {}})
    assert any("not in corpus" in e for e in validate_query(
        {"qid": "x", "question": "q", "category": "scope", "provenance": "p", "qrels": {"zz": 1}}, {"a"}))


def test_full_benchmark_writes_artifacts(tmp_path):
    out = benchmark.run("synthetic", ROOT / "data" / "synthetic_corpus.jsonl", ROOT / "data" / "synthetic_queries.jsonl",
                        tmp_path / "art", tmp_path / "docs", repeats=1)
    ids = {e["id"] for e in out["experiments"]}
    assert {"A", "B", "C", "D", "E", "F"} <= ids and {"D_ce", "F_ce"} <= {b["id"] for b in out["blocked"]}
    data = json.loads((tmp_path / "art" / "retrieval_benchmark_results.json").read_text())
    assert data["meta"]["label"] == "synthetic" and data["meta"]["real_data_benchmark"] == "BLOCKED"
    rows = list(csv.DictReader((tmp_path / "art" / "retrieval_benchmark_results.csv").open()))
    assert len(rows) == len(out["experiments"]) and {"recall@10", "mrr", "ndcg@10", "p50_total_ms"} <= set(rows[0])
    md = (tmp_path / "docs" / "RETRIEVAL_EVALUATION.md").read_text()
    assert "DATA LABEL: SYNTHETIC" in md and "BLOCKED" in md and "## 12." in md
    assert "5.6 Template-macro" in md and "5.7 Retrieval status" in md and "distinct templates" in md
    d = data["meta"]["dataset"]
    assert d["n_templates"] >= 60 and d["n_handwritten"] >= 30
    assert d["answerable_queries_stating"]["cross_company_wording"] > 0 and d["answerable_queries_stating"]["latest_wording"] > 0
    assert all("template_macro" in e and "status_counts" in e for e in out["experiments"])
    # metrics are in range; unanswerable queries excluded from rank metrics
    for e in out["experiments"]:
        assert e["n_unanswerable_excluded"] == data["meta"]["dataset"]["n_unanswerable"] > 4
        for k in ("recall@10", "mrr", "ndcg@10"):
            assert 0.0 <= e["metrics"][k] <= 1.0


def test_real_mode_reports_blocked_not_fabricated(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "chromadb", None)   # simulate missing dependency
    import argparse
    ns = argparse.Namespace(chroma_path="x", collection="y", embedder="a:b", field_map=None)
    with pytest.raises(benchmark.Blocked):
        benchmark.build_real(ns)


def test_v2_fixture_has_independent_templates_and_unnamed_cases():
    qs = load_queries(ROOT / "data" / "synthetic_queries.jsonl")
    hand = [q for q in qs if str(q.get("template", "")).startswith("H")]
    assert len(hand) >= 30 and len({q["template"] for q in hand}) == len(hand)   # one template per hand-written query
    assert all("template" in q for q in qs)
