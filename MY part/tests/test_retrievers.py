import pytest
from conftest import mk
from proofirst_retrieval.bm25 import BM25Retriever
from proofirst_retrieval.contracts import FieldMap, FilterSpec
from proofirst_retrieval.dense import ChromaDenseRetriever, HashingDenseRetriever, load_corpus_from_chroma
from proofirst_retrieval.query import QueryProcessor

CORPUS = [
    mk("rev_a", "Revenue from operations | 100 | 90", company="Acme", table_title="Consolidated P&L", row_label="Revenue from operations"),
    mk("tot_a", "Total income | 110 | 99", company="Acme", table_title="Consolidated P&L", row_label="Total income"),
    mk("rev_b", "Revenue from operations | 200 | 180", company="Borealis", table_title="Consolidated P&L", row_label="Revenue from operations"),
    mk("nar_a", "Acme expanded its workforce and cloud practice", company="Acme", content_type="narrative"),
    mk("nometa", "Revenue from operations summary text"),
]


def test_bm25_contract_and_one_based_ranks():
    r = BM25Retriever(CORPUS).search("revenue from operations", top_k=3)
    assert [c.ranks["bm25"] for c in r] == [1, 2, 3]
    assert all(c.scores["bm25"] > 0 for c in r) and r[0].scores["bm25"] >= r[1].scores["bm25"]
    top = r[0]
    assert top.chunk_id in {"rev_a", "rev_b"} and top.text and top.metadata.get("company") in {"Acme", "Borealis"}


def test_bm25_distinguishes_revenue_from_total_income():
    ids = [c.chunk_id for c in BM25Retriever(CORPUS).search("total income", top_k=5)]
    assert ids[0] == "tot_a"
    ids = [c.chunk_id for c in BM25Retriever(CORPUS).search("revenue from operations", top_k=5)]
    assert ids.index("tot_a") > ids.index("rev_a") if "tot_a" in ids else True


def test_bm25_hard_filter_and_missing_metadata():
    r = BM25Retriever(CORPUS)
    got = [c.chunk_id for c in r.search("revenue from operations", 10, FilterSpec({"company": ("Borealis",)}))]
    assert got == ["rev_b"]                       # 'nometa' (no company) is excluded, not leaked
    assert r.search("revenue", 10, FilterSpec({"company": ("Nobody",)})) == []


def test_bm25_alias_expansion_is_opt_in():
    a = QueryProcessor().analyze("net sales")
    assert a.metrics == ["revenue_from_operations"]
    assert BM25Retriever(CORPUS).search(a, 5) == []                      # plain BM25: no lexical overlap
    aliased = BM25Retriever(CORPUS, use_alias=True).search(a, 5)
    assert {c.chunk_id for c in aliased[:2]} <= {"rev_a", "rev_b", "nometa"} and aliased
    assert "tot_a" not in [c.chunk_id for c in aliased[:3]]               # expansion does not drag in total income
    assert BM25Retriever(CORPUS).query_terms(a) != BM25Retriever(CORPUS, use_alias=True).query_terms(a)


def test_bm25_empty_and_unknown_queries():
    r = BM25Retriever(CORPUS)
    assert r.search("", 5) == [] and r.search("zzzzqqq", 5) == []
    assert BM25Retriever([]).search("anything", 5) == []


def test_bm25_deterministic_ties():
    c = [mk("a", "same words"), mk("b", "same words")]
    assert [x.chunk_id for x in BM25Retriever(c).search("same words", 5)] == ["a", "b"]


def test_bm25_corpus_values():
    v = BM25Retriever(CORPUS).corpus_values(["company", "scope"])
    assert v["company"] == {"acme", "borealis"} and v["scope"] == set()


# ---- Chroma adapter contract with a fake collection (documented chromadb return shape) ----
class FakeCollection:
    def __init__(self):
        self.calls = []
        self.store = [(f"id{i}", f"doc{i}", {"co_name": "Acme", "page": i}) for i in range(5)]

    def query(self, **kw):
        self.calls.append(kw)
        n = kw["n_results"]
        rows = self.store[:n]
        return {"ids": [[r[0] for r in rows]], "documents": [[r[1] for r in rows]],
                "metadatas": [[r[2] for r in rows]], "distances": [[0.1 * (i + 1) for i in range(len(rows))]]}

    def get(self, include=None, limit=None, offset=0):
        rows = self.store[offset: offset + limit]
        return {"ids": [r[0] for r in rows], "documents": [r[1] for r in rows], "metadatas": [r[2] for r in rows]}


def test_chroma_adapter_contract():
    fm = FieldMap({"company": "co_name", "pdf_page_index": "page"})
    col = FakeCollection()
    d = ChromaDenseRetriever(col, lambda q: [0.1, 0.2], fm)
    res = d.search("q", top_k=3, filters=FilterSpec({"company": ("Acme", "Borealis")}))
    call = col.calls[0]
    assert call["n_results"] == 3 and call["query_embeddings"] == [[0.1, 0.2]]
    assert call["where"] == {"co_name": {"$in": ["Acme", "Borealis"]}}
    assert set(call["include"]) == {"documents", "metadatas", "distances"}
    assert [c.chunk_id for c in res] == ["id0", "id1", "id2"] and [c.ranks["dense"] for c in res] == [1, 2, 3]
    assert res[0].metadata == {"company": "Acme", "pdf_page_index": 0} and res[0].text == "doc0"
    assert res[0].scores["dense_distance"] == 0.1 and res[0].scores["dense"] == -0.1
    d.search("q", 2)
    assert "where" not in col.calls[1]            # no filter -> no where clause sent


def test_load_corpus_from_chroma_paginates():
    corpus = load_corpus_from_chroma(FakeCollection(), FieldMap({"company": "co_name"}), page=2)
    assert [c.chunk_id for c in corpus] == [f"id{i}" for i in range(5)]
    assert corpus[0].metadata["company"] == "Acme"


def test_hashing_standin_contract_and_filter():
    d = HashingDenseRetriever(CORPUS)
    assert d.stand_in is True
    r = d.search("revenue from operations Acme", 3)
    assert [c.ranks["dense"] for c in r] == [1, 2, 3] and r[0].scores["dense"] >= r[1].scores["dense"]
    only_b = d.search("revenue", 5, FilterSpec({"company": ("Borealis",)}))
    assert [c.chunk_id for c in only_b] == ["rev_b"]
