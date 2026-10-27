"""Sankalp's retrieval stages (MY part: BM25, RRF fusion, re-ranker) over Karthik's chunks (RAG/data/chunks), scored on the
team test set like eval/compare_retrieval.py (run its export and embed steps first):

    python eval/compare_stages.py .
"""
import json, sys
from pathlib import Path
ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "MY part")); sys.path.insert(0, str(ROOT / "eval")); sys.path.insert(0, str(ROOT))
from proofirst_retrieval.bm25 import BM25Retriever
from proofirst_retrieval.contracts import Candidate
from proofirst_retrieval.fusion import rrf_fuse
from proofirst_retrieval.rerank import LexicalFeatureReranker
from proofirst_retrieval.query import QueryProcessor
from compare_retrieval import OUT, TOP_K, pages_of, value_in

docs = {d["id"]: d for d in map(json.loads, (OUT / "docs.jsonl").open(encoding="utf-8"))}
queries = json.loads((OUT / "queries.json").read_text(encoding="utf-8"))
ranked = json.loads((OUT / "ranked.json").read_text())["results"]
cands = {cid: Candidate(chunk_id=cid, text=d["embed"], metadata={"company": d["company"], "pages": d["pages"]})
         for cid, d in docs.items()}
bm25 = {co: BM25Retriever([c for c in cands.values() if c.metadata["company"] == co]) for co in ("TCS", "Infosys", "HCLTech")}
qp, reranker = QueryProcessor(), LexicalFeatureReranker()
stages = {"BM25 (Sankalp)": [0, 0], "vectors (Karthik)": [0, 0], "RRF fusion": [0, 0], "fusion + rerank": [0, 0]}
rerank_error = None
for r in ranked:
    q, co = queries[r["q"]], r["company"]
    page, value = q["pages"].get(co), q["values"].get(co)
    analysis = qp.analyze(q["query"])
    lex = bm25[co].search(analysis, top_k=10)
    vec = [cands[t["id"]].copy() for t in r["top"]]
    fused = rrf_fuse([lex, vec], names=["bm25", "dense"])
    try:
        rer = reranker.rerank(analysis, fused[:20])
    except Exception as exc:
        rerank_error, rer = repr(exc), fused
    for name, lst in (("BM25 (Sankalp)", lex), ("vectors (Karthik)", vec), ("RRF fusion", fused), ("fusion + rerank", rer)):
        ids = [c.chunk_id for c in lst[:TOP_K]]
        stages[name][0] += any(page in pages_of(docs[i]["pages"]) for i in ids)
        stages[name][1] += value_in([docs[i]["text"] for i in ids], value)
n = len(ranked)
print(f"{n} (question, company) pairs, top {TOP_K} chunks; our keyword retrieval over facts.db: page 34/{n}, value 33/{n}")
for name, (p, v) in stages.items():
    print(f"  {name:18s}: page {p}/{n}, value {v}/{n}")
if rerank_error:
    print("  rerank failed:", rerank_error)
