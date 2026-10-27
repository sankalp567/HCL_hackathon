# MY part — Retrieval & Reranking Engine (Member 4)

Lexical (BM25) + dense adapter + RRF fusion + metadata filters + reranker + evidence assembly + evaluation harness.
Pure Python + numpy. No new services, no second vector DB, no document embeddings created here.

## Repo audit (what exists vs. what is assumed) — 2026-10-27
| Question | Finding |
|---|---|
| ChromaDB collection / Qwen3-Embedding-0.6B / Docling code in repo | **None.** grep for chroma/qwen3/bm25/rerank/docling hits only docs. Member 2's pipeline is not in the repo yet. |
| Packages installed (`.venv`) | numpy 2.5.4, pytest 9.1.1. **Missing:** chromadb, torch, transformers, sentence-transformers, rank_bm25. SQLite FTS5 is available (unused; own BM25 chosen so dense + lexical share one chunk list). |
| Existing retrieval code | Only the starter kit's `backend/app/{chunking,embeddings,evidence,ask}.py` (hash embeddings + numpy cosine over SQLite). Not reused: different chunk schema, would be a duplicate index. |
| Chunk metadata keys, collection name, query API | **Unknown / assumed.** Canonical keys are defined in `contracts.py` (`FieldMap` maps Member 2's real keys onto them). |
| Test framework | pytest (kit's backend uses pytest too). Hardware: Apple M5, 16 GB, no GPU used. |

## Layout
`proofirst_retrieval/` — `contracts` (Candidate, QueryAnalysis, FilterSpec, EvidenceRecord, FieldMap), `query` (rule-based analysis, alias registry),
`filters` (hard/soft, Chroma `where` builder, availability report), `bm25`, `dense` (`ChromaDenseRetriever`, `load_corpus_from_chroma`, hashing stand-in),
`fusion` (RRF k=60, 1-based ranks), `rerank` (none / lexical-heuristic / cross-encoder wrapper), `evidence` (table context, company quota),
`pipeline`, `metrics`, `dataset`, `benchmark`, `report`. `data/` synthetic fixture + generator. `tests/` 129 tests. `artifacts/` and `docs/RETRIEVAL_EVALUATION.md` are generated.

## Run
```bash
cd "MY part"
../.venv/bin/python -m pytest                                   # unit tests
../.venv/bin/python -m proofirst_retrieval.benchmark --label synthetic   # regenerates artifacts/ and docs/RETRIEVAL_EVALUATION.md
```
Real index (once Member 2/3 inputs exist, see section 12 of the report):
`python -m proofirst_retrieval.benchmark --label real --queries real_queries.jsonl --chroma-path PATH --collection NAME --embedder pkg.mod:embed_query [--field-map map.json]`

## Using it in the app
```python
chunks = load_corpus_from_chroma(collection, field_map)
pipe = RetrievalPipeline(BM25Retriever(chunks), ChromaDenseRetriever(collection, embed_query, field_map), chunks)
res = pipe.run(question, PipelineConfig(use_filters=True, per_entity=True, alias_expansion=True, final_k=8))
res.evidence   # EvidenceRecord list: company, doc_id, pdf_page_index, printed_page, chunk_id, source_text, table_context, scores...
res.limitations  # filters not applied, missing company evidence, reranker failures: surface these to the chatbot/refusal logic
```
Scores are relevance only; they never validate a number.
