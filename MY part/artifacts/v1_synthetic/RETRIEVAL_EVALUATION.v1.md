# ProofFirst AI — Retrieval Evaluation (Member 4)

> **DATA LABEL: SYNTHETIC.** Every number below was measured on a small fixture of *fictional* companies with *invented* figures and labels derived by construction. It validates the pipeline mechanics and compares configurations on a synthetic task. It does **not** measure retrieval quality on the real TCS / Infosys / HCLTech annual reports. The dense retriever is a hashing **stand-in**, not Qwen3-Embedding-0.6B, and the reranker is a lexical-feature heuristic, not a cross-encoder. **Real-data benchmark: BLOCKED** (inputs listed in section 12).

Generated 2026-10-27T08:31:07+00:00 by `python -m proofirst_retrieval.benchmark --label synthetic` (code version 0.1.0). Machine-readable: `artifacts/retrieval_benchmark_results.json`, `.csv`, `retrieval_benchmark_per_query.csv`.

## 1. Experimental setup

- Hardware: Apple M5, 16.0 GB RAM, macOS-27.0.1-arm64-arm-64bit-Mach-O, Python 3.13.13, GPU used: False.
- Dense: HashingDenseRetriever (1024-d signed hashing of words+char-trigrams): STAND-IN, not Qwen3-Embedding-0.6B.
- Lexical: BM25Retriever (own implementation, k1=1.5 b=0.75, unigram+bigram, crude plural stemmer).
- Vector store: none (in-memory numpy).
- Fusion: RRF, k=60, ranks **1-based**, score = `sum_lists 1/(k+rank)`; in-list repeats count once; ties broken by fusion score, then best single-list rank, then first appearance.
- Latency: 5 timed repeats per query after one warm-up, wall-clock `perf_counter`, single process, no GPU. p50/p95 are reported only if the sample has at least 20 timings.

## 2. Corpus and test dataset

- Corpus: synthetic fixture (fictional companies, invented numbers); **132 chunks**. Provenance of labels: synthetic.
- Queries: 53 (49 answerable, 4 unanswerable and excluded from rank metrics). Categories: historical=6, multi_company=6, numeric_lookup=9, paraphrase=12, qualitative=6, scope=4, terminology=6, unanswerable=4.
- **Disclosed biases:** (1) every answerable query literally states the right company, fiscal year and scope, so the hard filters in E/F act as an oracle and their gain is an **upper bound**; (2) the 49 queries come from ~12 templates x 3 companies and one generator, with qrels derived from chunk ids, so they are not independent and the bootstrap CIs are optimistic (no multiplicity correction); (3) the alias registry in `query.py` contains the exact paraphrase terms used in the queries ('net sales', 'bottom line', 'workforce'), so alias gains are inflated; (4) the hashing 'dense' stand-in is word + char-trigram based, i.e. effectively lexical, so hybrid gains from A+B fusion are NOT evidence of semantic/lexical complementarity; (5) historical queries give grade 2 to the other-year chunk (comparative column), which a hard fiscal-year filter excludes, so E/F recall on that category is lowered by the label design, not by a retrieval defect; (6) E/F also apply a x1.1 soft boost for `content_type=table` on numeric intents.
- Further design bias: table-row chunks do not contain the company name (as in real annual-report tables), so unfiltered retrieval cannot tell companies apart on table rows; the controlled alias registry and the paraphrase queries were written by the same author, which favours alias expansion.

## 3. Tested configurations

| ID | Name | Components actually executed | cand/retriever | rerank pool | soft boost |
|---|---|---|---|---|---|
| A | Dense only [dense = hashing stand-in] | dense | 20 | - | - |
| B | BM25 only | bm25 | 20 | - | - |
| C | Dense + BM25 + RRF [dense = hashing stand-in] | dense, bm25, rrf | 20 | - | - |
| D | Hybrid + RRF + rerank (stand-in heuristic) [dense = hashing stand-in] | dense, bm25, rrf, rerank:lexical-feature-heuristic | 20 | 20 | - |
| E | Hybrid + RRF + metadata filters [dense = hashing stand-in] | dense, bm25, rrf, metadata_filters, per_entity_retrieval | 20 | - | x1.1 for content_type=table (soft pref.) |
| F | Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in] | dense, bm25, rrf, metadata_filters, per_entity_retrieval, rerank:lexical-feature-heuristic | 20 | 20 | x1.1 for content_type=table (soft pref.) |
| B+alias | BM25 only (alias-assisted) | bm25, alias_expansion | 20 | - | - |
| C+alias | Dense + BM25 + RRF [dense = hashing stand-in] | dense, bm25, rrf, alias_expansion | 20 | - | - |
| D+alias | Hybrid + RRF + rerank (stand-in heuristic) [dense = hashing stand-in] | dense, bm25, rrf, alias_expansion, rerank:lexical-feature-heuristic | 20 | 20 | - |
| E+alias | Hybrid + RRF + metadata filters [dense = hashing stand-in] | dense, bm25, rrf, metadata_filters, per_entity_retrieval, alias_expansion | 20 | - | x1.1 for content_type=table (soft pref.) |
| F+alias | Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in] | dense, bm25, rrf, metadata_filters, per_entity_retrieval, alias_expansion, rerank:lexical-feature-heuristic | 20 | 20 | x1.1 for content_type=table (soft pref.) |
| C@10 | Hybrid, 10 candidates/retriever [dense = hashing stand-in] | dense, bm25, rrf | 10 | - | - |
| E@10 | Hybrid + filters, 10 candidates/retriever [dense = hashing stand-in] | dense, bm25, rrf, metadata_filters, per_entity_retrieval | 10 | - | x1.1 for content_type=table (soft pref.) |
| C@30 | Hybrid, 30 candidates/retriever [dense = hashing stand-in] | dense, bm25, rrf | 30 | - | - |
| E@30 | Hybrid + filters, 30 candidates/retriever [dense = hashing stand-in] | dense, bm25, rrf, metadata_filters, per_entity_retrieval | 30 | - | x1.1 for content_type=table (soft pref.) |

Not run (recorded, not substituted):

| ID | Configuration | Reason |
|---|---|---|
| D_ce | Hybrid + RRF + cross-encoder (MiniLM) | cross-encoder not run: python package 'torch' not installed in this environment |
| F_ce | Hybrid + RRF + filters + cross-encoder (MiniLM) | cross-encoder not run: python package 'torch' not installed in this environment |
| D_qwen3rr | Hybrid + Qwen3-Reranker-0.6B | adapter not implemented and not attempted (torch/transformers not installed; no model downloaded) |
| real_chroma | All experiments on the real ChromaDB collection + manual qrels | collection, metadata schema and query embedder from Member 2 and manual labels from Member 3 are not available; real-mode code path exists but has never been executed against a real collection |
| A_instr | Dense with vs without Qwen3 query instruction | needs Member 2's real Qwen3-Embedding-0.6B query embedder; the hashing stand-in has no instruction mode |

## 4. Metric definitions

- Relevance: chunk-level graded qrels (1-3); grade>0 counts as relevant for binary metrics (the fixture only uses grades 2 and 3). Ranked list = final top-10 chunks.
- Recall@K = relevant in top K / all relevant. Precision@K = relevant in top K / K (denominator fixed at K). MRR = 1/rank of first relevant (0 if absent).
- nDCG@K: gain 2^grade-1, discount log2(position+1), ideal = the labelled grades sorted descending, truncated to K. Queries with no relevant chunk return *undefined* and are excluded (not scored 0 or 1).
- Numerical-evidence metrics use top-5: source-page hit (expected doc+PDF page present), company+year hit, metric/table hit (expected table id), row evidence hit (expected row chunk), company coverage (share of requested companies present; multi-company queries only), full coverage, scope match@1. `company_year_hit` is computed for single-company queries only; coverage metrics for multi-company queries only; `source_page_hit` accepts *any* expected page, so it is lenient and saturates easily. A metric is *n/a* if the labels it needs are absent. These measure **retrieval**, not whether a parsed number is financially correct.

## 5. Results

### 5.1 Main matrix (A-F; lexical queries without alias expansion)

| ID | Config | R@5 | R@10 | P@5 | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| A | Dense only [dense = hashing stand-in] | 0.765 | 0.867 | 0.233 | 0.546 | 0.551 | 0.580 |
| B | BM25 only | 0.765 | 0.837 | 0.233 | 0.588 | 0.564 | 0.589 |
| C | Dense + BM25 + RRF [dense = hashing stand-in] | 0.765 | 0.847 | 0.233 | 0.553 | 0.564 | 0.589 |
| D | Hybrid + RRF + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.903 | 0.969 | 0.265 | 0.784 | 0.759 | 0.780 |
| E | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.923 | 0.929 | 0.257 | 0.779 | 0.772 | 0.773 |
| F | Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.872 | 0.872 | 0.253 | 0.789 | 0.757 | 0.757 |

| ID | page hit | company+year hit | metric/table hit | row hit | company coverage | full coverage | scope@1 |
|---|---|---|---|---|---|---|---|
| A | 1.000 | 1.000 | 0.930 | 0.721 | 1.000 | 1.000 | 1.000 |
| B | 0.837 | 0.977 | 0.744 | 0.698 | 1.000 | 1.000 | 0.812 |
| C | 0.939 | 1.000 | 0.860 | 0.721 | 1.000 | 1.000 | 0.906 |
| D | 0.939 | 1.000 | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 |
| E | 1.000 | 1.000 | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 |
| F | 0.939 | 1.000 | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 |

### 5.2 Alias-assisted variants

| ID | Config | R@5 | R@10 | P@5 | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| B+alias | BM25 only (alias-assisted) | 0.883 | 0.959 | 0.253 | 0.621 | 0.658 | 0.683 |
| C+alias | Dense + BM25 + RRF [dense = hashing stand-in] | 0.888 | 0.929 | 0.257 | 0.620 | 0.641 | 0.654 |
| D+alias | Hybrid + RRF + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.903 | 1.000 | 0.265 | 0.874 | 0.827 | 0.865 |
| E+alias | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.959 | 0.964 | 0.273 | 0.857 | 0.849 | 0.851 |
| F+alias | Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.903 | 0.908 | 0.265 | 0.827 | 0.804 | 0.806 |

| ID | page hit | company+year hit | metric/table hit | row hit | company coverage | full coverage | scope@1 |
|---|---|---|---|---|---|---|---|
| B+alias | 0.939 | 0.977 | 0.930 | 0.907 | 1.000 | 1.000 | 0.906 |
| C+alias | 0.980 | 1.000 | 0.907 | 0.860 | 1.000 | 1.000 | 0.938 |
| D+alias | 0.939 | 1.000 | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 |
| E+alias | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F+alias | 0.939 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

### 5.3 Candidate-pool sweep (per retriever; C and E at 10/20/30)

| ID | Config | R@5 | R@10 | P@5 | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| C@10 | Hybrid, 10 candidates/retriever [dense = hashing stand-in] | 0.806 | 0.842 | 0.241 | 0.562 | 0.584 | 0.595 |
| C | Dense + BM25 + RRF [dense = hashing stand-in] | 0.765 | 0.847 | 0.233 | 0.553 | 0.564 | 0.589 |
| C@30 | Hybrid, 30 candidates/retriever [dense = hashing stand-in] | 0.765 | 0.847 | 0.233 | 0.553 | 0.564 | 0.589 |
| E@10 | Hybrid + filters, 10 candidates/retriever [dense = hashing stand-in] | 0.923 | 0.954 | 0.257 | 0.779 | 0.772 | 0.784 |
| E | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.923 | 0.929 | 0.257 | 0.779 | 0.772 | 0.773 |
| E@30 | Hybrid + filters, 30 candidates/retriever [dense = hashing stand-in] | 0.923 | 0.929 | 0.257 | 0.779 | 0.772 | 0.773 |

| ID | page hit | company+year hit | metric/table hit | row hit | company coverage | full coverage | scope@1 |
|---|---|---|---|---|---|---|---|
| C@10 | 0.959 | 1.000 | 0.884 | 0.767 | 1.000 | 1.000 | 0.906 |
| C | 0.939 | 1.000 | 0.860 | 0.721 | 1.000 | 1.000 | 0.906 |
| C@30 | 0.939 | 1.000 | 0.860 | 0.721 | 1.000 | 1.000 | 0.906 |
| E@10 | 1.000 | 1.000 | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 |
| E | 1.000 | 1.000 | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 |
| E@30 | 1.000 | 1.000 | 1.000 | 0.930 | 1.000 | 1.000 | 1.000 |

### 5.4 Per-category recall@10 / MRR (main matrix)

| ID | historical (n=6) | multi_company (n=6) | numeric_lookup (n=9) | paraphrase (n=12) | qualitative (n=6) | scope (n=4) | terminology (n=6) |
|---|---|---|---|---|---|---|---|
| A | 1.00 / 0.61 | 1.00 / 0.83 | 1.00 / 0.61 | 0.46 / 0.28 | 1.00 / 0.39 | 1.00 / 0.71 | 1.00 / 0.67 |
| B | 1.00 / 0.56 | 0.92 / 0.83 | 1.00 / 0.74 | 0.38 / 0.28 | 1.00 / 0.62 | 1.00 / 0.54 | 1.00 / 0.75 |
| C | 1.00 / 0.59 | 1.00 / 0.75 | 1.00 / 0.61 | 0.38 / 0.28 | 1.00 / 0.50 | 1.00 / 0.71 | 1.00 / 0.72 |
| D | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.88 / 0.46 | 1.00 / 0.32 | 1.00 / 1.00 | 1.00 / 1.00 |
| E | 0.75 / 1.00 | 0.92 / 1.00 | 1.00 / 1.00 | 0.88 / 0.35 | 1.00 / 0.50 | 1.00 / 1.00 | 1.00 / 1.00 |
| F | 0.75 / 1.00 | 0.96 / 1.00 | 1.00 / 1.00 | 0.88 / 0.51 | 0.50 / 0.25 | 1.00 / 1.00 | 1.00 / 1.00 |

### 5.5 Paired bootstrap vs hybrid baseline C (95% CI of mean difference, 2000 resamples)

| Config | Metric | Mean diff vs C | 95% CI | n | CI excludes 0 |
|---|---|---|---|---|---|
| A | ndcg@10 | -0.009 | [-0.041, +0.024] | 49 | no |
| A | mrr | -0.008 | [-0.059, +0.046] | 49 | no |
| B | ndcg@10 | +0.000 | [-0.046, +0.047] | 49 | no |
| B | mrr | +0.035 | [-0.035, +0.104] | 49 | no |
| D | ndcg@10 | +0.191 | [+0.127, +0.258] | 49 | yes |
| D | mrr | +0.230 | [+0.150, +0.316] | 49 | yes |
| E | ndcg@10 | +0.184 | [+0.121, +0.251] | 49 | yes |
| E | mrr | +0.226 | [+0.131, +0.323] | 49 | yes |
| F | ndcg@10 | +0.168 | [+0.086, +0.248] | 49 | yes |
| F | mrr | +0.236 | [+0.149, +0.327] | 49 | yes |
| B+alias | ndcg@10 | +0.095 | [+0.022, +0.168] | 49 | yes |
| B+alias | mrr | +0.068 | [-0.023, +0.164] | 49 | no |
| C+alias | ndcg@10 | +0.065 | [+0.021, +0.117] | 49 | yes |
| C+alias | mrr | +0.066 | [+0.021, +0.118] | 49 | yes |
| E+alias | ndcg@10 | +0.262 | [+0.188, +0.338] | 49 | yes |
| E+alias | mrr | +0.304 | [+0.223, +0.388] | 49 | yes |
| F+alias | ndcg@10 | +0.217 | [+0.126, +0.304] | 49 | yes |
| F+alias | mrr | +0.273 | [+0.180, +0.370] | 49 | yes |

## 6. Latency and candidate counts

| ID | dense p50 ms | bm25 p50 ms | fusion p50 ms | rerank p50 ms | total p50 ms | total p95 ms | candidates in | after dedupe | reranked | returned |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.053 | n/a (not run) | n/a | n/a (not run) | 0.140 | 0.163 | 20.0 | 20.0 | 0.0 | 10.0 |
| B | n/a (not run) | 0.099 | n/a | n/a (not run) | 0.188 | 0.239 | 19.8 | 19.8 | 0.0 | 10.0 |
| C | 0.055 | 0.100 | 0.024 | n/a (not run) | 0.266 | 0.321 | 39.8 | 24.4 | 0.0 | 10.0 |
| D | 0.056 | 0.102 | 0.025 | 0.167 | 0.432 | 0.484 | 39.8 | 24.4 | 20.0 | 10.0 |
| E | 0.128 | 0.182 | 0.017 | n/a (not run) | 0.415 | 0.750 | 29.1 | 15.1 | 0.0 | 9.2 |
| F | 0.128 | 0.183 | 0.017 | 0.084 | 0.519 | 0.917 | 29.1 | 15.1 | 13.8 | 9.2 |
| C@30 | 0.059 | 0.106 | 0.035 | n/a (not run) | 0.287 | 0.337 | 59.5 | 36.0 | 0.0 | 10.0 |
| E@30 | 0.128 | 0.184 | 0.017 | n/a (not run) | 0.413 | 0.755 | 31.1 | 15.8 | 0.0 | 9.2 |

Latencies are **indicative only**: one benchmark run, no run-to-run variance, repeats of one query are correlated (effective sample is about the number of queries, not queries x repeats), and E/F p95 is driven by the multi-company queries that run per-entity retrieval. They are for a ~130-chunk in-memory corpus. They show relative stage cost only and **will not extrapolate** to a ChromaDB collection with thousands of chunks and a 0.6B embedder.

## 7. Failure cases and ranking differences

**Hardest queries for the unfiltered hybrid (C)** (lowest MRR):

| qid | category | question | MRR | top-2 chunk ids |
|---|---|---|---|---|
| q020 | paraphrase | How much net sales did Borealis book in FY2025-26 on a consolidated basis? | 0.00 | acme-fy2025-kpi-orderbook, cygnus-fy2025-kpi-orderbook |
| q021 | paraphrase | What was the bottom line for Borealis in FY2025-26, consolidated? | 0.00 | acme-fy2025-cons-oth, acme-fy2025-cons-tot |
| q023 | paraphrase | How much did Borealis earn from selling its services in FY2025-26, consolidated? | 0.00 | borealis-fy2025-nar-strategy, acme-fy2025-cons-oth |
| q024 | paraphrase | How much net sales did Cygnus book in FY2025-26 on a consolidated basis? | 0.00 | acme-fy2025-kpi-orderbook, cygnus-fy2025-kpi-orderbook |
| q025 | paraphrase | What was the bottom line for Cygnus in FY2025-26, consolidated? | 0.00 | cygnus-fy2025-nar-dividend, acme-fy2025-cons-oth |

**Hardest queries for F (filters + rerank)**:

| qid | category | question | MRR | top-2 chunk ids | limitations reported |
|---|---|---|---|---|---|
| q045 | qualitative | What risks does Acme highlight in FY2025-26? | 0.00 | acme-fy2025-nar-dividend, acme-fy2025-nar-governance | - |
| q047 | qualitative | What risks does Borealis highlight in FY2025-26? | 0.00 | borealis-fy2025-nar-dividend, borealis-fy2025-nar-governance | - |
| q049 | qualitative | What risks does Cygnus highlight in FY2025-26? | 0.00 | cygnus-fy2025-nar-dividend, cygnus-fy2025-nar-governance | - |
| q016 | paraphrase | How much net sales did Acme book in FY2025-26 on a consolidated basis? | 0.25 | acme-fy2025-cons-oth, acme-fy2025-kpi-orderbook | - |
| q017 | paraphrase | What was the bottom line for Acme in FY2025-26, consolidated? | 0.25 | acme-fy2025-cons-oth, acme-fy2025-cons-rev | - |

**Ranking differences C -> F**: 27 queries improved MRR, 3 got worse, 23 unchanged/undefined.

| qid | question | MRR C | MRR F | C top-1 | F top-1 |
|---|---|---|---|---|---|
| q045 | What risks does Acme highlight in FY2025-26? | 0.50 | 0.00 | acme-fy2025-nar-dividend | acme-fy2025-nar-dividend |
| q047 | What risks does Borealis highlight in FY2025-26? | 0.50 | 0.00 | borealis-fy2025-nar-dividend | borealis-fy2025-nar-dividend |
| q049 | What risks does Cygnus highlight in FY2025-26? | 0.50 | 0.00 | cygnus-fy2025-nar-dividend | cygnus-fy2025-nar-dividend |
| q033 | How did Cygnus's attrition rate change compared with the previous year in FY2025-26? | 0.33 | 1.00 | acme-fy2025-kpi-attrition | cygnus-fy2025-nar-people |
| q041 | What was Cygnus's standalone revenue from operations in FY2025-26? | 0.33 | 1.00 | acme-fy2025-stan-rev | cygnus-fy2025-stan-rev |
| q032 | What was Cygnus's consolidated revenue from operations in FY2024-25? | 0.20 | 1.00 | acme-fy2024-cons-rev | cygnus-fy2024-cons-rev |

## 8. Best configuration

Selection rule (fixed before looking at results): composite = mean(recall@10, mrr, ndcg@10, source_page_hit); candidate must have p95 total latency <= 100 ms; configurations within 0.01 composite are treated as tied; among ties prefer the one with no reranker, then lower p50 latency (latency differences of a fraction of a millisecond are timer noise and are NOT used as evidence). `source_page_hit` is saturated on this fixture and mildly favours filtered configs.

| ID | Config | Composite | p50 total ms |
|---|---|---|---|
| D+alias | Hybrid + RRF + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.919 | 0.446 |
| E+alias | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.918 | 0.426 |
| E@10 | Hybrid + filters, 10 candidates/retriever [dense = hashing stand-in] | 0.879 | 0.407 |
| E | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.870 | 0.415 |
| E@30 | Hybrid + filters, 30 candidates/retriever [dense = hashing stand-in] | 0.870 | 0.413 |
| F+alias | Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.870 | 0.529 |

**Default on this synthetic fixture: `E+alias` (Hybrid + RRF + metadata filters [dense = hashing stand-in]).**

- `D+alias` minus `E+alias` on nDCG@10: +0.014, 95% CI [-0.039, +0.070].

**Inconclusive:** D+alias, E+alias are statistically indistinguishable on this fixture; `E+alias` is named the default only because it is the simplest of the tied set (no reranker where possible). Latency does not discriminate: every config is far under 1 ms here.

Best per metric: recall@10: 1.000 (D+alias); mrr: 0.874 (D+alias); ndcg@10: 0.865 (D+alias); source_page_hit: 1.000 (A, E, E+alias, E@10, E@30); company_full_coverage: 1.000 (A, B, C, D, E, F...).

This selection is a statement about the **synthetic** task only; it is not a recommendation for the real reports until section 12 is done.

## 9. Does reranking pay for itself?

| Pair | Δ R@10 | Δ MRR | Δ nDCG@10 | Δ page hit | rerank p50 ms | Δ total p50 ms |
|---|---|---|---|---|---|---|
| C -> D | +0.122 | +0.230 | +0.191 | +0.000 | 0.167 | +0.166 |
| E -> F | -0.056 | +0.010 | -0.017 | -0.061 | 0.084 | +0.104 |

D vs C on nDCG@10: +0.191, CI [+0.127, +0.258]. F vs C on nDCG@10: +0.168, CI [+0.086, +0.248].

**Caveat on D/F:** the stand-in heuristic scores chunks partly on company/scope agreement taken from the query analysis, so its gain over C mixes 'reranking' with 'metadata-aware scoring' (E gets the same signal through filters). It also counts query tokens that filters already guarantee (e.g. the literal 'FY2025-26'), so a chunk that lacks that literal can be pushed *down* even when it is the right one — visible in the qualitative column of 5.4 and in the C -> F differences of section 7. Whether a learned cross-encoder shares this weakness is an untested hypothesis; it has to be measured, not assumed.

The reranker here is a deterministic heuristic, so its cost (microseconds-milliseconds) says nothing about a 22M-param MiniLM or a 0.6B Qwen3-Reranker, which are orders of magnitude slower per pair. The rerank decision for the real system **cannot be made from these results**; rerun D/F with the real cross-encoder on the real labelled set.

## 10. Trade-offs and limitations

- Small synthetic fixture (see section 2): CIs in 5.5 are wide; treat differences smaller than the CI as noise.
- Filters are only as good as metadata: a hard filter excludes chunks whose field is missing; filters on fields absent from the whole index, or on values that are not report years, are *not applied* and the limitation is returned to the caller (see section 7).
- Relevance scores (BM25, cosine, RRF, rerank) never prove that a number is right. Downstream numeric validation is a separate task.
- Alias expansion is a controlled, hand-written registry; it can help paraphrases it knows and cannot help ones it does not. Wrong aliases could pull confusable metrics, so `revenue from operations`, `total income` and `other income` are separate entries.

## 11. Label

**Results: SYNTHETIC only.** No measurement in this file comes from real annual-report data.

## 12. What is needed to unblock the real-data benchmark

1. **Member 2:** persistent ChromaDB path + collection name; the metadata keys on each chunk (map to canonical names in `FieldMap`: company, fiscal_year, doc_id, pdf_page_index, printed_page, content_type, scope, table_id, table_title, headers, row_label, row_index, unit); the collection's distance metric.
2. **Member 2:** an importable `embed_query(text) -> list[float]` using the *same* Qwen3-Embedding-0.6B revision, dimension and query-instruction setting as indexing.
3. **Member 3:** 30+ manually checked queries in the JSONL schema of `proofirst_retrieval/dataset.py` (qrels pointing at real chunk ids; `provenance: manual:<name>`), including at least a few unanswerable and multi-company ones.
4. A decision on installing `chromadb`, `torch`, `sentence-transformers` (not present in this environment) and downloading a cross-encoder; then run: `python -m proofirst_retrieval.benchmark --label real --queries real_queries.jsonl --chroma-path PATH --collection NAME --embedder pkg.mod:embed_query`.
*(The real-mode code path is implemented but has not been executed against a real Chroma collection.)*
