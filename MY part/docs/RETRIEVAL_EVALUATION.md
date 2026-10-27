# ProofFirst AI — Retrieval Evaluation (Member 4)

> **DATA LABEL: SYNTHETIC.** Every number below was measured on a small fixture of *fictional* companies with *invented* figures and labels derived by construction. It validates the pipeline mechanics and compares configurations on a synthetic task. It does **not** measure retrieval quality on the real TCS / Infosys / HCLTech annual reports. The dense retriever is a hashing **stand-in**, not Qwen3-Embedding-0.6B, and the reranker is a lexical-feature heuristic, not a cross-encoder. **Real-data benchmark: BLOCKED** (inputs listed in section 12).

Generated 2026-10-27T08:51:44+00:00 by `python -m proofirst_retrieval.benchmark --label synthetic` (code version 0.1.0). Machine-readable: `artifacts/retrieval_benchmark_results.json`, `.csv`, `retrieval_benchmark_per_query.csv`.

## 1. Experimental setup

- Hardware: Apple M5, 16.0 GB RAM, macOS-27.0.1-arm64-arm-64bit-Mach-O, Python 3.13.13, GPU used: False.
- Dense: HashingDenseRetriever (1024-d signed hashing of words+char-trigrams): STAND-IN, not Qwen3-Embedding-0.6B.
- Lexical: BM25Retriever (own implementation, k1=1.5 b=0.75, unigram+bigram, crude plural stemmer).
- Vector store: none (in-memory numpy).
- Fusion: RRF, k=60, ranks **1-based**, score = `sum_lists 1/(k+rank)`; in-list repeats count once; ties broken by fusion score, then best single-list rank, then first appearance.
- Latency: 5 timed repeats per query after one warm-up, wall-clock `perf_counter`, single process, no GPU. p50/p95 are reported only if the sample has at least 20 timings.

## 2. Corpus and test dataset

- Corpus: synthetic fixture (fictional companies, invented numbers); **144 chunks**. Provenance of labels: synthetic.
- Queries: 96 (87 answerable, 9 unanswerable and excluded from rank metrics). Categories: historical=10, multi_company=18, numeric_lookup=14, paraphrase=16, qualitative=15, scope=8, terminology=6, unanswerable=9.
- Query independence: **67 distinct templates** across 96 queries (43 hand-written, one template each; the rest are legacy loop templates repeated across companies). Section 5.6 reports template-macro scores (each template counts once) next to the per-query means.
- Explicitness of the answerable queries: company named in 74/87, fiscal year in 71/87, scope in 39/87; 'latest' wording in 6, cross-company wording without names in 7.
- **Disclosed biases:** (1) queries that name company/year/scope make the hard filters in E/F act like an oracle for those queries, so their gain is an **upper bound** (the unnamed / latest / omitted-scope queries above are the ones that test routing instead); (2) legacy templates are repeated across three companies and labels are derived from chunk ids; hand-written queries were written by the same author as the aliases and the heuristic; (3) the alias registry in `query.py` contains the exact paraphrase terms used in some queries ('net sales', 'bottom line', 'workforce'), so alias gains are inflated; (4) the hashing 'dense' stand-in is word + char-trigram based, i.e. effectively lexical, so hybrid gains from A+B fusion are NOT evidence of semantic/lexical complementarity; (5) historical queries give grade 2 to the other-year chunk (comparative column), which a hard fiscal-year filter excludes, so E/F recall on that category is lowered by label design, not by a retrieval defect; (6) E/F also apply a x1.1 soft boost for `content_type=table` on numeric intents and use `company_quota`; (7) 'latest' is resolved from index metadata, which on this fixture has two report years per company.
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
| A | Dense only [dense = hashing stand-in] | 0.705 | 0.851 | 0.274 | 0.550 | 0.532 | 0.583 |
| B | BM25 only | 0.722 | 0.868 | 0.285 | 0.658 | 0.594 | 0.638 |
| C | Dense + BM25 + RRF [dense = hashing stand-in] | 0.720 | 0.841 | 0.283 | 0.597 | 0.579 | 0.615 |
| D | Hybrid + RRF + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.838 | 0.916 | 0.320 | 0.794 | 0.740 | 0.763 |
| E | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.857 | 0.881 | 0.315 | 0.796 | 0.762 | 0.773 |
| F | Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.883 | 0.893 | 0.333 | 0.884 | 0.832 | 0.834 |

| ID | page hit | company+year hit | metric/table hit | row hit | company coverage | full coverage | scope@1 |
|---|---|---|---|---|---|---|---|
| A | 0.943 | 0.985 | 0.857 | 0.681 | 0.985 | 0.955 | 1.000 |
| B | 0.862 | 0.954 | 0.757 | 0.725 | 0.970 | 0.909 | 0.821 |
| C | 0.931 | 0.969 | 0.843 | 0.739 | 0.970 | 0.909 | 0.949 |
| D | 0.943 | 1.000 | 0.943 | 0.884 | 0.985 | 0.955 | 0.974 |
| E | 0.977 | 1.000 | 0.971 | 0.913 | 0.970 | 0.909 | 1.000 |
| F | 0.989 | 1.000 | 0.957 | 0.913 | 0.985 | 0.955 | 1.000 |

### 5.2 Alias-assisted variants

| ID | Config | R@5 | R@10 | P@5 | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| B+alias | BM25 only (alias-assisted) | 0.783 | 0.944 | 0.299 | 0.655 | 0.648 | 0.695 |
| C+alias | Dense + BM25 + RRF [dense = hashing stand-in] | 0.797 | 0.900 | 0.301 | 0.634 | 0.623 | 0.652 |
| D+alias | Hybrid + RRF + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.847 | 0.946 | 0.324 | 0.847 | 0.780 | 0.815 |
| E+alias | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.885 | 0.909 | 0.329 | 0.858 | 0.814 | 0.823 |
| F+alias | Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.900 | 0.920 | 0.340 | 0.900 | 0.862 | 0.867 |

| ID | page hit | company+year hit | metric/table hit | row hit | company coverage | full coverage | scope@1 |
|---|---|---|---|---|---|---|---|
| B+alias | 0.908 | 0.923 | 0.871 | 0.870 | 0.985 | 0.955 | 0.872 |
| C+alias | 0.966 | 0.985 | 0.886 | 0.826 | 0.985 | 0.955 | 0.949 |
| D+alias | 0.954 | 1.000 | 0.957 | 0.899 | 1.000 | 1.000 | 0.974 |
| E+alias | 0.977 | 1.000 | 0.971 | 0.957 | 0.985 | 0.955 | 1.000 |
| F+alias | 1.000 | 1.000 | 0.971 | 0.957 | 1.000 | 1.000 | 1.000 |

### 5.3 Candidate-pool sweep (per retriever; C and E at 10/20/30)

| ID | Config | R@5 | R@10 | P@5 | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|---|---|---|
| C@10 | Hybrid, 10 candidates/retriever [dense = hashing stand-in] | 0.732 | 0.844 | 0.285 | 0.600 | 0.582 | 0.616 |
| C | Dense + BM25 + RRF [dense = hashing stand-in] | 0.720 | 0.841 | 0.283 | 0.597 | 0.579 | 0.615 |
| C@30 | Hybrid, 30 candidates/retriever [dense = hashing stand-in] | 0.720 | 0.841 | 0.283 | 0.599 | 0.577 | 0.614 |
| E@10 | Hybrid + filters, 10 candidates/retriever [dense = hashing stand-in] | 0.857 | 0.896 | 0.315 | 0.796 | 0.762 | 0.779 |
| E | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.857 | 0.881 | 0.315 | 0.796 | 0.762 | 0.773 |
| E@30 | Hybrid + filters, 30 candidates/retriever [dense = hashing stand-in] | 0.857 | 0.881 | 0.315 | 0.796 | 0.762 | 0.773 |

| ID | page hit | company+year hit | metric/table hit | row hit | company coverage | full coverage | scope@1 |
|---|---|---|---|---|---|---|---|
| C@10 | 0.908 | 0.954 | 0.814 | 0.739 | 0.970 | 0.909 | 0.949 |
| C | 0.931 | 0.969 | 0.843 | 0.739 | 0.970 | 0.909 | 0.949 |
| C@30 | 0.931 | 0.969 | 0.843 | 0.725 | 0.970 | 0.909 | 0.949 |
| E@10 | 0.977 | 1.000 | 0.971 | 0.913 | 0.970 | 0.909 | 1.000 |
| E | 0.977 | 1.000 | 0.971 | 0.913 | 0.970 | 0.909 | 1.000 |
| E@30 | 0.977 | 1.000 | 0.971 | 0.913 | 0.970 | 0.909 | 1.000 |

### 5.4 Per-category recall@10 / MRR (main matrix)

| ID | historical (n=10) | multi_company (n=18) | numeric_lookup (n=14) | paraphrase (n=16) | qualitative (n=15) | scope (n=8) | terminology (n=6) |
|---|---|---|---|---|---|---|---|
| A | 0.90 / 0.53 | 0.83 / 0.73 | 0.89 / 0.55 | 0.59 / 0.29 | 0.93 / 0.56 | 1.00 / 0.59 | 1.00 / 0.67 |
| B | 1.00 / 0.53 | 0.81 / 0.89 | 0.86 / 0.68 | 0.72 / 0.42 | 0.90 / 0.77 | 1.00 / 0.46 | 1.00 / 0.75 |
| C | 0.90 / 0.54 | 0.84 / 0.77 | 0.93 / 0.57 | 0.53 / 0.36 | 0.90 / 0.66 | 1.00 / 0.59 | 1.00 / 0.72 |
| D | 0.95 / 0.92 | 0.84 / 0.85 | 0.93 / 0.88 | 0.91 / 0.53 | 0.90 / 0.69 | 1.00 / 0.92 | 1.00 / 1.00 |
| E | 0.80 / 0.93 | 0.76 / 0.79 | 0.93 / 0.95 | 0.91 / 0.51 | 0.90 / 0.69 | 1.00 / 1.00 | 1.00 / 1.00 |
| F | 0.80 / 1.00 | 0.81 / 0.92 | 0.93 / 0.95 | 0.91 / 0.63 | 0.90 / 0.87 | 1.00 / 1.00 | 1.00 / 1.00 |

### 5.5 Paired bootstrap vs hybrid baseline C (95% CI of mean difference, 2000 resamples)

| Config | Metric | Mean diff vs C | 95% CI | n | CI excludes 0 |
|---|---|---|---|---|---|
| A | ndcg@10 | -0.032 | [-0.057, -0.008] | 87 | yes |
| A | mrr | -0.047 | [-0.087, -0.006] | 87 | yes |
| B | ndcg@10 | +0.023 | [-0.014, +0.061] | 87 | no |
| B | mrr | +0.061 | [+0.004, +0.119] | 87 | yes |
| D | ndcg@10 | +0.149 | [+0.101, +0.195] | 87 | yes |
| D | mrr | +0.197 | [+0.131, +0.261] | 87 | yes |
| E | ndcg@10 | +0.158 | [+0.105, +0.210] | 87 | yes |
| E | mrr | +0.199 | [+0.123, +0.275] | 87 | yes |
| F | ndcg@10 | +0.219 | [+0.175, +0.265] | 87 | yes |
| F | mrr | +0.287 | [+0.233, +0.347] | 87 | yes |
| B+alias | ndcg@10 | +0.080 | [+0.029, +0.134] | 87 | yes |
| B+alias | mrr | +0.058 | [-0.006, +0.124] | 87 | no |
| C+alias | ndcg@10 | +0.037 | [+0.011, +0.069] | 87 | yes |
| C+alias | mrr | +0.038 | [+0.009, +0.070] | 87 | yes |
| E+alias | ndcg@10 | +0.208 | [+0.149, +0.267] | 87 | yes |
| E+alias | mrr | +0.262 | [+0.186, +0.338] | 87 | yes |
| F+alias | ndcg@10 | +0.253 | [+0.204, +0.302] | 87 | yes |
| F+alias | mrr | +0.304 | [+0.244, +0.364] | 87 | yes |

### 5.6 Template-macro scores, legacy vs hand-written queries

| ID | nDCG@10 per-query | nDCG@10 template-macro | MRR per-query | MRR template-macro | nDCG@10 legacy templates | nDCG@10 hand-written |
|---|---|---|---|---|---|---|
| A | 0.583 | 0.598 | 0.550 | 0.567 | 0.579 (n=49) | 0.587 (n=38) |
| B | 0.638 | 0.668 | 0.658 | 0.704 | 0.602 (n=49) | 0.685 (n=38) |
| C | 0.615 | 0.640 | 0.597 | 0.626 | 0.590 (n=49) | 0.647 (n=38) |
| D | 0.763 | 0.766 | 0.794 | 0.813 | 0.780 (n=49) | 0.742 (n=38) |
| E | 0.773 | 0.785 | 0.796 | 0.820 | 0.771 (n=49) | 0.774 (n=38) |
| F | 0.834 | 0.845 | 0.884 | 0.911 | 0.829 (n=49) | 0.841 (n=38) |
| E+alias | 0.823 | 0.818 | 0.858 | 0.870 | 0.848 (n=49) | 0.790 (n=38) |
| D+alias | 0.815 | 0.796 | 0.847 | 0.841 | 0.865 (n=49) | 0.750 (n=38) |

Template-macro averages inside each template first, so a template repeated across companies counts once. Prefer it when judging generalisation.

### 5.7 Retrieval status and unanswerable queries

| ID | status counts (all queries) | unanswerable queries that still returned candidates |
|---|---|---|
| A | ok=96 | 9/9 |
| B | ok=96 | 9/9 |
| C | ok=96 | 9/9 |
| D | ok=96 | 9/9 |
| E | ok=96 | 9/9 |
| F | ok=96 | 9/9 |

No relevance policy is configured in any benchmark run (no threshold is invented), so dense configurations return candidates even for unanswerable questions; `status=no_evidence` only appears when nothing at all is retrieved. Thresholds must be calibrated on real labelled data.

## 6. Latency and candidate counts

| ID | dense p50 ms | bm25 p50 ms | fusion p50 ms | rerank p50 ms | total p50 ms | total p95 ms | candidates in | after dedupe | reranked | returned |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.057 | n/a (not run) | n/a | n/a (not run) | 0.156 | 0.185 | 20.0 | 20.0 | 0.0 | 10.0 |
| B | n/a (not run) | 0.089 | n/a | n/a (not run) | 0.190 | 0.254 | 19.2 | 19.2 | 0.0 | 9.9 |
| C | 0.057 | 0.090 | 0.024 | n/a (not run) | 0.272 | 0.335 | 39.2 | 24.4 | 0.0 | 10.0 |
| D | 0.058 | 0.091 | 0.025 | 0.170 | 0.443 | 0.517 | 39.2 | 24.4 | 20.0 | 10.0 |
| E | 0.138 | 0.186 | 0.024 | n/a (not run) | 0.445 | 1.185 | 37.1 | 21.2 | 0.0 | 9.5 |
| F | 0.141 | 0.188 | 0.025 | 0.168 | 0.580 | 1.376 | 37.1 | 21.2 | 16.2 | 9.5 |
| C@30 | 0.061 | 0.095 | 0.035 | n/a (not run) | 0.296 | 0.358 | 57.7 | 35.8 | 0.0 | 10.0 |
| E@30 | 0.139 | 0.188 | 0.028 | n/a (not run) | 0.449 | 1.170 | 43.8 | 25.4 | 0.0 | 9.5 |

Latencies are **indicative only**: one benchmark run, no run-to-run variance, repeats of one query are correlated (effective sample is about the number of queries, not queries x repeats), and E/F p95 is driven by the multi-company queries that run per-entity retrieval. They are for a ~130-chunk in-memory corpus. They show relative stage cost only and **will not extrapolate** to a ChromaDB collection with thousands of chunks and a 0.6B embedder.

## 7. Failure cases and ranking differences

**Hardest queries for the unfiltered hybrid (C)** (lowest MRR):

| qid | category | question | MRR | top-2 chunk ids |
|---|---|---|---|---|
| q020 | paraphrase | How much net sales did Borealis book in FY2025-26 on a consolidated basis? | 0.00 | acme-fy2025-kpi-orderbook, acme-fy2025-cons-oth |
| q021 | paraphrase | What was the bottom line for Borealis in FY2025-26, consolidated? | 0.00 | acme-fy2025-cons-oth, acme-fy2025-cons-tot |
| q023 | paraphrase | How much did Borealis earn from selling its services in FY2025-26, consolidated? | 0.00 | borealis-fy2025-nar-strategy, acme-fy2025-cons-oth |
| q024 | paraphrase | How much net sales did Cygnus book in FY2025-26 on a consolidated basis? | 0.00 | acme-fy2025-kpi-orderbook, acme-fy2025-cons-oth |
| q025 | paraphrase | What was the bottom line for Cygnus in FY2025-26, consolidated? | 0.00 | acme-fy2025-cons-oth, cygnus-fy2025-nar-dividend |

**Hardest queries for F (filters + rerank)**:

| qid | category | question | MRR | top-2 chunk ids | limitations reported |
|---|---|---|---|---|---|
| q060 | multi_company | Rank everyone by profit after tax. | 0.00 | borealis-fy2024-nar-divpolicy, cygnus-fy2024-nar-divpolicy | - |
| q016 | paraphrase | How much net sales did Acme book in FY2025-26 on a consolidated basis? | 0.25 | acme-fy2025-kpi-orderbook, acme-fy2025-cons-oth | - |
| q017 | paraphrase | What was the bottom line for Acme in FY2025-26, consolidated? | 0.25 | acme-fy2025-cons-oth, acme-fy2025-cons-rev | - |
| q020 | paraphrase | How much net sales did Borealis book in FY2025-26 on a consolidated basis? | 0.25 | borealis-fy2025-kpi-orderbook, borealis-fy2025-cons-oth | - |
| q021 | paraphrase | What was the bottom line for Borealis in FY2025-26, consolidated? | 0.25 | borealis-fy2025-cons-oth, borealis-fy2025-cons-rev | - |

**Ranking differences C -> F**: 48 queries improved MRR, 0 got worse, 48 unchanged/undefined.

| qid | question | MRR C | MRR F | C top-1 | F top-1 |
|---|---|---|---|---|---|
| q016 | How much net sales did Acme book in FY2025-26 on a consolidated basis? | 0.14 | 0.25 | acme-fy2025-cons-oth | acme-fy2025-kpi-orderbook |
| q017 | What was the bottom line for Acme in FY2025-26, consolidated? | 0.14 | 0.25 | acme-fy2025-cons-oth | acme-fy2025-cons-oth |
| q063 | Cygnus staff turnover in the current year | 0.17 | 0.33 | acme-fy2025-stan-pat | cygnus-fy2025-stan-pat |
| q032 | What was Cygnus's consolidated revenue from operations in FY2024-25? | 0.20 | 1.00 | acme-fy2024-cons-rev | cygnus-fy2024-cons-rev |
| q079 | How big was Cygnus's other income in fiscal 2026? | 0.14 | 1.00 | acme-fy2024-stan-oth | cygnus-fy2025-stan-oth |
| q087 | Set Cygnus's headcount for the year ended March 2025 beside the year ended March 2026 | 0.00 | 1.00 | acme-fy2024-nar-divpolicy | cygnus-fy2025-kpi-headcount |

Limitations the pipeline reported at run time (filters it refused to apply, missing evidence):

- 'latest' resolved to the newest report year present in the index (FY2025-26); not guessed

## 8. Best configuration

Selection rule (fixed before looking at results): composite = mean(recall@10, mrr, ndcg@10, source_page_hit); candidate must have p95 total latency <= 100 ms; configurations within 0.01 composite are treated as tied; among ties prefer the one with no reranker, then lower p50 latency (latency differences of a fraction of a millisecond are timer noise and are NOT used as evidence). `source_page_hit` is saturated on this fixture and mildly favours filtered configs.

| ID | Config | Composite | p50 total ms |
|---|---|---|---|
| F+alias | Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.922 | 0.595 |
| F | Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.900 | 0.580 |
| E+alias | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.892 | 0.460 |
| D+alias | Hybrid + RRF + rerank (stand-in heuristic) [dense = hashing stand-in] | 0.891 | 0.459 |
| E@10 | Hybrid + filters, 10 candidates/retriever [dense = hashing stand-in] | 0.862 | 0.439 |
| E | Hybrid + RRF + metadata filters [dense = hashing stand-in] | 0.857 | 0.445 |

**Default on this synthetic fixture: `F+alias` (Hybrid + RRF + filters + rerank (stand-in heuristic) [dense = hashing stand-in]).**

Best configuration **without** a reranker: `E+alias` (composite 0.892). `F+alias` minus `E+alias` on nDCG@10: +0.045, 95% CI [+0.015, +0.079]. The winner uses the stand-in heuristic reranker, which was tuned against this fixture's failures (section 9); treat the reranker's advantage as unproven for real reports.

Best per metric: recall@10: 0.946 (D+alias); mrr: 0.900 (F+alias); ndcg@10: 0.867 (F+alias); source_page_hit: 1.000 (F+alias); company_full_coverage: 1.000 (D+alias, F+alias).

This selection is a statement about the **synthetic** task only; it is not a recommendation for the real reports until section 12 is done.

## 9. Does reranking pay for itself?

| Pair | Δ R@10 | Δ MRR | Δ nDCG@10 | Δ page hit | rerank p50 ms | Δ total p50 ms |
|---|---|---|---|---|---|---|
| C -> D | +0.075 | +0.197 | +0.149 | +0.011 | 0.170 | +0.171 |
| E -> F | +0.011 | +0.088 | +0.062 | +0.011 | 0.168 | +0.134 |

D vs C on nDCG@10: +0.149, CI [+0.101, +0.195]. F vs C on nDCG@10: +0.219, CI [+0.175, +0.265].

**Caveat on D/F:** the stand-in heuristic scores chunks partly on company/scope agreement taken from the query analysis (only for constraints the retriever did *not* enforce), so its gain over C mixes 'reranking' with 'metadata-aware scoring' (E gets the same signal through filters). An earlier version also scored down chunks that lacked literals already guaranteed by hard filters (e.g. 'FY2025-26'), which pushed correct risk-narrative chunks out of the top 10 (archived v1 results: `artifacts/v1_synthetic/`). It now removes enforced-constraint spans before scoring (`rerank.relevance_text`). That fix was written after inspecting failures on this fixture, so the heuristic's advantage here is partly fitted to it. Whether a learned cross-encoder helps is an untested hypothesis.

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
