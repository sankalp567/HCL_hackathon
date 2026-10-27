# IT Annual Reports, Compared and Cited

HCLTech AI Hackathon, IIT Gandhinagar · Use Case 2: compare the FY2025-26 annual reports of **TCS, Infosys and HCLTech**.

- **R1, comparison chart:** revenue, net profit, operating/EBIT margin, headcount and attrition for the three companies. Every value carries the PDF page (and the exact table cell) it came from.
- **R2, questions answered only from the reports:** each answer cites company · report · PDF page, every sentence is checked against what it cites, and the app says "Not found in the reports" when the evidence is absent and "Out of scope" for investment questions.
- Everything runs on free, self-hosted tools: no paid API.

Design document with diagrams: [`docs/pipeline.pdf`](docs/pipeline.pdf).

## How it works

```
PDFs ──► Docling (GPU): text, tables as cell grids, page + box ──► facts.db (SQLite)
                                                                    ├─ facts: one row per number, with unit, FY, page, box, quote, checks
                                                                    ├─ table cells, passages + full-text index (FTS5)
question ─► scope rules ─► model check: another company? only part of a metric, or a value computed from it? which metric?
         ─► facts lookup + keyword search ─► answer ─► check every sentence ─► UI (streamed, cited)
            (metric lookups are answered from facts; other questions by Qwen3-30B-A3B, with Qwen3-4B as fallback)
```

1. **Extraction (once, offline):** `extract/docling_run.py` turns each report into Docling JSON on a GPU (layout model + TableFormer). `extract/build_facts.py` builds `data/facts.db`: tables keep their structure, numbers are parsed (Indian and Western digit grouping, ₹ crore, brackets for negatives), each company's own wording is mapped to a metric (`extract/metrics.yaml`, e.g. HCLTech's "EBIT margin", Infosys's "turnover rate"), and every number is checked against the page's text layer and against other places in the report that state it. Consolidated figures win over standalone ones (view `best_facts`).
2. **Answering:** keyword rules handle the common cases instantly. One short model call then checks what rules miss, choosing only from fixed lists and never supplying a number:
   - a question about a company we don't cover ("apples company revenue") is answered "Not found", and one asking for investment advice in any wording ("is it a good investment", "should I short it") "Out of scope";
   - a question about only part of a metric ("revenue from North America") or a value computed from it ("revenue growth") is answered from the report's text, not from the metric template;
   - wording the rules don't know ("staff numbers trend for infy") is mapped to a metric and its chart.

   A question naming a metric ("compare revenue", "Infosys attrition FY26") is answered straight from `facts.db`. Other questions retrieve passages (BM25 with metric synonyms) plus matching facts, and the self-hosted **Qwen3-30B-A3B-Instruct-2507** (FP8, vLLM on an A100) answers only from those numbered sources; **Qwen3-4B-Instruct-2507** on a second server is the fallback. Every sentence is then checked: its numbers must appear in the sources it cites (otherwise the citation is re-pointed, or the sentence removed). Model checks and model answers are cached by exact question, so a repeated question costs no model call.
3. **UI:** streamed answers (SSE) with citation cards that open the cited PDF page with the cell or passage highlighted; comparison chart, table and insights built from the same facts.

## Results (`eval/`)

43 questions: numeric lookups for every metric and company, comparisons, descriptive questions, unanswerable and out-of-scope ones, and 12 adversarial ones: companies we don't cover (Apple, Microsoft, Tata Motors), years the reports lack, investment advice in other words, and controls that must still be answered. Citations are checked against each PDF page's own text layer, not against our database.

| Score | Result |
|---|---|
| Expected outcome (answered / not found / out of scope) | 43/43 (100%) |
| Numeric answers with the expected numbers | 25/25 (100%) |
| Descriptive answers mentioning the expected subject | 4/4 |
| Numeric claims whose numbers are on a cited PDF page | 41/41 (100%) |
| Latency p50 / p95 | 1.6 s / 5.7 s for new questions; ≈0 s / 4.2 s when the question check is cached |

Full table: [`eval/results.md`](eval/results.md). We chose the model by measurement; the two models were run separately on the first 31 questions:

| | Qwen3-4B (sustain, RTX A4000) | Qwen3-30B-A3B FP8 (ramanujan, A100) |
|---|---|---|
| Numeric answers with the expected numbers | 20/21 (gave Infosys's final dividend, ₹25, not the year's ₹48) | 21/21 |
| Numeric claims on a cited page | 35/35 | 36/36 |
| Latency p95 | 6.5 s | 4.2 s |
| Choosing metric and chart for 12 unusual phrasings | 11/12 | 12/12 |
| Question check on 16 tricky questions (other companies, parts, computed values) | 15/16 | 16/16 |

**Team test set** (`eval/test_queries.csv`, 42 harder questions with expected value and page: EPS, dividends, growth, "which company", computed values): **28/42** right (`python eval/run_test_queries.py`, results in [`eval/test_queries_results.md`](eval/test_queries_results.md)). The misses are documented, not hidden: five headcount questions expect the headline headcount (TCS 5,84,519) while we extract the BRSR total including non-permanent staff (6,17,437); HCLTech's ₹60 dividend (p. 8) is not retrieved; TCS's 13.7% attrition is printed as "13." in the PDF's text layer; the report prints two HCLTech profit figures (16,642 and 16,652); and we never compute values the reports do not print (a revenue gap, a combined headcount).

**Vector search vs keyword search** (`eval/compare_retrieval.py`): the `RAG/` ChromaDB chunks, re-embedded with the same model (Qwen3-Embedding-0.6B), against our FTS5/BM25 retrieval, on the test set's 74 (question, company) pairs. Expected page retrieved: vectors 25/74, keywords 34/74; expected value retrieved: 23/74 vs 33/74. Vectors found 2 values keywords missed, keywords 12 that vectors missed: the two complement each other, so the planned next step is hybrid retrieval (BM25 + vectors, fused, then re-ranked), which `MY part/` implements.

**Does the teammates' retrieval stack improve ours?** (`eval/compare_stages.py`): Sankalp's `MY part/` stages run over Karthik's chunks, same 74 pairs, top 4 per company. Expected page / value retrieved: his BM25 32 / 27; vectors 25 / 23; RRF fusion of both 34 / 28; fusion + lexical re-ranker **35 / 30**; our keyword retrieval over `facts.db` **34 / 33**. The full stack matches ours on pages and trails on values, because our passages carry each table's title and footnotes and our facts carry the clean number. Each finds things the other misses, so the integration worth doing after the freeze is: our facts first, then his fusion + re-ranker over both chunk sets as the passage retriever, measured on this test set.

**Chat session context:** a short follow-up ("and for Infosys?", "what about FY2024-25?", "why?") takes the company, metric and year it does not name from the last three questions of the chat; the UI sends them with each question. **Chart requests:** "pie chart of headcount", "bar graph", "line plot" set the chart type the answer card draws (a pie is drawn as one bar per company for now, with a note).

## Run it

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). A GPU is needed only to re-extract the PDFs or to host the model.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# put the three reports in data/reports/ (names in the source register below)

# backend + UI on http://127.0.0.1:8000, answers from facts and extracts only (no model needed)
.venv/bin/python -m uvicorn app.main:app --port 8000

# with the self-hosted models: on each GPU server, copy serve/run.py (and serve/hf_server.py) with a run.json, then
python run.py          # shows up in ps / nvidia-smi only as `python run.py`
# ours: serve/run.ramanujan.json (Qwen3-30B-A3B FP8, vLLM, A100, port 8001; `python run.py --download-only` first)
#       serve/run.sustain.json   (Qwen3-4B, transformers, RTX A4000, port 8002; the fallback)
# on the laptop: tunnel to both and list them in order, with one model name per server
ssh -N -L 8001:127.0.0.1:8001 ramanujan &   ssh -N -L 8002:127.0.0.1:8002 sustain &
LLM_BACKEND=vllm LLM_BASE_URL="http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1" LLM_MODEL="qwen3-30b-a3b,qwen3-4b" \
  .venv/bin/python -m uvicorn app.main:app --port 8000

.venv/bin/python -m pytest -q          # 115 tests
.venv/bin/python eval/run_eval.py      # against a running backend; writes eval/results.md
```

`data/facts.db` is committed. To rebuild it: run `extract/docling_run.py` on a GPU (`requirements-gpu.txt`) to produce `data/extracted/*.docling.json`, then `python -m extract.build_facts --summary` (about 10 seconds).

## Source register

| Company | Report | Fiscal year | Pages | URL | Downloaded | SHA-256 |
|---|---|---|---|---|---|---|
| TCS | TCS Annual Report 2025-26 | FY2025-26 | 360 | [tcs.com/…/annual-report-2025-2026.pdf](https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2025-26/ar/annual-report-2025-2026.pdf) | 2026-10-27 | `a19bddbe…a5a0` |
| Infosys | Infosys Integrated Annual Report 2025-26 | FY2025-26 | 383 | [infosys.com/…/infosys-ar-26.pdf](https://www.infosys.com/investors/reports-filings/annual-report/annual/documents/infosys-ar-26.pdf) | 2026-10-27 | `602f04ec…078f` |
| HCLTech | HCLTech Integrated Annual Report 2025-26 | FY2025-26 | 462 | [hcltech.com/…/Annual-Report-2025-26](https://www.hcltech.com/hcl-annual-report-2026/pdfs/Annual-Report-2025-26) | 2026-10-27 | `6cb198ff…82d0` |

Only these three reports are used; no news, broker notes or websites.

## Failed attempts (and what we did instead)

1. Downloading the reports by script: blocked by the companies' bot protection. Downloaded by hand.
2. Plain text extraction: tables flatten, chart labels scramble (TCS's revenue chart reads "2 67, 0 2 …"), KPI tiles lose label–value pairs, and TCS's fonts extract ₹ as "H"/"J". Moved to Docling with TableFormer.
3. PyMuPDF4LLM's default run crashed in OCR (no Tesseract language data); Docling's picture classifier labelled TCS's revenue chart as a "Logo", so it can't be trusted alone to find charts.
4. Installing Docling on the offline GPU server: two attempts failed on CUDA package conflicts before a no-dependency install of a pre-resolved package list worked.
5. First facts build picked Infosys's standalone revenue; fixed by preferring consolidated figures.
6. TCS's text layer damages numbers ("6, 17 437", "97 5", "13."). We repair only what commas prove ("6, 17 437" = 6,17,437 = 5,84,519 + 32,918); the rest is left unread rather than guessed, so TCS's FY26 attrition is missing.
7. A header row Docling did not mark mixed up standalone and consolidated margins for Infosys (21.9% vs 20.3%), and a section heading leaked a "discount rate" into operating margin. Fixed header and label handling.
8. The model refused "Infosys voluntary attrition" although the fact was retrieved: the defining footnote was not attached and the context was clipped. Attaching table footnotes fixed it.
9. vLLM 0.32 ships PyTorch for CUDA 13; the servers' driver supports CUDA 12.8. Pinned vLLM 0.11.0, and transformers below 5 (vLLM 0.11 calls a tokenizer method that transformers 5 removed).
10. The evaluation found four more bugs (an uncovered company answered, "how many" sent to the model, a correct line removed by the checker, the dividend miss); three are fixed, and the 30B model fixed the fourth.
11. Letting the model write a free-form answer plan (what to show: text, chart, table) failed: the 4B model often answered instead of planning, inventing reasons and claiming dividend data "is not available". Restricting it to choices from fixed lists, then checking those choices, worked.
12. "what was apples company revenue" was answered with our three companies' revenue: the keyword gate knew only a list of IT companies. The model check now catches companies we don't cover (16/16 on our probe set with the 30B). A single "whole metric?" flag first missed growth and per-employee questions; separate "part" and "computed" flags fixed that.
13. Two more from trying the app: "what is tcs revenue was there in year 2023?" was answered with FY2025-26 figures (a bare year was ignored; it now means the fiscal year ending that March, and an answer that gives no figure for a figure question is "Not found"), and "Is Infosys a good investment right now?" slipped past the keyword scope rule (the model check now flags investment advice in any wording). Both are in the evaluation now, with ten similar adversarial questions.

## Next steps (measured against `eval/`)

1. **Hybrid retrieval:** facts first, then Sankalp's BM25 + vector fusion and re-ranker (`MY part/`) over both chunk sets (`RAG/data/chunks/`, our passages) as the passage retriever; accept only if `eval/run_test_queries.py` improves on 28/42.
2. **Charts the model designs, never draws from its own numbers:** first a validated JSON chart spec (pie, stacked, dual axis) filled from `best_facts` and drawn by the existing Chart.js card; then, for what a spec cannot express, a generated matplotlib script with static checks (import whitelist, no numeric literals), a no-network sandbox, and a post-run comparison of what was plotted against the injected rows. Pie charts in the card come first.
3. **Headline headcount** (TCS 5,84,519) as a second headcount metric next to the BRSR total, and OCR for TCS's damaged pages (its 13.7% attrition).
4. **A second check after answering:** does the answer cover the asked company, year and measure? Every new failure goes into `eval/questions.jsonl`.

## Team

| Member | Roll no. | Contribution (three sentences) |
|---|---|---|
| Balbir Prasad | 25210034 | **Local LLM hosting and optimization.** Set up and optimized the locally hosted LLM for natural-language question understanding and answer generation. Worked on inference configuration, model performance, and structured outputs to support the application's RAG pipeline. Contributed to integrating LLM capabilities while balancing response quality, latency, and local resource constraints. |
| Karthik | 23110070 | **Document parsing, chunking, and vector database.** Implemented annual-report processing using Docling, extracting document text, tables, and relevant source-page information. Developed structure-aware chunking and generated Qwen3-Embedding-0.6B embeddings for storage in ChromaDB. Prepared the document retrieval infrastructure to support semantic search and evidence-grounded answers across TCS, Infosys, and HCLTech annual reports. |
| Avinash | 23110123 | **Frontend and testing.** Developed the frontend experience for interacting with annual-report intelligence and comparing company information. Worked on the chatbot interface, static company comparison dashboard, dynamic visualizations, and citation-based evidence inspection. Created test cases to evaluate answer correctness, source citations, numerical comparisons, and safe-refusal behavior. |
| Sankalp Turankar | 25210115 | **Retrieval, reranking, and evaluation.** Implemented a modular retrieval pipeline incorporating query analysis, BM25 search, metadata filtering, Reciprocal Rank Fusion (RRF), and reranking. Developed evidence assembly, multi-company retrieval, source-provenance preservation, failure handling, and automated tests for retrieval quality. Built a benchmark framework to compare retrieval configurations using Recall@K, MRR, nDCG, source-page accuracy, and latency, while preparing integration with the real ChromaDB index. |

## Use of AI assistants

We used Claude Code (Anthropic) for research on PDF parsers, scaffolding, code and tests, and for the design document. Every number shown was checked: by tests against values read from the PDFs, by the evaluation set (citations checked against each page's own text), and by reading the cited pages ourselves.
