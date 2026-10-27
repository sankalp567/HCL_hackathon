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
question ─► scope guard ─► facts lookup + keyword search ─► answer ─► check every sentence ─► UI (streamed, cited)
                                   (metric lookups are answered from facts; other questions by Qwen3-4B)
```

1. **Extraction (once, offline):** `extract/docling_run.py` turns each report into Docling JSON on a GPU (layout model + TableFormer). `extract/build_facts.py` builds `data/facts.db`: tables keep their structure, numbers are parsed (Indian and Western digit grouping, ₹ crore, brackets for negatives), each company's own wording is mapped to a metric (`extract/metrics.yaml`, e.g. HCLTech's "EBIT margin", Infosys's "turnover rate"), and every number is checked against the page's text layer and against other places in the report that state it. Consolidated figures win over standalone ones (view `best_facts`).
2. **Answering:** a question naming a metric ("compare revenue", "Infosys attrition FY26") is answered straight from `facts.db`. Other questions retrieve passages (BM25 with metric synonyms) plus matching facts, and a self-hosted **Qwen3-4B-Instruct-2507** answers only from those numbered sources. Afterwards every sentence is checked: its numbers must appear in the sources it cites (otherwise the citation is re-pointed, or the sentence removed).
3. **UI:** streamed answers (SSE) with citation cards that open the cited PDF page with the cell or passage highlighted; comparison chart, table and insights built from the same facts.

## Results (`eval/`)

31 questions: numeric lookups for every metric and company, comparisons, descriptive questions, unanswerable and out-of-scope ones. Citations are checked against each PDF page's own text layer, not against our database.

| Score | Result |
|---|---|
| Expected outcome (answered / not found / out of scope) | 31/31 (100%) |
| Numeric answers with the expected numbers | 20/21 (95%) |
| Descriptive answers mentioning the expected subject | 4/4 |
| Numeric claims whose numbers are on a cited PDF page | 35/35 (100%) |
| Latency p50 / p95 | ≈0 s (answered from facts) / 4.5 s (model) |

The miss: asked for Infosys's dividend per share, the model gave the final dividend (₹25, correctly cited) instead of the year's total (₹48). Full table: [`eval/results.md`](eval/results.md).

## Run it

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). A GPU is needed only to re-extract the PDFs or to host the model.

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# put the three reports in data/reports/ (names in the source register below)

# backend + UI on http://127.0.0.1:8000, answers from facts and extracts only (no model needed)
.venv/bin/python -m uvicorn app.main:app --port 8000

# with the self-hosted model: on the GPU server, copy serve/run.py, serve/hf_server.py and a run.json
# (serve/run.sustain.json is ours: engine "transformers", GPU 1, port 8002; engine "vllm" also works), then
python run.py          # shows up in ps / nvidia-smi only as `python run.py`
# on the laptop: tunnel to it and point the backend at it. Several servers can be listed; the next one is tried
# when one does not answer (ours: sustain on 8002, the ramanujan safety net on 8001, serve/run.ramanujan.json)
ssh -N -L 8002:127.0.0.1:8002 sustain &   ssh -N -L 8001:127.0.0.1:8001 ramanujan &
LLM_BACKEND=vllm LLM_BASE_URL="http://127.0.0.1:8002/v1,http://127.0.0.1:8001/v1" LLM_MODEL=qwen3-4b \
  .venv/bin/python -m uvicorn app.main:app --port 8000

.venv/bin/python -m pytest -q          # 91 tests
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
10. The evaluation found four more bugs (an uncovered company answered, "how many" sent to the model, a correct line removed by the checker, the dividend miss); three are fixed.

## Team

| Member | Roll no. | Contribution (three sentences) |
|---|---|---|
| Balbir Prasad | TODO | TODO |
| Sankalp Turankar | 25210115 | TODO |
| Karthik | 23110070 | TODO |
| Avinash | 2311012 | TODO |

## Use of AI assistants

We used Claude Code (Anthropic) for research on PDF parsers, scaffolding, code and tests, and for the design document. Every number shown was checked: by tests against values read from the PDFs, by the evaluation set (citations checked against each page's own text), and by reading the cited pages ourselves.
