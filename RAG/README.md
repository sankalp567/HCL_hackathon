# RAG: annual reports → searchable vector database

This folder turns the three FY2025-26 annual reports (**TCS, Infosys, HCLTech**, 1,205 pages) into a **ChromaDB vector database** that can be searched by meaning. Every result says which company and PDF page it came from, so an answer can always be cited.

**The database is already built and committed** (`data/vectorstore/`). To use it you don't need the PDFs, a GPU, or the 30-minute build. Start with [Quick start](#quick-start-use-the-ready-made-database).

> This folder is independent of the main app (`app/`, `extract/`, `web/`). It doesn't change or depend on any of their files.

---

## Contents

| Path | What it is |
|---|---|
| `data/vectorstore/` | **The vector database** (ChromaDB, collection `annual_reports`, 6,172 chunks) |
| `data/chunks/*.jsonl` | The same chunks as plain JSON lines: text and metadata, one file per company. Useful for keyword (BM25) search or inspection. |
| `pipeline/parse_reports.py` | Step 1: reads the PDFs (Docling + OCR + PyMuPDF) |
| `pipeline/chunk_reports.py` | Step 2: splits the reports into chunks and attaches metadata |
| `pipeline/embed_chunks.py` | Step 3: embeds the chunks and writes the database. Also has `get_vectorstore()` for querying. |
| `pipeline/config.py` | All settings: report list, embedding model, chunk size, paths |
| `benchmarks/ocr_benchmark.py` | Measures text quality before vs after OCR on every OCR'd page. Saved output: `benchmarks/ocr_benchmark_results.txt` |
| `requirements.txt` | Exact package versions used |

---

## Quick start: use the ready-made database

```bash
# from the repository root
python -m venv .venv-rag
.venv-rag\Scripts\python -m pip install -r RAG/requirements.txt
```

```python
import sys; sys.path.insert(0, "RAG/pipeline")
from embed_chunks import get_vectorstore

store = get_vectorstore()          # opens RAG/data/vectorstore with the right embedding model

hits = store.similarity_search_with_relevance_scores(
    "What was the consolidated revenue from operations?", k=5, filter={"company": "HCLTech"})

for doc, score in hits:
    m = doc.metadata
    print(f"{score:.2f}  {m['company']}  p.{m['pages']}  [{m['chunk_type']}]  {m['text'][:100]}")
```
```
0.45  HCLTech  p.129  [table]  Revenue from operations, Consolidated.FYended.March 31,2026 = 1,30,144. Revenue from operations, Con…
```

- The first run downloads the embedding model, `Qwen/Qwen3-Embedding-0.6B` (about 1.2 GB).
- Queries work on the CPU; a GPU is used automatically if PyTorch has CUDA.
- Filters work on any metadata field: `{"company": "TCS"}`, `{"chunk_type": "table"}`, or both with `{"$and": [...]}`.

**Without LangChain**, you can use plain `chromadb`. Embed the question with the **same model and its `query` prompt**:

```python
import chromadb
from sentence_transformers import SentenceTransformer

col = chromadb.PersistentClient("RAG/data/vectorstore").get_collection("annual_reports")
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
q = model.encode("TCS attrition rate", prompt_name="query", normalize_embeddings=True)
res = col.query(query_embeddings=[q.tolist()], n_results=5, where={"company": "TCS"})
```

⚠️ Don't call `col.query(query_texts=[...])`. Chroma would embed the question with its own default model, which doesn't match this database, and the results would be meaningless.

---

## What is stored for each chunk

| Field | Example | Meaning |
|---|---|---|
| `chunk_id` (also the record ID) | `tcs-p0167-001` | report, first PDF page, sequence number |
| `company` | `TCS` | `HCLTech`, `TCS` or `Infosys` |
| `fiscal_year` | `FY2025-26` | |
| `pages` | `167` or `166-167` | **PDF page(s), counted from 1**: the page a PDF viewer jumps to. Use it for citations. |
| `chunk_type` | `table` | `text`, `table`, or `mixed` (a table plus surrounding text) |
| `n_tokens` | `475` | chunk length in the embedding model's tokens |
| `text` | `Revenue from operations, Year ended March 31, 2026 = 2,67,021. …` | **the chunk's own words**, to show and quote |
| *document* | `TCS (Tata Consultancy Services Limited) annual report FY2025-26 \| Consolidated Financial Statements \| Consolidated Statement of Profit and Loss \| Revenue from …` | **the text that was embedded**: `text` with the company, report part and headings in front |

Two points about this design:
- **The embedded text has context in front.** A P&L table never says "TCS" or "consolidated", so without the added context, the question "TCS consolidated revenue" couldn't find it.
- **Tables are stored as `row, column = value`.** Every number carries its column header, so `2,67,021` is always tied to "Year ended March 31, 2026" and can't be confused with the previous year's column.

Database facts: 6,172 chunks (HCLTech 2,063 · TCS 1,938 · Infosys 2,171), 1,024-dimension embeddings, cosine distance. A chunk is at most 480 tokens; the median is 190–300.

---

## How the pipeline works

### Step 1: Parse (`parse_reports.py`)
- **Docling on every page, on the GPU.** Docling is IBM's PDF parser. A vision model finds headings, paragraphs and tables, and a second model rebuilds each table's rows and columns.
- **OCR, automatically, on pages whose text is damaged inside the PDF.** Some TCS fonts store wrong characters, so the PDF's own text is wrong even though the page looks right. The script spots this and re-reads those pages from the image (RapidOCR).
- **PyMuPDF on every page as a second opinion,** used in step 2 to catch text Docling missed.

### Step 2: Chunk (`chunk_reports.py`)
- **Split by section.** Docling's `HybridChunker` splits on headings and measures size in the embedding model's own tokens.
  - A long section is split, and every part keeps its heading.
  - Small pieces merge only with neighbours under the **same** heading, never across sections.
  - Tables stay whole. A table too big for one chunk is split by rows, with the header row repeated in each part.
- **Pages from OCR replace the damaged originals.**
- **Completeness check:** if Docling captured less than 85% of a page's words (compared with PyMuPDF), that page's PyMuPDF text is added as extra chunks.
- **Text repairs:** the ₹ sign is restored (Infosys extracts it as `` ` ``, TCS as `h`), and TCS's broken digits are converted back.

### Step 3: Embed (`embed_chunks.py`)
- **Model:** each chunk's document text is embedded with **Qwen3-Embedding-0.6B**, a strong open retrieval model that fits an 8 GB GPU.
- **Storage:** vectors, text and metadata go into ChromaDB through LangChain.
- **Smoke test:** the script finishes by running test questions for each company.

---

## Quality: what we measured and fixed

### Choosing the parser
We tested PyMuPDF and Docling on CPU and GPU, on the same pages. "Values in the right cell" counts how many key P&L figures (FY26 and FY25) each tool placed in the correct row and column.

| | PyMuPDF | Docling (CPU) | Docling (GPU) |
|---|---|---|---|
| Speed (seconds per page) | 0.02 | 1.0–2.3 | **0.26–0.68** |
| P&L values in the right cell (3 companies) | 2 / 12 | 11 / 12 | **11 / 12** |
| Infosys tables without ruling lines | 0 tables found | read correctly | read correctly |

- **The CPU and GPU give the same output** (byte-identical on 8 of 9 test pages), so the GPU only adds speed.
- **The one miss** was a TCS value the PDF itself stores wrongly. OCR fixed it.

### Problems found in the PDFs, and the fixes

| Problem | Example | Fix | Pages fixed |
|---|---|---|---|
| TCS bold rows drop digits from the PDF text | printed `48,797` extracts as `48,97` | detect numbers with a stray space (`65, 487`) and OCR the page | 91 TCS, 1 HCLTech |
| TCS fonts swap characters | `ϮϬϮϲ` = 2026, `ĂƌŶŝŶŐƐ` = Earnings, `&z` = FY | detect odd characters and OCR the page; convert leftover digits back | 22 TCS (+3 HCLTech/Infosys with accented names; OCR reads those correctly too) |
| Docling dropped whole pages or pull-quotes | HCLTech pp. 156–161 came back empty; Infosys CEO quote missing | add PyMuPDF text where Docling captured less than 85% | 62 HCLTech, 18 TCS, 71 Infosys |
| The ₹ sign extracts as another character | Infosys `` (` crore) ``, TCS `(h crore)` | fixed during chunking | all |
| Two-page spreads and rotated pages | 32 TCS and 31 Infosys spreads | use PDF page numbers for citations, not printed folios | all |

Every chunk was checked afterwards. Apart from real accented names (Iași, Mondelēz) and TCS's product name "BαNCS", only **2 chunks** still contain damaged characters (see limitations).

### OCR: before vs after (benchmark)

**Why OCR at all?** A PDF stores two things for every character: the shape that is drawn, and a code saying which letter it is. On some TCS pages the shapes are right but the codes are wrong. So the page looks fine, while Docling and PyMuPDF, which both read the codes, get broken text (`48,797` → `48,97`, `2026` → `ϮϬϮϲ`). OCR ignores the codes and reads the rendered image instead. For the 117 flagged pages, the OCR text **replaces** the PDF's text.

`benchmarks/ocr_benchmark.py` compares the two versions of every OCR'd page: *before* (Docling reading the PDF's text) and *after* (Docling with full-page OCR). The full output is in `benchmarks/ocr_benchmark_results.txt`.

**1. Damage markers on the 117 OCR'd pages** (lower is better)

| | Before OCR | After OCR |
|---|---|---|
| Numbers with a stray space (`65, 487`) | 564 | **30** |
| Garbled-font characters (`ϮϬϮϲ`, `ĂƌŶŝŶŐƐ`, `༤`) | 760 | **15** |
| Pages still showing damage | 115 | **7** |

Of the 7 pages still showing damage, 3 are HCLTech/Infosys pages flagged only because of real accented names (Iași, Mondelēz), which were never damaged. That leaves 4 TCS pages.

**2. Accuracy on 33 hand-checked values**

These are figures and names read off the rendered page images, deliberately mixing values the PDF gets wrong with values it gets right:

| | Before OCR | After OCR |
|---|---|---|
| Correct values | 18 / 33 | **24 / 33** |

- **Fixed by OCR (11):** all the damaged TCS P&L totals (`70,013`, `65,487`, `65,331`, `16,033`, `49,454`, `48,797`), consolidated revenue in the Board's Report (`2,67,021`, `2,55,324`), the 10-year table's headers and labels (`FY 2017`, "Earnings per share"), and employee expense `154,994`.
- **Broken by OCR (5):** values that were already correct and that OCR misread:
  - TCS p. 69: `30,399` → `30339`, `1,993` → `19993`, `394,998` lost;
  - HCLTech p. 323: `459,759,520` turned into `4s s`;
  - HCLTech p. 77: `Iași` lost its diacritic.
- **Wrong both ways (4):** TCS p. 23 is an infographic that Docling treats as a picture. Most of its text is missing whether or not OCR is used.

**3. Cost:** 0.87 s/page normally vs 6.41 s/page with OCR, about 7.4 times slower (6 pages, RTX 4060). Over 117 pages that adds about 11 minutes, once.

**Takeaway.** OCR is clearly worth it on pages where the **numbers** are damaged; it repairs every damaged TCS P&L total. All 5 regressions are on pages flagged only for **garbled labels**, whose numbers were already correct. A possible refinement, not implemented:
- for pages flagged only for garbled characters, keep the PDF's numbers and take only the labels from OCR, or
- use OCR only on pages flagged for spaced-out numbers.

To re-run: `python RAG/benchmarks/ocr_benchmark.py`. It needs `data/parsed/`, which `parse_reports.py` produces.

### Retrieval smoke test (plain vector search, top 2 results, run on the committed database)
| Question | HCLTech | TCS | Infosys |
|---|---|---|---|
| "How is the company investing in generative AI?" | ✅ AI strategy, pp. 181-182 | ✅ AI client work, p. 16 | ✅ AI strategy, p. 86 |
| "Consolidated revenue from operations for the year" | ✅ p. 129, `1,30,144` (FY26) | ❌ accounting-policy note and chairman's letter; the revenue table (p. 69, `FY 2026 = 267,021`) is not in the top 2 | ⚠️ the **standalone** figure (`1,48,819`, p. 33) ranks first |
| "What was the employee attrition rate?" | ❌ parental-leave retention tables | ❌ parental-leave retention table, talent policy text | ⚠️ turnover table (13.2% male, p. 144) ranks **2nd** |

The data is in the database (for example, TCS's revenue table is clean after the broken-digit fix), but plain vector search doesn't always rank it first.

**Recommendation for the retrieval step:**
- Combine keyword search (BM25 over `data/chunks/*.jsonl`) with vector search.
- Re-rank the top 20–30 results with a cross-encoder such as `BAAI/bge-reranker-base`.
- Ask for more than 2 results (`k` ≥ 10) before re-ranking.

Exact terms like "attrition", "consolidated" or "revenue from operations" need keyword matching. Vector search alone favours passages that are about the same topic but don't contain the figure.

---

## Rebuilding from scratch (only if the PDFs or the code change)

```bash
# GPU strongly recommended: install the CUDA build of PyTorch first
.venv-rag\Scripts\python -m pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 --index-url https://download.pytorch.org/whl/cu128

.venv-rag\Scripts\python RAG/pipeline/parse_reports.py   # ~30 min on an RTX 4060, ~2 h on CPU; reruns skip finished batches
.venv-rag\Scripts\python RAG/pipeline/chunk_reports.py   # ~2 min
.venv-rag\Scripts\python RAG/pipeline/embed_chunks.py    # ~4 min, then prints the smoke test
```

The pipeline looks for the PDFs in this order:
1. `$RAG_REPORTS_DIR`
2. `RAG/reports/`
3. the main project's `data/reports/`

It accepts both the downloaded names and the main project's names (`TCS_FY2025-26.pdf` and so on). The source register with URLs is in the top-level README.

Before committing a rebuilt database, compact it so `chroma.sqlite3` stays under GitHub's 100 MB file limit (it shrinks from about 107 MB to 75 MB):
```bash
python -c "import sqlite3; c=sqlite3.connect('RAG/data/vectorstore/chroma.sqlite3'); c.execute('vacuum'); c.close()"
```

---

## Known limitations
- **Two damaged TCS chunks:** one on pp. 44–45 (a chunk that crosses into an OCR'd page) and one value reading `6ε6` on p. 305.
- **OCR misreads:** on pages flagged only for garbled labels, OCR misread a few figures that were correct in the PDF (e.g. TCS p. 69 `30,399` → `30339`). See [the OCR benchmark](#ocr-before-vs-after-benchmark).
- **Duplicate text:** the PyMuPDF fallback can repeat text Docling already captured on the same page. These chunks are plain text without table structure.
- **Ranking:** plain vector search misses some numeric questions in its top 2 (TCS revenue, attrition; see the smoke test above). BM25 plus a re-ranker is the planned fix.
- **Citations use PDF page numbers, not the numbers printed on the page.** TCS's printed numbers are damaged, and spreads put two printed pages on one PDF page.
