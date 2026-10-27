# RAG ingestion pipeline: parse → chunk → embed

This pipeline turns the three FY2025-26 annual reports (TCS, Infosys, HCLTech) into a ChromaDB vector store. Every chunk records its company and PDF page, so answers can cite them.
It is separate from the main app in this repository and doesn't change any of its files.

```
PDFs ─► parse_reports.py ─► chunk_reports.py ─► embed_chunks.py ─► data/vectorstore/ (ChromaDB)
        Docling (GPU)        section-aware        Qwen3-Embedding-0.6B
        + OCR damaged pages  token-sized chunks   cosine, normalized
        + PyMuPDF check      + metadata
```

## Setup

```bash
python -m venv .venv-gpu
.venv-gpu\Scripts\python -m pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.venv-gpu\Scripts\python -m pip install -r RAG/requirements.txt
```

The scripts also run on the CPU-only build of PyTorch, but parsing takes about 4 times longer.

**Reports.** The pipeline looks for the PDFs in this order:
1. `$RAG_REPORTS_DIR`
2. `RAG/reports/`
3. the main project's `data/reports/`

It accepts both the downloaded file names and the main project's names (`TCS_FY2025-26.pdf` and so on). See the source register in the top-level README.

## Run

```bash
.venv-gpu\Scripts\python RAG/pipeline/parse_reports.py   # ~27 min first time on an RTX 4060; reruns skip finished batches
.venv-gpu\Scripts\python RAG/pipeline/chunk_reports.py   # ~2 min
.venv-gpu\Scripts\python RAG/pipeline/embed_chunks.py    # ~4 min, ends with a retrieval smoke test
```

| Script | Writes |
|---|---|
| `parse_reports.py` | `data/parsed/<company>/`: Docling JSON in 50-page batches, OCR output for damaged pages, per-page PyMuPDF records, `manifest.json` |
| `chunk_reports.py` | `data/chunks/<company>.jsonl`: `id`, `text`, `embed_text`, `metadata` |
| `embed_chunks.py` | `data/vectorstore/`: ChromaDB collection `annual_reports`, plus `embedding_info.json` |

`data/` and `reports/` are git-ignored; rebuild them with the three scripts.

## Stored in ChromaDB (6,172 chunks)

| Field | Example | Meaning |
|---|---|---|
| id / `chunk_id` | `tcs-p0167-001` | report, first PDF page, sequence |
| `company` | `TCS` | `HCLTech`, `TCS` or `Infosys` |
| `fiscal_year` | `FY2025-26` | |
| `pages` | `167` or `166-167` | 1-based **PDF** page(s): the page a PDF viewer jumps to |
| `chunk_type` | `table` | `text`, `table` or `mixed`, from Docling's layout labels (no LLM) |
| `n_tokens` | `475` | size in the embedding model's tokens |
| `text` | `Revenue from operations, Year ended March 31, 2026 = 2,67,021 …` | the chunk's own text, to show and quote |
| document | `TCS (…) annual report FY2025-26 \| Consolidated Financial Statements \| …` | the text that was embedded: `text` with company, report part and headings in front |

Tables are written as `row, column = value`, so every number carries its column header (year).

## Querying

```python
import sys; sys.path.insert(0, "RAG/pipeline")
from embed_chunks import get_vectorstore

store = get_vectorstore()   # same model + query prompt the index was built with
for doc, score in store.similarity_search_with_relevance_scores(
        "What was TCS's attrition rate?", k=5, filter={"company": "TCS"}):
    print(round(score, 3), doc.metadata["company"], "p.", doc.metadata["pages"], doc.metadata["text"][:100])
```

You can also use plain `chromadb` (`chromadb.PersistentClient("RAG/data/vectorstore")`). In that case, embed the query yourself with `Qwen/Qwen3-Embedding-0.6B` and `prompt_name="query"`. Don't use `query_texts`: Chroma would embed the query with its own default model, and the results wouldn't match this index.

## Design decisions

- **Docling for every page, on the GPU.** Infosys's tables have no ruling lines: PyMuPDF and pdfplumber found 0 tables on its P&L page, while Docling found the right values. On P&L pages, PyMuPDF placed 2 of 12 key values correctly and Docling 11 of 12. The CPU and GPU give the same output; the GPU is 3–6 times faster.
- **OCR only on pages with a damaged text layer** (RapidOCR via Docling; 113 TCS pages, 3 HCLTech, 1 Infosys). Detection is automatic:
  - numbers with a stray space, like `65, 487` (TCS's bold rows drop digits: `48,797` comes out as `48,97`);
  - broken-font characters: digits as Coptic letters (`ϮϬϮϲ` for 2026), letters as Latin-Extended letters, ₹ as `༤`.
- **PyMuPDF completeness check.** If Docling captured less than 85% of a page's words, that page's PyMuPDF text is added as extra chunks. This caught pages Docling dropped entirely (HCLTech pp. 156–161) and pull-quotes it skipped.
- **Chunking:** Docling's `HybridChunker`, measured in the embedding model's own tokens (max 480).
  - It splits on headings; oversized sections keep their heading in every part.
  - Small chunks merge only with neighbours under the same heading.
  - Tables stay whole, or are split by rows with the header repeated.
- **Embedding:** `Qwen/Qwen3-Embedding-0.6B` (1,024 dimensions, 32k context, runs on an 8 GB GPU). Queries use its `query` prompt.
- **Text clean-up:** the ₹ sign is repaired (Infosys extracts it as `` ` ``, TCS as `h`), and TCS's shifted digits and `&z` (for "FY") are converted back.
- **No printed page numbers.** TCS's folios are damaged, and spreads and rotated pages break detection, so citations use PDF page numbers.

## Known limitations

- Two TCS chunks still contain damaged characters: pp. 44–45, where a chunk crosses into an OCR'd page, and one value `6ε6` on p. 305.
- Plain vector search ranks some numeric questions poorly. For "attrition", parental-leave retention tables rank above turnover tables, and for "consolidated revenue", Infosys's standalone figure comes first. Use keyword (BM25) plus vector search and a re-ranker (for example `BAAI/bge-reranker-base`) at query time.
