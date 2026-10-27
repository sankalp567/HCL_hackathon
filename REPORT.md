# Report

**1. Approach.** The three reports are extracted once into SQLite: Docling rebuilds tables as cell grids, and every number becomes a fact with its page, checked against the page text. Keyword rules and a short, fixed-choice model check read each question. Metric questions are answered from the facts; others by a self-hosted model, only from retrieved, numbered sources, with every sentence checked against what it cites.

**2. Tools, models, libraries.** Docling (layout model + TableFormer) on RTX A4000s; PyMuPDF; SQLite with FTS5; Qwen3-30B-A3B-Instruct-2507 (FP8, vLLM, A100), with Qwen3-4B-Instruct-2507 as fallback; FastAPI with server-sent events; Chart.js; pytest.

**3. Constraints.** No paid APIs; GPU servers without internet; 4.5 hours to build.

**4. Assumptions.** FY2025-26 ends 31 March 2026. Consolidated figures are preferred. Citations use PDF page numbers.

**5. Top five design decisions.**
1. Numbers are data, not text: one row per number, with its page and cell.
2. Metric questions bypass the model, so their numbers cannot be invented.
3. Every sentence must be supported by its cited source, or is re-pointed or removed.
4. Refusal is designed in: scope rules, the model check, an evidence gate, an abstain rule.
5. Each company's own wording is kept: HCLTech reports EBIT margin, Infosys a turnover rate.

**6. AI-assistant support.** Claude Code helped with research, code, tests and documents; we checked its output with tests, an evaluation set and the PDFs.

**7. Prior work.** Docling and TableFormer (IBM Research); Okapi BM25 through SQLite FTS5; retrieval-augmented generation with citations (Lewis et al., 2020); vLLM (PagedAttention); Qwen3 (Alibaba).

**8. Failed attempts.** Plain-text extraction scrambled tables and charts. TCS's fonts damaged numbers; we repair only what commas prove. A missing footnote made the model refuse a correct answer. Free-form plans let the small model invent content, and keyword rules answered an Apple question with our companies' revenue. Details in the README; results: 43 test questions, 100% expected outcomes and cited pages.
