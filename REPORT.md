# Report

**Approach.** We extract the three FY2025-26 annual reports once into SQLite: Docling rebuilds tables as cell grids with page and position, and every number becomes a fact (company, metric, year, value, unit, page, quote) checked against the page text and other pages. Metric questions are answered from these facts; for others, a self-hosted Qwen3-4B answers only from retrieved, numbered sources, and each sentence is checked against what it cites.

**Tools, models, libraries.** Docling (layout model + TableFormer) on two RTX A4000s; PyMuPDF; SQLite with FTS5; Qwen3-4B-Instruct-2507 served by vLLM; FastAPI with server-sent events; HTML, JavaScript and Chart.js; pytest.

**Constraints.** No paid APIs; GPU servers without internet; 4.5 hours to build.

**Assumptions.** FY2025-26 is the year ending 31 March 2026. Consolidated figures are preferred. Citations use PDF page numbers.

**Top five design decisions.**
1. Numbers are data, not text: one database row per number, with its page and cell.
2. Metric questions bypass the model, so their numbers cannot be invented.
3. Every sentence must be supported by a cited source; otherwise its citation is re-pointed or it is removed.
4. Refusal is designed in: scope guard, evidence gate, abstain rule, coverage check.
5. Each company's own wording is kept: HCLTech reports EBIT margin, Infosys a turnover rate.

**AI-assistant support.** Claude Code helped research parsers and write code, tests and documents. We checked its output with tests, an evaluation set and the PDFs.

**Prior work.** Docling and TableFormer (IBM); BM25 retrieval-augmented generation.

**Failed attempts.** Plain-text extraction scrambled charts and tables. TCS's fonts damaged numbers; we repair only what commas prove. A missing footnote made the model refuse a correct answer. vLLM's newest build needed a CUDA 13 driver. Details are in the README.

**Results.** On 31 test questions: 100% expected outcomes, 95% numeric answers correct, and 100% of cited numbers found on the cited page.
