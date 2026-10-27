# Report

**Approach.** We extract the three reports once into SQLite: Docling rebuilds tables as cell grids with positions, and every number becomes a fact with its page, checked against the page text and other pages. A short model call checks what keyword rules miss. Metric questions are answered from the facts; others by a self-hosted model, only from retrieved, numbered sources, with every sentence checked against what it cites.

**Tools, models, libraries.** Docling (layout model + TableFormer) on RTX A4000s; PyMuPDF; SQLite with FTS5; Qwen3-30B-A3B-Instruct-2507 (FP8, vLLM, A100), with Qwen3-4B-Instruct-2507 as fallback; FastAPI with server-sent events; Chart.js; pytest.

**Constraints.** No paid APIs; GPU servers without internet; 4.5 hours to build.

**Assumptions.** FY2025-26 is the year ending 31 March 2026. Consolidated figures are preferred. Citations use PDF page numbers.

**Top five design decisions.**
1. Numbers are data, not text: one database row per number, with its page and cell.
2. Metric questions bypass the model, so their numbers cannot be invented.
3. Every sentence must be supported by its cited source, or is re-pointed or removed.
4. Refusal is designed in: scope rules, a fixed-choice model check (uncovered companies, partial or computed values), an evidence gate, an abstain rule.
5. Each company's own wording is kept: HCLTech reports EBIT margin, Infosys a turnover rate.

**AI-assistant support.** Claude Code helped with research, code, tests and documents; we checked its output with tests, an evaluation set and the PDFs.

**Failed attempts.** Plain-text extraction scrambled tables and charts. TCS's fonts damaged numbers; we repair only what commas prove. A missing footnote made the model refuse a correct answer. Free-form plans made the small model invent content, and keyword rules answered an Apple question with our companies' revenue. Details in the README.

**Results.** On 43 test questions, 12 of them adversarial: 100% expected outcomes, 100% correct numeric answers, 100% of cited numbers on the cited page. A side-by-side run chose Qwen3-30B-A3B over Qwen3-4B (21/21 vs 20/21; p95 4.2 s vs 6.5 s).
