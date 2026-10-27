"""Shared settings for the parse -> chunk -> embed pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Annual-report PDFs: $RAG_REPORTS_DIR, else RAG/reports/, else the main project's data/reports/
REPORTS_DIR = Path(os.environ.get("RAG_REPORTS_DIR", ROOT / "reports"))
REPORTS_FALLBACK_DIR = ROOT.parent / "data" / "reports"
DATA_DIR = ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"        # parse_reports.py output
CHUNKS_DIR = DATA_DIR / "chunks"        # chunk_reports.py output
VECTOR_DIR = DATA_DIR / "vectorstore"   # embed_chunks.py output (Chroma)

# ---- Embedding / chunking ----
# Qwen3-Embedding-0.6B: strong open retrieval model, 1024-dim, 32k context, fits an 8 GB GPU.
# Queries must use the "query" prompt; documents are embedded as-is.
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_QUERY_PROMPT = "query"
# Chunks stay well under the model limit: smaller chunks retrieve more precisely.
CHUNK_MAX_TOKENS = 480                  # headings included; context prefix adds ~30 more
FALLBACK_MIN_RECALL = 0.85              # PyMuPDF text added when Docling captured less than this
COLLECTION = "annual_reports"

# Text-layer damage marker: "65, 487" (a space inside a number) never appears in clean text
BROKEN_NUMBER_PATTERN = r"\d, \d{3}\b"
BROKEN_NUMBER_MIN_HITS = 2              # pages at or above this are re-parsed with OCR
# Broken font mapping: TCS digits extract as Coptic ("ϮϬϮϲ"), letters as Latin Extended ("ĂƌŶŝŶŐƐ"),
# ₹ as Tibetan "༤". A few real accented names also match, which only costs an OCR pass.
GARBLED_GLYPH_PATTERN = r"[Ā-ɏͰ-Ͽༀ-࿿]"
GARBLED_GLYPH_MIN_HITS = 4

DOCLING_BATCH_PAGES = 50

# ---- Report registry ----
# parts: (first_pdf_page, name) ranges, used when the PDF has no usable bookmarks.
# part_renames: bookmark title -> readable name.
# rupee_glyph: character the report's rupee font maps to in extracted text.
# file / alt_file: name as downloaded / name used by the main project's data/reports/.
REPORTS = {
    "hcltech": {
        "company": "HCLTech",
        "legal_name": "HCL Technologies Limited",
        "nse_symbol": "HCLTECH",
        "fiscal_year": "FY2025-26",
        "fy_end": "2026-03-31",
        "report_title": "HCLTech Annual Report 2025-26",
        "file": "HCLtech_Annual-Report-2025-26.pdf",
        "alt_file": "HCLTech_FY2025-26.pdf",
        "part_renames": {
            "1": "Integrated Report",
            "2": "Integrated Report",
            "01 HCL Tech AR_2025-26_ Directors Report with Annexures": "Directors' Report",
            "02 HCL Tech AR_2025-26_CGR_DNV": "Corporate Governance Report",
            "03 HCL Tech AR_2025-26_BRSR": "Business Responsibility and Sustainability Report",
            "04 HCL Tech AR_2025-26-Standalone": "Standalone Financial Statements",
            "05 HCL Tech AR_2025-26-Consolidated": "Consolidated Financial Statements",
        },
        "rupee_glyph": None,
    },
    "tcs": {
        "company": "TCS",
        "legal_name": "Tata Consultancy Services Limited",
        "nse_symbol": "TCS",
        "fiscal_year": "FY2025-26",
        "fy_end": "2026-03-31",
        "report_title": "TCS Integrated Annual Report 2025-26",
        "file": "TCS_annual-report-2025-2026.pdf",
        "alt_file": "TCS_FY2025-26.pdf",
        # No bookmarks in this PDF; ranges read off the section title pages
        "parts": [
            (1, "Integrated Report"),
            (35, "Statutory Reports (Board's Report, Corporate Governance, BRSR)"),
            (157, "Consolidated Financial Statements"),
            (239, "Standalone Financial Statements"),
            (317, "Form AOC-1 (Subsidiaries)"),
            (323, "Notice of Annual General Meeting"),
            (340, "Glossary, GRI Index and Other Information"),
        ],
        "rupee_glyph": "h",
    },
    "infosys": {
        "company": "Infosys",
        "legal_name": "Infosys Limited",
        "nse_symbol": "INFY",
        "fiscal_year": "FY2025-26",
        "fy_end": "2026-03-31",
        "report_title": "Infosys Integrated Annual Report 2025-26",
        "file": "infosys-annual_report_2025_2026.pdf",
        "alt_file": "Infosys_FY2025-26.pdf",
        "part_renames": {},
        "rupee_glyph": "`",
    },
}


def report_dir(key: str) -> Path:
    return PARSED_DIR / key


def pdf_path(rep: dict) -> Path:
    for folder in (REPORTS_DIR, REPORTS_FALLBACK_DIR):
        for name in (rep["file"], rep.get("alt_file")):
            if name and (folder / name).is_file():
                return folder / name
    raise FileNotFoundError(f"{rep['file']} (or {rep.get('alt_file')}) not found in "
                            f"{REPORTS_DIR} or {REPORTS_FALLBACK_DIR}; set RAG_REPORTS_DIR")
