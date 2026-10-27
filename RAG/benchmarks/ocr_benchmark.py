"""Benchmark: page text before vs after OCR, on every page the pipeline sent to OCR.

before = Docling's normal parse (reads the PDF's own text layer)
after  = Docling with full-page OCR (reads the rendered page image)

Measures, per report:
  * damage markers - numbers with a stray space ("65, 487") and garbled-font characters ("ϮϬϮϲ")
  * accuracy on hand-checked figures - values read off the rendered page images (GROUND_TRUTH),
    including values that were already correct, to catch OCR breaking good text
  * time per page, from the parse manifests

Reads data/parsed/ (written by pipeline/parse_reports.py); changes nothing.
Usage:  python RAG/benchmarks/ocr_benchmark.py [--time-pages 5] [--dump REPORT PAGE]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (BROKEN_NUMBER_PATTERN, GARBLED_GLYPH_PATTERN, REPORTS,  # noqa: E402
                    pdf_path, report_dir)
from docling_core.types.doc import DoclingDocument, TableItem  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
BROKEN = re.compile(BROKEN_NUMBER_PATTERN)
GARBLED = re.compile(GARBLED_GLYPH_PATTERN)

# (report, pdf_page, what, value exactly as printed) - read off the rendered page images.
# Mixes values the PDF's text layer gets wrong with values it gets right, so OCR is
# credited for fixes and charged for anything it breaks.
GROUND_TRUTH = [
    # TCS consolidated P&L: bold total rows have a damaged text layer
    ("tcs", 167, "Revenue from operations FY26", "2,67,021"),
    ("tcs", 167, "Revenue from operations FY25", "2,55,324"),
    ("tcs", 167, "Other income FY26", "4,402"),
    ("tcs", 167, "Profit before exceptional items and tax FY26", "70,013"),
    ("tcs", 167, "Profit before tax FY26", "65,487"),
    ("tcs", 167, "Profit before tax FY25", "65,331"),
    ("tcs", 167, "Current tax FY26", "16,388"),
    ("tcs", 167, "Total tax expense FY26", "16,033"),
    ("tcs", 167, "Profit for the year FY26", "49,454"),
    ("tcs", 167, "Profit for the year FY25", "48,797"),
    # TCS 10-year trend: headers/labels in a garbled font, figures mostly clean
    ("tcs", 69, "Column header", "FY 2017"),  # FY 2026 also appears in the footnote
    ("tcs", 69, "Revenue from operations FY2026", "267,021"),
    ("tcs", 69, "United Kingdom revenue FY2022", "30,399"),
    ("tcs", 69, "Intangible and Goodwill FY2020", "1,993"),
    ("tcs", 69, "Profit before tax FY2026", "70,355"),
    ("tcs", 69, "Profit after tax FY2026", "52,820"),
    ("tcs", 69, "Headcount FY2026", "584,519"),
    ("tcs", 69, "Headcount FY2018", "394,998"),
    ("tcs", 69, "Row label", "Earnings per share"),
    # TCS Board's report text
    ("tcs", 36, "Consolidated revenue FY26", "2,67,021"),
    ("tcs", 36, "Consolidated revenue FY25", "2,55,324"),
    ("tcs", 36, "Retained earnings closing balance", "84,329"),
    # TCS integrated business model (infographic spread; Docling treats most of it as a picture)
    ("tcs", 23, "Employees", "584,519"),
    ("tcs", 23, "Employee benefits expense", "154,994"),
    ("tcs", 23, "Equity", "107,240"),
    ("tcs", 23, "Invested funds", "50,020"),
    ("tcs", 23, "R&I spend", "2,900"),
    # Pages flagged only because of real accented names
    ("hcltech", 323, "Shares in HCL Bermuda", "459,759,520"),
    ("hcltech", 323, "Investment value", "4,294"),
    ("hcltech", 77, "City name", "Iași"),
    ("hcltech", 77, "City name", "Timișoara"),
    ("hcltech", 77, "Organisation name", "Krakowska"),
    ("infosys", 5, "Client name", "Mondelēz"),
]


def page_text(doc, page):
    parts = []
    for item, _ in doc.iterate_items():
        if any(p.page_no == page for p in item.prov or []):
            parts.append(item.export_to_markdown(doc=doc) if isinstance(item, TableItem)
                         else getattr(item, "text", "") or "")
    return "\n".join(parts)


def load_pages(key):
    """{page: (before_text, after_text)} for every OCR'd page of a report."""
    base = report_dir(key)
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    ocr_pages = set(manifest["ocr_pages"])
    before = {}
    for path in sorted((base / "docling").glob("*.json")):
        first, last = map(int, path.stem.split("_")[1:])
        wanted = [p for p in ocr_pages if first <= p <= last]
        if wanted:
            doc = DoclingDocument.load_from_json(path)
            before.update({p: page_text(doc, p) for p in wanted})
    after = {p: page_text(DoclingDocument.load_from_json(base / "ocr" / f"p{p:04d}.json"), p)
             for p in ocr_pages}
    return {p: (before.get(p, ""), after[p]) for p in sorted(ocr_pages)}, manifest


def time_pages(samples):
    """Seconds per page for a normal Docling parse vs a full-page-OCR parse of the same pages."""
    import torch
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    out = {}
    for name, ocr in (("text layer", False), ("OCR", True)):
        opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
        if ocr:
            opts.ocr_options = RapidOcrOptions(force_full_page_ocr=True)
        opts.accelerator_options = AcceleratorOptions(device=device)
        conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        conv.convert(str(pdf_path(REPORTS["tcs"])), page_range=(1, 1))  # warm-up, not timed
        t = time.perf_counter()
        for key, page in samples:
            conv.convert(str(pdf_path(REPORTS[key])), page_range=(page, page))
        out[name] = (time.perf_counter() - t) / len(samples)
    return out, device.value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", nargs=2, metavar=("REPORT", "PAGE"), help="print before/after text of one page")
    ap.add_argument("--time-pages", type=int, default=5, help="OCR'd pages to time (0 to skip)")
    args = ap.parse_args()

    if args.dump:
        key, page = args.dump[0], int(args.dump[1])
        pages, _ = load_pages(key)
        before, after = pages[page]
        print(f"===== BEFORE (text layer) {key} p{page}\n{before}\n\n===== AFTER (OCR)\n{after}")
        return

    data = {key: load_pages(key)[0] for key in REPORTS}

    print("1. DAMAGE MARKERS on OCR'd pages (lower is better)")
    print(f"   {'report':<9}{'pages':>6} | {'stray-space numbers':>21} | {'garbled characters':>20} | {'pages still damaged':>20}")
    totals = [0, 0, 0, 0, 0, 0, 0]
    for key, pages in data.items():
        row = [len(pages), 0, 0, 0, 0, 0, 0]
        for before, after in pages.values():
            row[1] += len(BROKEN.findall(before)); row[2] += len(BROKEN.findall(after))
            row[3] += len(GARBLED.findall(before)); row[4] += len(GARBLED.findall(after))
            row[5] += bool(BROKEN.search(before) or len(GARBLED.findall(before)) >= 4)
            row[6] += bool(BROKEN.search(after) or len(GARBLED.findall(after)) >= 4)
        totals = [a + b for a, b in zip(totals, row)]
        print(f"   {key:<9}{row[0]:>6} | {row[1]:>9} -> {row[2]:<9} | {row[3]:>8} -> {row[4]:<9} | {row[5]:>8} -> {row[6]:<9}")
    t = totals
    print(f"   {'total':<9}{t[0]:>6} | {t[1]:>9} -> {t[2]:<9} | {t[3]:>8} -> {t[4]:<9} | {t[5]:>8} -> {t[6]:<9}")

    print("\n2. HAND-CHECKED VALUES (read off the rendered pages)")
    score = {"before": 0, "after": 0}
    outcome = {"fixed": [], "broken": [], "both ok": [], "both wrong": []}
    for key, page, what, value in GROUND_TRUTH:
        before, after = data[key][page]
        ok_b, ok_a = value in before, value in after
        score["before"] += ok_b; score["after"] += ok_a
        tag = "both ok" if ok_b and ok_a else "fixed" if ok_a else "broken" if ok_b else "both wrong"
        outcome[tag].append(f"{key} p{page} {what} = {value}")
        print(f"   {'✓' if ok_b else '✗'} -> {'✓' if ok_a else '✗'}  {key:<8} p{page:<4} {what:<46} {value}")
    n = len(GROUND_TRUTH)
    print(f"\n   correct before OCR: {score['before']}/{n} | after OCR: {score['after']}/{n}")
    for tag in ("fixed", "broken", "both wrong"):
        print(f"   {tag}: {len(outcome[tag])}" + "".join(f"\n      - {x}" for x in outcome[tag]))

    if args.time_pages:
        samples = [(k, p) for k in ("tcs", "hcltech", "infosys") for p in data[k]][: args.time_pages]
        timing, device = time_pages(samples)
        print(f"\n3. TIME PER PAGE on {len(samples)} OCR'd pages ({device})")
        for name, s in timing.items():
            print(f"   {name:<11} {s:.2f} s/page")
        print(f"   OCR costs {timing['OCR'] / timing['text layer']:.1f}x the normal parse")


if __name__ == "__main__":
    main()
