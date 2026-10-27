"""Step 1: parse each annual-report PDF.

Outputs per report in data/parsed/<key>/:
  pages.jsonl          one record per PDF page from PyMuPDF: report part, statement scope,
                       spread flag, damaged-text flag, raw text. Pages are PDF page numbers
                       (printed folios are unreliable: TCS's are damaged, Infosys has rotated pages)
  docling/*.json       Docling documents (layout + tables), 50-page batches, no OCR
  ocr/p####.json       Docling with full-page OCR, only for pages whose text layer is damaged
                       (spaced-out numbers like "65, 487" or garbled font glyphs like "ϮϬϮϲ")
  manifest.json        what was parsed, with which settings, and how long it took

Re-running skips batches that already exist (use --force to redo).

Usage (GPU venv):  .venv-gpu\\Scripts\\python pipeline\\parse_reports.py [--reports tcs infosys] [--force]
"""
import argparse
import json
import re
import statistics
import time
from datetime import datetime, timezone

import fitz

from config import (BROKEN_NUMBER_MIN_HITS, BROKEN_NUMBER_PATTERN, DOCLING_BATCH_PAGES,
                    GARBLED_GLYPH_MIN_HITS, GARBLED_GLYPH_PATTERN, REPORTS, pdf_path, report_dir)

BROKEN_NUMBER = re.compile(BROKEN_NUMBER_PATTERN)
GARBLED_GLYPH = re.compile(GARBLED_GLYPH_PATTERN)
WORD = re.compile(r"[A-Za-z]{2,}|\d[\d,.]*")


# ---------------------------------------------------------------- PyMuPDF pass

def part_lookup(rep, doc):
    """pdf_page -> (report_part, part_group) from config ranges or PDF bookmarks."""
    n = len(doc)
    if rep.get("parts"):
        starts = rep["parts"] + [(n + 1, None)]
        return {p: (name, "") for (s, name), (e, _) in zip(starts, starts[1:]) for p in range(s, e)}

    renames = rep.get("part_renames", {})
    toc = doc.get_toc()
    lookup = {p: ("", "") for p in range(1, n + 1)}
    top_level = [t for t in toc if t[0] == 1]
    deepest = max((t[0] for t in toc), default=1)
    # Each page takes the last bookmark at or before it, at the top and deepest levels
    for level, target in ((1, "group"), (deepest, "part")):
        entries = [(page, renames.get(title, title).strip()) for lvl, title, page in toc if lvl == level]
        if not entries:
            continue
        current = ""
        it = iter(sorted(entries))
        nxt = next(it, None)
        for p in range(1, n + 1):
            while nxt and nxt[0] <= p:
                current = nxt[1]
                nxt = next(it, None)
            part, group = lookup[p]
            lookup[p] = (current, group) if target == "part" else (part, current)
    if len(top_level) == len(toc):  # single-level bookmarks: group would duplicate part
        lookup = {p: (part, "") for p, (part, _) in lookup.items()}
    # Pages before the first sub-bookmark fall back to their top-level part
    return {p: (part or group, group) for p, (part, group) in lookup.items()}


def statement_scope(report_part):
    part = report_part.lower()
    if "consolidated" in part:
        return "consolidated"
    if "standalone" in part:
        return "standalone"
    return ""


def pymupdf_pass(key, rep):
    doc = fitz.open(pdf_path(rep))
    parts = part_lookup(rep, doc)
    typical_width = statistics.median(p.rect.width for p in doc)
    records = []
    for page in doc:
        pno = page.number + 1
        text = page.get_text(sort=True)
        is_spread = page.rect.width > 1.6 * typical_width
        part, group = parts[pno]
        records.append({
            "pdf_page": pno,
            "width": round(page.rect.width),
            "height": round(page.rect.height),
            "is_spread": is_spread,
            "is_landscape": page.rect.width > page.rect.height and not is_spread,
            "report_part": part,
            "part_group": group,
            "statement_scope": statement_scope(part),
            "broken_number_hits": len(BROKEN_NUMBER.findall(text)),
            "garbled_glyph_hits": len(GARBLED_GLYPH.findall(text)),
            "word_count": len(WORD.findall(text)),
            "text": text,
        })
    return records


# ---------------------------------------------------------------- Docling passes

def make_converter(device, ocr):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=ocr, do_table_structure=True)
    if ocr:
        opts.ocr_options = RapidOcrOptions(force_full_page_ocr=True)
    opts.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if device == "cuda" else AcceleratorDevice.CPU)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def convert_pages(converter, pdf, first, last, out_path, force):
    if out_path.exists() and not force:
        return "cached", 0.0
    t = time.perf_counter()
    result = converter.convert(str(pdf), page_range=(first, last), raises_on_error=False)
    elapsed = time.perf_counter() - t
    if result.document is None or not result.document.pages:
        return f"FAILED ({result.status})", elapsed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.document.save_as_json(out_path)
    return str(result.status.value), elapsed


def parse_report(key, rep, device, force, converters):
    out = report_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    pdf = pdf_path(rep)
    started = time.perf_counter()

    records = pymupdf_pass(key, rep)
    with (out / "pages.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_pages = len(records)
    ocr_pages = [r["pdf_page"] for r in records
                 if r["broken_number_hits"] >= BROKEN_NUMBER_MIN_HITS
                 or r["garbled_glyph_hits"] >= GARBLED_GLYPH_MIN_HITS]
    print(f"[{key}] {n_pages} pages | spreads {sum(r['is_spread'] for r in records)} | "
          f"damaged text layer on {len(ocr_pages)} pages", flush=True)

    batches = []
    for first in range(1, n_pages + 1, DOCLING_BATCH_PAGES):
        last = min(first + DOCLING_BATCH_PAGES - 1, n_pages)
        status, dt = convert_pages(converters("layout"), pdf, first, last,
                                   out / "docling" / f"pages_{first:04d}_{last:04d}.json", force)
        batches.append({"first": first, "last": last, "status": status, "seconds": round(dt, 1)})
        print(f"[{key}] docling pages {first}-{last}: {status} {dt:.1f}s", flush=True)

    ocr_results = []
    for pno in ocr_pages:
        status, dt = convert_pages(converters("ocr"), pdf, pno, pno, out / "ocr" / f"p{pno:04d}.json", force)
        ocr_results.append({"page": pno, "status": status, "seconds": round(dt, 1)})
        print(f"[{key}] OCR page {pno}: {status} {dt:.1f}s", flush=True)

    import docling
    manifest = {
        "key": key,
        **{k: v for k, v in rep.items() if k not in ("parts", "part_renames")},
        "pdf": str(pdf),
        "pages": n_pages,
        "ocr_pages": ocr_pages,
        "docling_batches": batches,
        "ocr_results": ocr_results,
        "device": device,
        "docling_version": getattr(docling, "__version__", "unknown"),
        "pymupdf_version": fitz.VersionBind,
        "parsed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{key}] done in {manifest['total_seconds']}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="*", default=list(REPORTS), choices=list(REPORTS))
    ap.add_argument("--force", action="store_true", help="re-run Docling even if output exists")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Docling device: {device}")

    cache = {}

    def converters(kind):  # build lazily: OCR models load only if a report needs them
        if kind not in cache:
            cache[kind] = make_converter(device, ocr=(kind == "ocr"))
        return cache[kind]

    for key in args.reports:
        parse_report(key, REPORTS[key], device, args.force, converters)


if __name__ == "__main__":
    main()
