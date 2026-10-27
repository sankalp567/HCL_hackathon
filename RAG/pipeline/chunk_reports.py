"""Step 2: turn parsed reports into retrieval chunks with citation metadata.

Reads data/parsed/<key>/ (from parse_reports.py), writes data/chunks/<key>.jsonl:
  {"id", "text", "embed_text", "metadata"}
    text        chunk text shown to the LLM / user (₹ symbol repaired)
    embed_text  text that gets embedded: company, FY, report part and headings put in front,
                so "TCS attrition" can match a passage that only says "attrition"
    metadata    chunk_id, company, fiscal_year, pages ("167" or "166-167", 1-based PDF pages),
                chunk_type (text | table | mixed), n_tokens

Chunking rules:
  * Docling HybridChunker sized with the embedding model's own tokenizer (CHUNK_MAX_TOKENS)
  * Section-aware: splits on headings; oversized sections are split with the heading kept in each part;
    undersized chunks merge only with neighbours under the same heading (merge_peers)
  * Tables stay whole; oversized tables are split by rows with the header row repeated
  * Pages with a damaged text layer use the OCR parse instead of the main parse
  * Pages where Docling captured < FALLBACK_MIN_RECALL of PyMuPDF's words get extra PyMuPDF chunks

Usage:  .venv-gpu\\Scripts\\python pipeline\\chunk_reports.py [--reports tcs]
"""
import argparse
import bisect
import json
import re
from collections import Counter, defaultdict

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument, TableItem
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import (CHUNK_MAX_TOKENS, CHUNKS_DIR, EMBED_MODEL, FALLBACK_MIN_RECALL, REPORTS,
                    report_dir)

HEADING_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}
VOCAB = re.compile(r"[a-z]{3,}")
# Some TCS fonts extract digits as Coptic letters: U+03EC..U+03F5 -> 0..9 ("ϮϬϮϲ" = "2026")
SHIFTED_DIGITS = str.maketrans({chr(0x03EC + d): str(d) for d in range(10)})


# ---------------------------------------------------------------- text helpers

def fix_rupee(text, glyph):
    """The reports' rupee fonts extract as '`' (Infosys) or 'h' (TCS)."""
    if glyph == "`":
        return text.replace("`", "₹")
    if glyph == "h":
        units = r"(crore|lakh|million|billion|cr)\b"
        text = re.sub(r"\(h\s*" + units, r"(₹ \1", text)
        text = re.sub(r"\bin h " + units, r"in ₹ \1", text)
        text = re.sub(r"\bh\s?(?=\d[\d,]*(?:\.\d+)?\s*" + units + ")", "₹", text)
    return text


def clean(text, glyph):
    text = text.translate(SHIFTED_DIGITS)
    text = re.sub(r"&z(?=\s?\d{2,4}\b)", "FY", text)  # same font: "FY" extracts as "&z"
    text = fix_rupee(text, glyph)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def vocab(text):
    # Drop PyMuPDF's doubled tokens on spreads ("assuranceassurance") so they don't count as missing
    return {w for w in VOCAB.findall(text.lower())
            if not (len(w) % 2 == 0 and w[: len(w) // 2] == w[len(w) // 2:])}


def item_text(item, doc):
    if isinstance(item, TableItem):
        return item.export_to_markdown(doc=doc)
    return getattr(item, "text", "") or ""


# ---------------------------------------------------------------- loading

def load_report(key):
    base = report_dir(key)
    pages = {}
    with (base / "pages.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pages[rec["pdf_page"]] = rec
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    main_docs = [DoclingDocument.load_from_json(p) for p in sorted((base / "docling").glob("*.json"))]
    ocr_docs = {int(p.stem[1:]): DoclingDocument.load_from_json(p) for p in sorted((base / "ocr").glob("p*.json"))}
    return pages, manifest, main_docs, ocr_docs


def heading_index(docs):
    """Sorted (page, heading) for every heading, to give orphan chunks the section they sit in."""
    out = []
    for doc in docs:
        for item, _ in doc.iterate_items():
            if item.label in HEADING_LABELS and item.prov and item.text.strip():
                out.append((item.prov[0].page_no, item.text.strip()))
    return sorted(out, key=lambda x: x[0])  # stable: keeps reading order within a page


def heading_before(index, page):
    i = bisect.bisect_right(index, (page, "￿"))
    return index[i - 1][1] if i else ""


def drop_pages(doc, pages):
    """Remove items that sit only on the given pages (they are replaced by the OCR parse)."""
    doomed = [item for item, _ in doc.iterate_items()
              if item.prov and all(p.page_no in pages for p in item.prov)]
    if doomed:
        doc.delete_items(node_items=doomed)


def page_letters(docs):
    """Per page: Docling's text as one lowercase letters-only string."""
    parts = defaultdict(list)
    for doc in docs:
        for item, _ in doc.iterate_items():
            text = re.sub(r"[^a-z]", "", item_text(item, doc).lower())
            for prov in item.prov or []:
                parts[prov.page_no].append(text)
    return {p: "".join(v) for p, v in parts.items()}


def recall(ref_words, letters):
    # Substring match, so PyMuPDF's glued words ("basegmbh") count as found
    return sum(w in letters for w in ref_words) / len(ref_words)


# ---------------------------------------------------------------- chunking

def make_chunk(key, rep, pages_meta, page_list, headings, body, n_tokens, seq, chunk_type, source):
    meta = {
        "chunk_id": f"{key}-p{page_list[0]:04d}-{seq:03d}",
        "company": rep["company"],
        "fiscal_year": rep["fiscal_year"],
        # 1-based PDF page numbers: "167", or "166-167" when the chunk crosses a page break
        "pages": str(page_list[0]) if len(page_list) == 1 else f"{page_list[0]}-{page_list[-1]}",
        "chunk_type": chunk_type,                 # text | table | mixed (table + surrounding text)
        "n_tokens": n_tokens,
    }
    # Report part and headings go into the embedded text only, not the metadata
    context = f"{rep['company']} ({rep['legal_name']}) annual report {rep['fiscal_year']}"
    part = pages_meta[page_list[0]]["report_part"]
    if part:
        context += f" | {part}"
    path = " > ".join(headings)
    embedded = f"{context}\n{path}\n{body}" if path else f"{context}\n{body}"
    return {"id": meta["chunk_id"], "text": body, "embed_text": embedded, "metadata": meta,
            "_stats": {"first_page": page_list[0], "source": source}}


def chunk_report(key, rep, chunker, splitter, count_tokens):
    pages_meta, manifest, main_docs, ocr_docs = load_report(key)
    glyph = rep.get("rupee_glyph")
    ocr_pages = set(ocr_docs)

    headings_idx = heading_index(main_docs)          # before OCR pages are dropped
    for doc in main_docs:
        drop_pages(doc, ocr_pages)

    sources = [(doc, "docling") for doc in main_docs] + [(ocr_docs[p], "docling_ocr") for p in sorted(ocr_docs)]
    chunks, seq = [], Counter()
    for doc, source in sources:
        for ch in chunker.chunk(dl_doc=doc):
            items = ch.meta.doc_items
            page_list = sorted({p.page_no for it in items for p in (it.prov or [])})
            if not page_list:
                continue
            body = clean(ch.text, glyph)
            if not body:
                continue
            headings = [clean(h, glyph) for h in (ch.meta.headings or [])]
            if not headings:  # batch boundary or OCR page: inherit the section it falls in
                h = heading_before(headings_idx, page_list[0])
                headings = [h] if h else []
            is_table = [it.label == DocItemLabel.TABLE for it in items]
            chunk_type = "table" if all(is_table) else "mixed" if any(is_table) else "text"
            seq[page_list[0]] += 1
            chunks.append(make_chunk(key, rep, pages_meta, page_list, headings, body,
                                     count_tokens(body), seq[page_list[0]], chunk_type, source))

    # Completeness check against PyMuPDF
    docling_letters = page_letters(main_docs + list(ocr_docs.values()))
    fallback_pages = []
    for pno, pm in pages_meta.items():
        if pno in ocr_pages:
            continue
        ref = vocab(pm["text"])
        if len(ref) < 30:
            continue
        r = recall(ref, docling_letters.get(pno, ""))
        if r >= FALLBACK_MIN_RECALL:
            continue
        fallback_pages.append((pno, round(r, 2)))
        h = heading_before(headings_idx, pno)
        headings = [h] if h else []
        for part in splitter.split_text(clean(pm["text"], glyph)):
            seq[pno] += 1
            chunks.append(make_chunk(key, rep, pages_meta, [pno], headings, part,
                                     count_tokens(part), seq[pno], "text", "pymupdf_fallback"))

    chunks.sort(key=lambda c: (c["_stats"]["first_page"], c["id"]))
    return chunks, fallback_pages, sorted(ocr_pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="*", default=list(REPORTS), choices=list(REPORTS))
    args = ap.parse_args()

    hf_tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=CHUNK_MAX_TOKENS),
        merge_peers=True,
        repeat_table_header=True,
    )
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        hf_tok, chunk_size=CHUNK_MAX_TOKENS, chunk_overlap=60)
    count_tokens = lambda s: len(hf_tok.encode(s, add_special_tokens=False))

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for key in args.reports:
        chunks, fallback_pages, ocr_pages = chunk_report(key, REPORTS[key], chunker, splitter, count_tokens)
        with (CHUNKS_DIR / f"{key}.jsonl").open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({k: v for k, v in c.items() if k != "_stats"}, ensure_ascii=False) + "\n")

        tokens = sorted(c["metadata"]["n_tokens"] for c in chunks)
        pages_covered = set()
        for c in chunks:
            first, _, last = c["metadata"]["pages"].partition("-")
            pages_covered.update(range(int(first), int(last or first) + 1))
        print(f"[{key}] {len(chunks)} chunks | pages covered {len(pages_covered)} | "
              f"tokens median {tokens[len(tokens) // 2]} p95 {tokens[int(len(tokens) * 0.95)]} max {tokens[-1]}")
        print(f"   types   {dict(Counter(c['metadata']['chunk_type'] for c in chunks))}")
        print(f"   sources {dict(Counter(c['_stats']['source'] for c in chunks))}")
        print(f"   OCR pages {len(ocr_pages)} | PyMuPDF fallback pages {len(fallback_pages)}: {fallback_pages[:15]}")


if __name__ == "__main__":
    main()
