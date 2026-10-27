"""Build data/facts.db from the Docling output of each report (one-time, offline, about a minute).

    python -m extract.build_facts             # reads data/sources.json, data/extracted/, data/reports/
    python -m extract.build_facts --summary   # also print the facts found per company, metric and year

Writes a temporary file and swaps it in, so a running backend never sees a half-built database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
import yaml

from extract.numbers import detect_unit, fy_from_header, normalise_label, parse_fy, parse_number, printed_number

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
EXTRACTOR = "docling 2.143.0 (layout heron, TableFormer accurate, no OCR)"
PASSAGE_CHARS = 1200
TABLE_ROWS_PER_PASSAGE = 15
TEXT_KINDS = {"section_header", "title", "text", "list_item", "caption", "footnote"}
ROW_MARKER = re.compile(r"^\(?(?:[ivxlc]+|[a-z]|\d{1,2})\)?\.?$", re.IGNORECASE)  # 'I', '(a)', '1.' before a label
NUMERIC_ISH = re.compile(r"^[\d\s,.()%+-]+$")  # an unreadable number ('13.', '97 5') is not part of a row label
NOTE_COLUMN = re.compile(r"\bnotes?\b", re.IGNORECASE)
STATEMENT = re.compile(r"statement of profit and loss|balance sheet|statement of cash flows?", re.IGNORECASE)
HIGHLIGHTS = re.compile(r"highlight|key trends|performance overview|at a glance|key (?:financial )?(?:metrics|figures)"
                        r"|financial (?:performance|summary)|five[- ]year|ten[- ]year", re.IGNORECASE)


@dataclass
class Rule:
    metric: str
    row: list[re.Pattern]
    not_row: list[re.Pattern] = field(default_factory=list)
    context: re.Pattern | None = None
    column: re.Pattern | None = None
    units: tuple[str, ...] = ()
    assume_unit: str | None = None
    low: float = float("-inf")
    high: float = float("inf")
    default_fy: bool = False

    def matches_row(self, label: str, path: str) -> bool:
        """Patterns test the row's own label; only patterns written with ' > ' test 'section > label'."""
        if not any(p.search(label) or (" > " in p.pattern and p.search(path)) for p in self.row):
            return False
        return not any(p.search(label) or p.search(path) for p in self.not_row)

    def accept_unit(self, unit: str | None, source: str | None) -> str | None:
        """The unit to store for a value, or None if this rule cannot use it."""
        if unit in self.units:
            return unit
        weak = unit is None or source == "context"  # only the table's surroundings named a unit
        if weak and "count" in self.units:
            return "count"
        if weak and self.assume_unit:
            return self.assume_unit
        return None


@dataclass
class TextRule:
    metric: str
    pattern: re.Pattern
    not_text: list[re.Pattern]
    unit: str
    low: float
    high: float


def load_rules(path: Path) -> tuple[dict, list[Rule], list[TextRule]]:
    spec = yaml.safe_load(path.read_text())
    rx = lambda p: re.compile(p, re.IGNORECASE)  # noqa: E731
    rules = []
    for r in spec["rules"]:
        low, high = r.get("range", (float("-inf"), float("inf")))
        rules.append(Rule(metric=r["metric"], row=[rx(p) for p in r["row"]], not_row=[rx(p) for p in r.get("not_row", [])],
                          context=rx(r["context"]) if r.get("context") else None,
                          column=rx(r["column"]) if r.get("column") else None,
                          units=tuple(r.get("units", [])), assume_unit=r.get("assume_unit"),
                          low=low, high=high, default_fy=r.get("default_fy", False)))
    text_rules = [TextRule(metric=r["metric"], pattern=rx(r["pattern"]), not_text=[rx(p) for p in r.get("not_text", [])],
                           unit=r["unit"], low=r["range"][0], high=r["range"][1]) for r in spec.get("text_rules", [])]
    return spec["metrics"], rules, text_rules


def page_quality(text: str) -> str:
    if len(text.strip()) < 200:
        return "image_only"
    words = re.findall(r"[A-Za-z]+", text)
    single = sum(1 for w in words if len(w) == 1)
    return "letter_spaced" if len(words) > 30 and single / len(words) > 0.25 else "ok"


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def top_left(bbox: dict, page_height: float) -> tuple[float, float, float, float]:
    """Docling boxes are bottom-left origin for items, top-left for cells; store top-left (PyMuPDF)."""
    if bbox.get("coord_origin", "BOTTOMLEFT") == "BOTTOMLEFT":
        return bbox["l"], page_height - bbox["t"], bbox["r"], page_height - bbox["b"]
    return bbox["l"], bbox["t"], bbox["r"], bbox["b"]


def basis_of(*texts: str) -> str:
    for text in texts:
        t = (text or "").lower()
        if "standalone" in t:
            return "standalone"
        if "consolidated" in t:
            return "consolidated"
    return "unknown"


def walk(doc: dict):
    """Yield (kind, item) for every body item in Docling's reading order."""
    stack = [c["$ref"] for c in reversed(doc["body"]["children"])]
    while stack:
        _, kind, idx = stack.pop().split("/")
        item = doc[kind][int(idx)]
        if kind != "groups":
            yield kind, item
        stack.extend(c["$ref"] for c in reversed(item.get("children", [])))


def is_header_row(cells: list[dict]) -> bool:
    if all(c["column_header"] for c in cells):
        return True
    years = sum(1 for c in cells if parse_fy(c["text"]))
    values = sum(1 for c in cells if parse_number(c["text"]) and not parse_fy(c["text"]))
    return years >= 2 and values == 0


def resolve_unit(cell_text: str, is_percent: bool, col_label: str, row_label: str,
                 running: tuple[str | None, str | None]) -> tuple[str | None, str | None]:
    """(unit, where it came from): the cell itself, its column, its row, else the table's running unit."""
    if is_percent:
        return "%", "cell"
    for text, source in ((cell_text, "cell"), (col_label, "column"), (row_label, "row")):
        if unit := detect_unit(text):
            return unit, source
    return running


@dataclass
class Builder:
    con: sqlite3.Connection
    metrics: dict
    rules: list[Rule]
    text_rules: list[TextRule]
    facts: list[dict] = field(default_factory=list)

    # ---- documents and pages -------------------------------------------------------------------------------
    def add_document(self, src: dict) -> dict[int, dict]:
        pdf_path = DATA / "reports" / src["file"]
        pdf = pymupdf.open(pdf_path)
        sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        self.con.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)",
                         (src["doc_id"], src["company"], src["title"], src["fy"], src["file"], sha, pdf.page_count,
                          src.get("url"), src.get("downloaded")))
        pages = {}
        for page in pdf:
            text = page.get_text("text")
            pages[page.number + 1] = {"text": text, "compact": compact(text), "height": page.rect.height,
                                      "quality": page_quality(text)}
            self.con.execute("INSERT INTO pages VALUES (?,?,?,?,?,?)",
                             (src["doc_id"], page.number + 1, page.rect.width, page.rect.height, text,
                              pages[page.number + 1]["quality"]))
        return pages

    # ---- one document --------------------------------------------------------------------------------------
    def add_docling(self, src: dict, doc: dict, pages: dict[int, dict]) -> None:
        doc_id, company = src["doc_id"], src["company"]
        section = ""
        recent: list[tuple[int, str]] = []  # (page, text) of the last text items, for table context
        texts_by_ref = {t["self_ref"]: t for t in doc["texts"]}
        passages = Passages(self.con, doc_id, company)
        for seq, (kind, item) in enumerate(walk(doc)):
            if not item.get("prov"):
                continue
            prov = item["prov"][0]
            page_no = prov["page_no"]
            x0, y0, x1, y1 = top_left(prov["bbox"], pages[page_no]["height"])
            element_id = f"{doc_id}{item['self_ref']}"
            label = item.get("label", kind)
            extra = None
            if kind == "texts":
                text = (item.get("text") or "").strip()
                if label == "section_header":
                    section = text
                    passages.flush()
                recent = (recent + [(page_no, text)])[-3:]
                if label in TEXT_KINDS and text and pages[page_no]["quality"] != "letter_spaced":
                    passages.add(page_no, section, text, element_id, bullet=label == "list_item")
                if label in TEXT_KINDS and text:
                    self.match_text_facts(src, element_id, page_no, pages[page_no], text, (x0, y0, x1, y1))
            elif kind == "tables":
                captions = " ".join(texts_by_ref[c["$ref"]]["text"] for c in item.get("captions", [])
                                    if c["$ref"] in texts_by_ref)
                above = [t for p, t in recent if p == page_no][-2:]
                context = " | ".join(x for x in [section, *above, captions] if x)
                text = context
                extra = json.dumps({"rows": item["data"]["num_rows"], "cols": item["data"]["num_cols"]})
                self.add_table(src, item, element_id, page_no, section, context, pages[page_no], passages)
            else:  # pictures
                classes = (item.get("meta") or {}).get("classification", {}).get("predictions") or []
                best = max(classes, key=lambda c: c["confidence"], default=None)
                text = best["class_name"] if best else None
                extra = json.dumps({"class": text, "confidence": round(best["confidence"], 3)} if best else {})
            self.con.execute("INSERT INTO elements VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (element_id, doc_id, page_no, seq, "table" if kind == "tables" else
                              "picture" if kind == "pictures" else label, x0, y0, x1, y1, section, text, extra))
        passages.flush()
        # Letter-spaced pages (TCS design pages): Docling returns single letters, so index PyMuPDF's text instead.
        for page_no, page in pages.items():
            if page["quality"] == "letter_spaced":
                passages.add_raw(page_no, "", page["text"])

    # ---- one table -----------------------------------------------------------------------------------------
    def add_table(self, src: dict, item: dict, element_id: str, page_no: int, section: str, context: str,
                  page: dict, passages: Passages) -> None:
        rows: dict[int, list[dict]] = defaultdict(list)
        for cell in item["data"]["table_cells"]:
            rows[cell["start_row_offset_idx"]].append(cell)
        running = (detect_unit(context, allow_percent=False), "context")
        rank = 1 if STATEMENT.search(context) else 2 if HIGHLIGHTS.search(context) else 3
        block: dict[int, list[str]] = defaultdict(list)
        in_block = False
        col_label: dict[int, str] = {}
        col_fy: dict[int, str | None] = {}
        note_cols: set[int] = set()
        row_section = ""
        lines: list[str] = []
        seen_values = False
        for r in sorted(rows):
            cells = sorted(rows[r], key=lambda c: c["start_col_offset_idx"])
            no_values = not any(parse_number(c["text"]) for c in cells if c["start_col_offset_idx"] > 0)
            if is_header_row(cells) or (not seen_values and no_values):  # rows above the first values are headers
                if not in_block:
                    block, in_block = defaultdict(list), True
                labelled = [c for c in cells if c["text"].strip()]
                for i, c in enumerate(labelled):
                    text = c["text"].strip()
                    # a header label covers the columns to its right until the next label ('Consolidated' over
                    # its two year columns), even when the cell itself was not given a column span
                    stop = max(c["end_col_offset_idx"],
                               labelled[i + 1]["start_col_offset_idx"] if i + 1 < len(labelled) else 0)
                    for k in range(c["start_col_offset_idx"], stop):
                        if text not in block[k]:
                            block[k].append(text)
                for c in cells:
                    self.add_cell(element_id, c, page["height"], is_header=1)
                if unit := detect_unit(" ".join(c["text"] for c in cells), allow_percent=False):
                    running = (unit, "header")
                continue
            if in_block:
                col_label = {k: " - ".join(v) for k, v in block.items()}
                col_fy = {k: fy_from_header(v) for k, v in col_label.items()}
                note_cols = {k for k, v in col_label.items() if NOTE_COLUMN.search(v)}
                in_block = False

            label_parts, values = [], []
            for c in cells:
                k, text = c["start_col_offset_idx"], c["text"].strip()
                parsed = None if (k in note_cols or k == 0) else parse_number(text)
                if parsed is None:
                    in_year_column = bool(col_fy.get(k))  # 'NA' or '-' in a year column is not label text
                    if text and not in_year_column and not ROW_MARKER.match(text) and not NUMERIC_ISH.match(text):
                        label_parts.append(text)
                else:
                    values.append((c, parsed))
            label = " ".join(label_parts)
            seen_values = seen_values or bool(values)
            if label and not values:
                row_section = label
                for c in cells:
                    self.add_cell(element_id, c, page["height"], row_label=label)
                continue
            path = f"{row_section} > {label}" if row_section and label else label or row_section
            norm_label, norm_path = normalise_label(label), normalise_label(path)
            printed = []
            for c in cells:
                k = c["start_col_offset_idx"]
                match = next((v for cc, v in values if cc is c), None)
                if match is None:
                    self.add_cell(element_id, c, page["height"], row_label=path, col_label=col_label.get(k))
                    continue
                value, is_percent = match
                unit, source = resolve_unit(c["text"], is_percent, col_label.get(k, ""), label, running)
                fy = col_fy.get(k)
                self.add_cell(element_id, c, page["height"], row_label=path, col_label=col_label.get(k),
                              value=value, unit=unit, fy=fy)
                printed.append(f"{col_label.get(k) or 'value'}: {c['text'].strip()}")
                if label:
                    self.match_fact(src, element_id, page_no, page, c, value, unit, source, fy, label, norm_label,
                                    norm_path, col_label.get(k, ""), context, rank, any(col_fy.values()))
            if printed:
                lines.append(f"{path or '(row)'} | " + "; ".join(printed))
        passages.add_table(page_no, section, element_id, context, lines)

    def add_cell(self, element_id: str, c: dict, page_height: float, is_header: int = 0, row_label: str | None = None,
                 col_label: str | None = None, value: float | None = None, unit: str | None = None,
                 fy: str | None = None) -> None:
        x0, y0, x1, y1 = top_left(c["bbox"], page_height)
        self.con.execute("INSERT OR IGNORE INTO table_cells VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (element_id, c["start_row_offset_idx"], c["start_col_offset_idx"], c["row_span"],
                          c["col_span"], is_header, row_label, col_label, c["text"], value, unit, fy, x0, y0, x1, y1))

    def match_fact(self, src: dict, element_id: str, page_no: int, page: dict, c: dict, value: float,
                   unit: str | None, source: str | None, fy: str | None, label: str, norm_label: str,
                   norm_path: str, col_label: str, context: str, rank: int, table_has_years: bool) -> None:
        for rule in self.rules:
            if not rule.matches_row(norm_label, norm_path):
                continue
            if rule.context and not rule.context.search(context):
                continue
            if rule.column and not rule.column.search(col_label):
                continue
            fact_unit = rule.accept_unit(unit, source)
            if fact_unit is None or not rule.low <= value <= rule.high:
                continue
            fact_fy = fy or (src["fy"] if rule.default_fy and not table_has_years else None)
            if not fact_fy:
                continue
            number = printed_number(c["text"])
            x0, y0, x1, y1 = top_left(c["bbox"], page["height"])
            self.facts.append({
                "company": src["company"], "metric": rule.metric, "fy": fact_fy, "dimension": None, "value": value,
                "unit": fact_unit, "basis": basis_of(col_label, context), "label_in_report": label,
                "source_kind": "table", "source_rank": rank, "element_id": element_id,
                "cell_row": c["start_row_offset_idx"], "cell_col": c["start_col_offset_idx"], "doc_id": src["doc_id"],
                "page_no": page_no, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "quote": f"{label} | {col_label or 'value'}: {c['text'].strip()}",
                "in_text_layer": int(bool(number) and compact(number) in page["compact"]),
                "agrees_with": 0, "conflicts_with": 0, "status": "", "verified_by": None, "extractor": EXTRACTOR})
            return

    def match_text_facts(self, src: dict, element_id: str, page_no: int, page: dict, text: str,
                         box: tuple[float, float, float, float]) -> None:
        for rule in self.text_rules:
            for match in rule.pattern.finditer(text):
                start = text.rfind(". ", 0, match.start()) + 1
                end = text.find(". ", match.end())
                sentence = text[start:len(text) if end < 0 else end + 1].strip()
                value = float(match.group("value"))
                if any(p.search(sentence) for p in rule.not_text) or not rule.low <= value <= rule.high:
                    continue
                self.facts.append({
                    "company": src["company"], "metric": rule.metric, "fy": parse_fy(sentence) or src["fy"],
                    "dimension": None, "value": value, "unit": rule.unit, "basis": basis_of(sentence),
                    "label_in_report": match.group(0)[:80], "source_kind": "text", "source_rank": 4,
                    "element_id": element_id, "cell_row": None, "cell_col": None, "doc_id": src["doc_id"],
                    "page_no": page_no, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                    "quote": sentence[:300], "in_text_layer": int(match.group("value") in page["compact"]),
                    "agrees_with": 0, "conflicts_with": 0, "status": "", "verified_by": None, "extractor": EXTRACTOR})
                break  # one value per rule and text item

    # ---- cross-checks --------------------------------------------------------------------------------------
    def check_and_store_facts(self) -> None:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for f in self.facts:
            groups[(f["company"], f["metric"], f["fy"], f["unit"], f["dimension"])].append(f)
        for group in groups.values():
            for f in group:
                for other in group:
                    if other is f or other["element_id"] == f["element_id"]:
                        continue
                    if "unknown" not in (f["basis"], other["basis"]) and f["basis"] != other["basis"]:
                        continue  # consolidated and standalone figures are allowed to differ
                    tolerance = 0.1 if f["unit"] == "%" else 0.005 * max(abs(f["value"]), abs(other["value"]))
                    if abs(f["value"] - other["value"]) <= max(tolerance, 0.5 if f["unit"] != "%" else 0):
                        f["agrees_with"] += 1
                    else:
                        f["conflicts_with"] += 1
                f["status"] = ("not_in_text" if not f["in_text_layer"] else "corroborated" if f["agrees_with"]
                               else "conflict" if f["conflicts_with"] else "single_source")
        columns = list(self.facts[0].keys()) if self.facts else []
        self.con.executemany(f"INSERT INTO facts ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                             [tuple(f[c] for c in columns) for f in self.facts])


@dataclass
class Passages:
    """Groups text items under their heading into ~1,200-character passages; tables become row lines."""
    con: sqlite3.Connection
    doc_id: str
    company: str
    page_no: int = 0
    section: str = ""
    element_id: str | None = None
    parts: list[str] = field(default_factory=list)

    def add(self, page_no: int, section: str, text: str, element_id: str, bullet: bool = False) -> None:
        if page_no != self.page_no or section != self.section or sum(map(len, self.parts)) > PASSAGE_CHARS:
            self.flush()
            self.page_no, self.section, self.element_id = page_no, section, element_id
        self.parts.append(("- " if bullet else "") + text)

    def flush(self) -> None:
        if self.parts:
            body = "\n".join(self.parts)
            self.insert(self.page_no, "text", self.element_id, self.section,
                        f"{self.section}\n{body}" if self.section and body != self.section else body)
        self.parts = []

    def add_raw(self, page_no: int, section: str, text: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        for start in range(0, len(text), PASSAGE_CHARS):
            self.insert(page_no, "text", None, section, text[start:start + PASSAGE_CHARS])

    def add_table(self, page_no: int, section: str, element_id: str, context: str, lines: list[str]) -> None:
        for start in range(0, len(lines), TABLE_ROWS_PER_PASSAGE):
            head = f"Table: {context}" if context else "Table"
            self.insert(page_no, "table", element_id, section,
                        "\n".join([head] + lines[start:start + TABLE_ROWS_PER_PASSAGE]))

    def insert(self, page_no: int, kind: str, element_id: str | None, section: str, text: str) -> None:
        self.con.execute("INSERT INTO passages (doc_id, company, page_no, kind, element_id, section, text) "
                         "VALUES (?,?,?,?,?,?,?)", (self.doc_id, self.company, page_no, kind, element_id, section, text))


def build(out: Path) -> Path:
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript((HERE / "schema.sql").read_text())
    metrics, rules, text_rules = load_rules(HERE / "metrics.yaml")
    raw = yaml.safe_load((HERE / "metrics.yaml").read_text())
    con.executemany("INSERT INTO metric_dictionary VALUES (?,?,?,?)",
                    [(m, spec["label"], spec["unit"],
                      json.dumps([r for r in raw["rules"] + raw.get("text_rules", []) if r["metric"] == m]))
                     for m, spec in metrics.items()])
    builder = Builder(con, metrics, rules, text_rules)
    for src in json.loads((DATA / "sources.json").read_text()):
        pages = builder.add_document(src)
        doc = json.loads((DATA / "extracted" / f"{src['doc_id']}.docling.json").read_text())
        builder.add_docling(src, doc, pages)
    builder.check_and_store_facts()
    con.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
    con.commit()
    con.close()
    os.replace(tmp, out)
    return out


def summary(db: Path) -> None:
    con = sqlite3.connect(db)
    for table in ("documents", "pages", "elements", "table_cells", "facts", "passages"):
        print(f"{table:12} {con.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]:>8}")
    print("\nbest_facts (the value the chart and the Q&A use):")
    rows = con.execute("""
        SELECT company, metric, fy, value, unit, basis, page_no, status, agrees_with, conflicts_with, label_in_report
        FROM best_facts WHERE fy >= 'FY2024-25' ORDER BY metric, company, fy""").fetchall()
    for r in rows:
        print(f"  {r[1]:16} {r[0]:8} {r[2]:10} {r[3]:>12,.2f} {r[4]:11} {r[5]:12} p.{r[6]:<4} {r[7]:13} "
              f"+{r[8]}/-{r[9]}  '{r[10][:50]}'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DATA / "facts.db")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    db = build(args.out)
    print(f"wrote {db} ({db.stat().st_size / 1e6:.1f} MB)")
    if args.summary:
        summary(db)


if __name__ == "__main__":
    main()
