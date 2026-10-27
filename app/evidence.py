"""Evidence cards for the UI: a cited fact or passage with a mini table around the cited row, and search snippets."""
from __future__ import annotations

import sqlite3

from app.retrieval import fts_query
from extract.numbers import fy_from_header

AROUND = 2           # table rows shown above and below the cited row
MAX_VALUE_COLS = 4   # year/value columns shown next to the row label


def _column_name(label: str | None) -> str:
    """'(In ` crore) - Year ended March 31, - 2026' -> 'FY2025-26'; keep 'Standalone'/'Consolidated' if present."""
    if not label:
        return ""
    if fy := fy_from_header(label):
        basis = next((b for b in ("Standalone", "Consolidated") if b.lower() in label.lower()), "")
        return f"{basis} {fy}".strip()
    return label.split(" - ")[-1]


def table_snippet(conn: sqlite3.Connection, element_id: str, focus_row: int | None = None,
                  focus_col: int | None = None) -> dict | None:
    cells = conn.execute("SELECT row, col, is_header, row_label, col_label, text FROM table_cells "
                         "WHERE element_id = ? ORDER BY row, col", (element_id,)).fetchall()
    body = [c for c in cells if not c["is_header"]]
    if not body:
        return None
    text_at = {(c["row"], c["col"]): c["text"] for c in body}
    label_of: dict[int, str] = {}
    col_label: dict[int, str] = {}
    for c in body:
        label_of.setdefault(c["row"], c["row_label"] or "")
        col_label.setdefault(c["col"], c["col_label"] or "")
    label_col = min(col_label)
    value_cols = [k for k in sorted(col_label) if fy_from_header(col_label[k])] or \
                 [k for k in sorted(col_label) if k != label_col]
    if focus_col in value_cols:  # keep the cited column in view
        i = value_cols.index(focus_col)
        start = max(0, min(i - MAX_VALUE_COLS // 2, len(value_cols) - MAX_VALUE_COLS))
        value_cols = value_cols[start:start + MAX_VALUE_COLS]
    else:
        value_cols = value_cols[:MAX_VALUE_COLS]
    rows = sorted({r for (r, k) in text_at if k in value_cols})
    if focus_row in rows:
        i = rows.index(focus_row)
        rows = rows[max(0, i - AROUND):i + AROUND + 1]
    else:
        rows = rows[:2 * AROUND + 1]
    return {
        "columns": ["Particulars"] + [_column_name(col_label[k]) for k in value_cols],
        "rows": [{"cells": [label_of.get(r, "").split(" > ")[-1]] + [text_at.get((r, k), "") for k in value_cols],
                  "highlight": r == focus_row} for r in rows],
        "highlight_col": value_cols.index(focus_col) + 1 if focus_col in value_cols else None,
    }


def fact_evidence(conn: sqlite3.Connection, fact_id: int) -> dict | None:
    f = conn.execute("""SELECT f.*, d.title, d.file, e.text AS context, e.section
                        FROM facts f JOIN documents d ON d.doc_id = f.doc_id
                        LEFT JOIN elements e ON e.element_id = f.element_id WHERE f.fact_id = ?""", (fact_id,)).fetchone()
    if f is None:
        return None
    from_table = f["source_kind"] == "table"
    return {
        "kind": "fact", "fact_id": f["fact_id"], "company": f["company"], "report": f["title"], "pdf": f["file"],
        "doc_id": f["doc_id"], "page": f["page_no"], "bbox": [f["x0"], f["y0"], f["x1"], f["y1"]],
        "title": ((f["context"] or "").split(" | ")[0] if from_table else f["section"]) or "",
        "context": f["context"] if from_table else None, "quote": f["quote"], "status": f["status"],
        "metric": f["metric"], "fy": f["fy"], "value": f["value"], "unit": f["unit"], "basis": f["basis"],
        "table": table_snippet(conn, f["element_id"], f["cell_row"], f["cell_col"]) if from_table else None,
    }


def passage_evidence(conn: sqlite3.Connection, passage_id: int) -> dict | None:
    p = conn.execute("""SELECT p.*, d.title, d.file FROM passages p JOIN documents d ON d.doc_id = p.doc_id
                        WHERE p.passage_id = ?""", (passage_id,)).fetchone()
    if p is None:
        return None
    box = conn.execute("SELECT x0, y0, x1, y1 FROM elements WHERE element_id = ?", (p["element_id"],)).fetchone() \
        if p["element_id"] else None
    return {
        "kind": p["kind"], "passage_id": p["passage_id"], "company": p["company"], "report": p["title"],
        "pdf": p["file"], "doc_id": p["doc_id"], "page": p["page_no"], "bbox": list(box) if box else None,
        "title": p["section"] or "", "quote": " ".join(p["text"].split())[:600],
        "table": table_snippet(conn, p["element_id"]) if p["kind"] == "table" and p["element_id"] else None,
    }


def search_snippets(conn: sqlite3.Connection, q: str, company: str | None = None, limit: int = 25) -> list[dict]:
    query = fts_query(q)
    if not query:
        return []
    sql = """SELECT p.passage_id, p.company, p.doc_id, p.page_no AS page, p.kind, p.section,
                    d.title AS report, d.file AS pdf,
                    snippet(passages_fts, 0, '[', ']', '…', 24) AS snippet, bm25(passages_fts, 1.0, 0.5) AS bm25
             FROM passages_fts JOIN passages p ON p.passage_id = passages_fts.rowid
             JOIN documents d ON d.doc_id = p.doc_id
             WHERE passages_fts MATCH ?""" + (" AND p.company = ?" if company else "") + " ORDER BY bm25 LIMIT ?"
    params = (query, company, limit) if company else (query, limit)
    return [{k: r[k] for k in r.keys() if k != "bm25"} | {"score": round(-r["bm25"], 3)}
            for r in conn.execute(sql, params).fetchall()]
