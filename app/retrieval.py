"""Turn a question into numbered sources: matching facts first, then report passages."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass

COMPANIES = {  # canonical name -> how people write it
    "TCS": re.compile(r"\btcs\b|tata consultancy", re.I),
    "Infosys": re.compile(r"\binfosys\b|\binfy\b", re.I),
    "HCLTech": re.compile(r"\bhcl\s*tech\b|\bhcl\s+technologies\b|\bhcl\b", re.I),
}
METRICS = [  # first match wins, so the more specific phrases come first
    ("operating_margin", re.compile(r"operating (profit )?margin|ebit margin|\bebit\b", re.I)),
    ("net_profit", re.compile(r"net profit|profit after tax|\bpat\b|net income|profit for the year", re.I)),
    ("attrition", re.compile(r"attrition|turnover", re.I)),
    ("headcount", re.compile(r"head\s?count|employees|employee (count|strength)|workforce|staff strength", re.I)),
    ("revenue", re.compile(r"revenue|\bsales\b|top[- ]line", re.I)),
]
STOPWORDS = frozenset("""
a about all an and any are as at be been by can compare comparison companies company describe describes
did do does each for from give has have how in is it its me much many of on or please report reports
say says show tell than that the their them these this those three to vs versus was were what when where which
who why with year years annual fiscal fy tcs tata consultancy services infosys infy hcl hcltech technologies ltd limited
""".split())
PER_COMPANY = 4   # passages kept per company
MAX_FACTS = 6


@dataclass
class Understood:
    companies: list[str]      # named companies, or all three when none is named
    metric: str | None        # canonical metric id from the question, if any
    fy: str | None            # 'FY2025-26'
    named: bool               # True if the question named at least one company

    def as_dict(self) -> dict:
        return asdict(self)


def _fy(end_year: int) -> str:
    return f"FY{end_year - 1}-{end_year % 100:02d}"


def fiscal_year(text: str) -> str | None:
    """'FY26', 'FY 2026', '2025-26', 'FY2025-26', 'fiscal 2026' -> 'FY2025-26' (Indian FY ends 31 March)."""
    t = text.lower()
    m = re.search(r"(?<!\d)(20\d{2})\s*[-–/]\s*(\d{2}|20\d{2})(?!\d)", t)
    if m:
        start, end = int(m[1]), int(m[2])
        end = end if end > 100 else 2000 + end
        if end == start + 1:
            return _fy(end)
    m = re.search(r"\bfy\s*'?(20\d{2}|\d{2})(?!\d)", t)
    if m:
        end = int(m[1])
        return _fy(end if end > 100 else 2000 + end)
    m = re.search(r"\bfiscal\s+(?:year\s+)?(20\d{2})\b", t)
    if m:
        return _fy(int(m[1]))
    return None


def mentions_fy(text: str, fy: str) -> bool:
    """Does the text mention this fiscal year in any common spelling?"""
    start = int(fy[2:6])
    end = start + 1
    yy = end % 100
    pattern = (rf"{start}\s*[-–/]\s*(?:20)?{yy:02d}(?!\d)|\bfy\s*'?(?:20)?{yy:02d}(?!\d)|fiscal\s+(?:year\s+)?{end}"
               rf"|march\s*31,?\s*{end}|31(?:st)?\s*march,?\s*{end}")
    return re.search(pattern, text, re.I) is not None


def understand(question: str) -> Understood:
    named = [name for name, pattern in COMPANIES.items() if pattern.search(question)]
    metric = next((mid for mid, pattern in METRICS if pattern.search(question)), None)
    return Understood(companies=named or list(COMPANIES), metric=metric, fy=fiscal_year(question), named=bool(named))


def fts_query(question: str) -> str:
    """A safe FTS5 query: the question's content words, quoted and OR-ed (numbers and FY tokens dropped)."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    keep = [w for w in words if len(w) > 1 and w not in STOPWORDS and not w.isdigit() and not re.fullmatch(r"fy\d+", w)]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(keep))


def search_passages(conn: sqlite3.Connection, question: str, companies: list[str]) -> list[dict]:
    query = fts_query(question)
    if not query:
        return []
    hits: list[dict] = []
    for company in companies:
        rows = conn.execute(
            """SELECT p.passage_id, p.doc_id, p.company, p.page_no, p.kind, p.element_id, p.section, p.text,
                      d.title, d.file, bm25(passages_fts, 1.0, 0.5) AS bm25
               FROM passages_fts
               JOIN passages p ON p.passage_id = passages_fts.rowid
               JOIN documents d ON d.doc_id = p.doc_id
               WHERE passages_fts MATCH ? AND p.company = ?
               ORDER BY bm25 LIMIT ?""",
            (query, company, PER_COMPANY),
        ).fetchall()
        hits.extend(dict(r) | {"score": round(-r["bm25"], 3)} for r in rows)  # bm25() is lower-is-better
    return sorted(hits, key=lambda h: -h["score"])


def lookup_facts(conn: sqlite3.Connection, companies: list[str], metric: str | None, fy: str | None) -> list[dict]:
    """The same fact the chart uses for each company x metric x FY x unit (view best_facts in extract/schema.sql:
    consolidated first, then checked status, most authoritative source, most agreement)."""
    if not metric:
        return []
    marks = ",".join("?" * len(companies))
    rows = conn.execute(
        f"""SELECT f.*, d.title, d.file FROM best_facts f JOIN documents d ON d.doc_id = f.doc_id
            WHERE f.metric = ? AND f.company IN ({marks})""" + (" AND f.fy = ?" if fy else ""),
        (metric, *companies, *([fy] if fy else [])),
    ).fetchall()
    chosen = list(rows)
    if not fy:  # no year asked: keep the two latest years per company
        latest = {c: sorted({r["fy"] for r in chosen if r["company"] == c})[-2:] for c in companies}
        chosen = [r for r in chosen if r["fy"] in latest[r["company"]]]
    order = {c: i for i, c in enumerate(companies)}
    chosen.sort(key=lambda r: (order[r["company"]], r["fy"]))
    return [dict(r) for r in chosen[:MAX_FACTS]]


def _fact_text(f: dict) -> str:
    return f"{f['company']} {f['metric'].replace('_', ' ')} {f['fy']} ({f['unit']}, {f['basis']}): {f['quote']}"


def build_sources(conn: sqlite3.Connection, question: str) -> tuple[Understood, list[dict]]:
    understood = understand(question)
    facts = lookup_facts(conn, understood.companies, understood.metric, understood.fy)
    passages = search_passages(conn, question, understood.companies)
    sources: list[dict] = []
    for f in facts:
        sources.append({
            "kind": "fact", "company": f["company"], "report": f["title"], "pdf": f["file"], "doc_id": f["doc_id"],
            "page": f["page_no"], "text": _fact_text(f), "score": None, "fact_id": f["fact_id"],
            "bbox": [f["x0"], f["y0"], f["x1"], f["y1"]] if f["x0"] is not None else None,
        })
    for p in passages:
        sources.append({
            "kind": p["kind"], "company": p["company"], "report": p["title"], "pdf": p["file"], "doc_id": p["doc_id"],
            "page": p["page_no"], "text": " ".join(p["text"].split()), "score": p["score"], "passage_id": p["passage_id"],
            "bbox": None,
        })
    for n, source in enumerate(sources, start=1):
        source["n"] = n
    return understood, sources
