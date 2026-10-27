"""Turn a question into numbered sources: matching facts first, then report passages."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass

from extract.numbers import format_value

COMPANIES = {  # canonical name -> how people write it
    "TCS": re.compile(r"\btcs\b|tata consultancy", re.I),
    "Infosys": re.compile(r"\binfosys\b|\binfy\b", re.I),
    "HCLTech": re.compile(r"\bhcl\s*tech\b|\bhcl\s+technologies\b|\bhcl\b", re.I),
}
METRICS = [  # first match wins, so the more specific phrases come first
    ("operating_margin", re.compile(r"operating (profit )?margin|ebit margin|\bebit\b", re.I)),
    ("net_profit", re.compile(r"net profit|profit after tax|\bpat\b|net income|profit for the year", re.I)),
    ("attrition", re.compile(r"attrition|turnover", re.I)),
    ("headcount", re.compile(r"head\s?count|employees|employee (head|count|strength)|workforce|staff strength", re.I)),
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


SYNONYMS = {  # the words each report uses for a metric (extract/metrics.yaml), added to the keyword search
    "revenue": ["revenue", "operations"],
    "revenue_usd": ["revenue", "million"],
    "net_profit": ["profit", "attributable", "net"],
    "operating_margin": ["operating", "margin", "ebit"],
    "headcount": ["employees", "headcount", "workforce"],
    "attrition": ["attrition", "turnover", "voluntary"],
}


def expansion(understood: Understood) -> list[str]:
    """Synonyms of the asked metric plus how the reports write the year ('fiscal 2026', 'March 31, 2026')."""
    terms = list(SYNONYMS.get(understood.metric or "", []))
    if understood.fy:
        terms += [str(int(understood.fy[2:6]) + 1), "fiscal"]
    return terms


def fts_query(question: str, extra: list[str] | tuple = ()) -> str:
    """A safe FTS5 query: the question's content words, quoted and OR-ed (numbers and FY tokens dropped),
    plus any extra terms (synonyms, the year)."""
    words = re.findall(r"[a-z0-9]+", question.lower())
    keep = [w for w in words if len(w) > 1 and w not in STOPWORDS and not w.isdigit() and not re.fullmatch(r"fy\d+", w)]
    return " OR ".join(f'"{w}"' for w in dict.fromkeys(keep + [t.lower() for t in extra]))


def search_passages(conn: sqlite3.Connection, question: str, companies: list[str],
                    extra: list[str] | tuple = ()) -> list[dict]:
    query = fts_query(question, extra)
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
        f"""SELECT f.*, d.title, d.file, e.text AS context, m.label AS metric_label
            FROM best_facts f JOIN documents d ON d.doc_id = f.doc_id
            LEFT JOIN elements e ON e.element_id = f.element_id
            LEFT JOIN metric_dictionary m ON m.metric = f.metric
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
    """What the model reads for a fact: our metric name, the report's own wording, and the table's heading and notes
    (e.g. Infosys's 'Turnover rate*' table whose footnote says '* Voluntary attrition for IT services')."""
    label = f.get("metric_label") or f["metric"].replace("_", " ")
    basis = f"{f['basis']}, " if f["basis"] != "unknown" else ""
    # the clean value first: the printed text can be damaged (TCS prints '6, 17 437' for 6,17,437)
    text = (f"{f['company']} · {label} · {f['fy']}: {format_value(f['value'], f['unit'])} ({basis}{f['unit']}). "
            f"As printed: {f['quote']}")
    if f.get("source_kind") == "table" and f.get("context"):
        # only the table's title and its notes: other context lines (e.g. a previous table's footnote) mislead a
        # small model, and a clipped note was why it refused the Infosys attrition question
        parts = [" ".join(p.split()) for p in f["context"].split(" | ")]
        notes = [p for p in parts if p.startswith("Note:")]
        text += f". Table: {parts[0][:160]}" + "".join(f". {n[:240]}" for n in notes)
    return text


def build_sources(conn: sqlite3.Connection, question: str,
                  companies: list[str] | None = None) -> tuple[Understood, list[dict]]:
    """companies: the UI's company toggle, used only when the question itself names no company."""
    understood = understand(question)
    scope = [c for c in COMPANIES if c in (companies or [])]
    if scope and not understood.named:
        understood.companies = scope
    facts = lookup_facts(conn, understood.companies, understood.metric, understood.fy)
    passages = search_passages(conn, question, understood.companies, expansion(understood))
    sources: list[dict] = []
    for f in facts:
        sources.append({
            "kind": "fact", "company": f["company"], "report": f["title"], "pdf": f["file"], "doc_id": f["doc_id"],
            "page": f["page_no"], "text": _fact_text(f), "score": None, "fact_id": f["fact_id"],
            "bbox": [f["x0"], f["y0"], f["x1"], f["y1"]] if f["x0"] is not None else None,
            "metric": f["metric"], "fy": f["fy"], "value": f["value"], "unit": f["unit"], "quote": f["quote"],
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
