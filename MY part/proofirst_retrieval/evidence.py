"""Evidence assembly: ranked records for the chatbot / numerical engine, with whole-table context and company coverage."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .contracts import Candidate, EvidenceRecord, QueryAnalysis


class CorpusIndex:
    """Lookup of table siblings (for full-table context). Built from the same chunk list the lexical index uses."""
    def __init__(self, chunks: Sequence[Candidate]):
        by_table: dict[str, list[Candidate]] = defaultdict(list)
        for c in chunks:
            t = c.metadata.get("table_id")
            if t:
                by_table[t].append(c)
        self.tables = {t: sorted(v, key=lambda c: (c.metadata.get("row_index") is None, c.metadata.get("row_index") or 0,
                                                    c.chunk_id)) for t, v in by_table.items()}

    def table_text(self, table_id: str) -> str | None:
        rows = self.tables.get(table_id)
        if not rows:
            return None
        m = rows[0].metadata
        head = [str(m[k]).replace("|", " | ") for k in ("table_title", "headers") if m.get(k)]
        unit = f"(unit: {m['unit']})" if m.get("unit") else ""
        return "\n".join(([*head, unit] if unit else head) + [r.text for r in rows])


def _same(a, b) -> bool:
    return a is not None and b is not None and str(a).casefold() == str(b).casefold()


def select_with_company_quota(cands: Sequence[Candidate], k: int, companies: Sequence[str]) -> list[Candidate]:
    """Top-k that guarantees each requested company at least ceil(k/(2*n)) slots *when it has candidates*, then fills by rank.
    Never fabricates evidence: a company with no candidates just gets none (reported by the pipeline)."""
    if len(companies) < 2 or k <= 0:
        return list(cands[:k])
    quota = max(1, -(-k // (2 * len(companies))))   # ceil(k / 2n)
    chosen: list[Candidate] = []
    ids: set[str] = set()
    for comp in companies:
        n = 0
        for c in cands:
            if n >= quota:
                break
            if _same(c.metadata.get("company"), comp) and c.chunk_id not in ids:
                chosen.append(c); ids.add(c.chunk_id); n += 1
    for c in cands:
        if len(chosen) >= k:
            break
        if c.chunk_id not in ids:
            chosen.append(c); ids.add(c.chunk_id)
    rank = {c.chunk_id: i for i, c in enumerate(cands)}
    return sorted(chosen, key=lambda c: rank[c.chunk_id])


def assemble(cands: Sequence[Candidate], corpus: CorpusIndex | None, analysis: QueryAnalysis | None = None,
             collapse_tables: bool = True) -> list[EvidenceRecord]:
    """Candidates (already ranked) -> evidence. Table-row hits get the full table text; hits from the same table collapse into
    one record (best-ranked chunk first, others in matched_chunk_ids). Numerical intents always carry table context."""
    out: list[EvidenceRecord] = []
    seen_tables: dict[str, EvidenceRecord] = {}
    for c in cands:
        m = c.metadata
        tid = m.get("table_id")
        if collapse_tables and tid and tid in seen_tables:
            seen_tables[tid].matched_chunk_ids.append(c.chunk_id)
            continue
        ctx = corpus.table_text(tid) if (corpus and tid) else None
        rec = EvidenceRecord(
            company=m.get("company"), doc_id=m.get("doc_id"), pdf_page_index=m.get("pdf_page_index"),
            printed_page=m.get("printed_page"), chunk_id=c.chunk_id, source_text=c.text,
            content_type=m.get("content_type"), scope=m.get("scope"), table_id=tid, table_title=m.get("table_title"),
            headers=m.get("headers"), row_label=m.get("row_label"), unit=m.get("unit"), periods=m.get("periods"),
            fiscal_year=m.get("fiscal_year"), table_context=ctx, matched_chunk_ids=[c.chunk_id],
            scores=dict(c.scores), ranks=dict(c.ranks), fusion_score=c.fusion_score, rerank_score=c.rerank_score)
        out.append(rec)
        if tid:
            seen_tables[tid] = rec
    return out


def company_coverage(records_or_cands: Iterable, requested: Sequence[str]) -> dict[str, bool]:
    got = set()
    for r in records_or_cands:
        comp = r.company if isinstance(r, EvidenceRecord) else r.metadata.get("company")
        if comp:
            got.add(str(comp).casefold())
    return {c: str(c).casefold() in got for c in requested}
