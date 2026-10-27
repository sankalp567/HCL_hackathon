"""Evaluation dataset format (JSONL), validation and loading.

queries.jsonl, one object per line:
{
  "qid": "q001", "question": "...", "category": "numeric_lookup|terminology|paraphrase|historical|multi_company|scope|qualitative|unanswerable",
  "qrels": {"<chunk_id>": 3, ...},                 # graded 1-3; empty/absent => unanswerable
  "expected": {                                    # all optional; used for numerical-evidence metrics
     "companies": ["Acme"], "fiscal_year": "FY2025-26", "scope": "consolidated",
     "table_ids": [...], "row_chunk_ids": [...], "pages": [["doc_id", pdf_page_index], ...]},
  "provenance": "synthetic|manual:<who>"           # who created the label. NEVER "model-generated" ground truth.
}
corpus.jsonl, one chunk per line: {"chunk_id", "text", "metadata": {canonical fields...}}
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import Candidate

CATEGORIES = {"numeric_lookup", "terminology", "paraphrase", "historical", "multi_company", "scope", "qualitative",
              "unanswerable"}


def load_corpus(path: str | Path) -> list[Candidate]:
    out, seen = [], set()
    for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if d["chunk_id"] in seen:
            raise ValueError(f"{path}:{ln}: duplicate chunk_id {d['chunk_id']}")
        seen.add(d["chunk_id"])
        out.append(Candidate(chunk_id=d["chunk_id"], text=d["text"], metadata=d.get("metadata", {})))
    return out


def load_queries(path: str | Path, corpus_ids: set[str] | None = None) -> list[dict]:
    qs, seen = [], set()
    for ln, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        q = json.loads(line)
        errs = validate_query(q, corpus_ids)
        if q.get("qid") in seen:
            errs.append("duplicate qid")
        seen.add(q.get("qid"))
        if errs:
            raise ValueError(f"{path}:{ln}: " + "; ".join(errs))
        qs.append(q)
    return qs


def validate_query(q: dict, corpus_ids: set[str] | None = None) -> list[str]:
    errs = []
    for k in ("qid", "question", "category", "provenance"):
        if not q.get(k):
            errs.append(f"missing '{k}'")
    if q.get("category") not in CATEGORIES:
        errs.append(f"unknown category {q.get('category')!r}")
    qrels = q.get("qrels") or {}
    for cid, g in qrels.items():
        if g not in (1, 2, 3):
            errs.append(f"qrels grade for {cid} must be 1-3, got {g}")
        if corpus_ids is not None and cid not in corpus_ids:
            errs.append(f"qrels chunk {cid} not in corpus")
    if q.get("category") == "unanswerable" and qrels:
        errs.append("unanswerable query must have empty qrels")
    if q.get("category") != "unanswerable" and not qrels:
        errs.append("answerable query needs qrels")
    return errs
