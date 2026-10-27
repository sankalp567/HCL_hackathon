"""Retrieval metrics.

Relevance convention: qrels = {chunk_id: grade}, grade in {0,1,2,3}; grade > 0 is "relevant" for binary metrics.
  Recall@K    = |relevant in top K| / |relevant|
  Precision@K = |relevant in top K| / K          (denominator is K even if fewer results were returned)
  MRR         = 1 / rank of first relevant (0 if none retrieved)
  nDCG@K      = DCG@K / IDCG@K, gain = 2^grade - 1, discount = log2(position + 1); IDCG from all labelled grades
Queries with NO relevant chunks (unanswerable) return None: excluded from means, reported separately.
Numerical-evidence metrics are retrieval-level only; they do not establish that a parsed number is correct.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


def _rel(qrels: Mapping[str, int]) -> set[str]:
    return {c for c, g in qrels.items() if g > 0}


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float | None:
    rel = _rel(qrels)
    if not rel:
        return None
    return len(rel & set(ranked[:k])) / len(rel)


def precision_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float | None:
    rel = _rel(qrels)
    if not rel or k <= 0:
        return None
    return len(rel & set(ranked[:k])) / k


def mrr(ranked: Sequence[str], qrels: Mapping[str, int]) -> float | None:
    rel = _rel(qrels)
    if not rel:
        return None
    for i, c in enumerate(_uniq(ranked), 1):
        if c in rel:
            return 1.0 / i
    return 0.0


def _uniq(ranked: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(ranked))


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float | None:
    ranked = _uniq(ranked)   # a repeated id must not earn gain twice
    if not _rel(qrels):
        return None
    def dcg(grades):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))
    gains = [qrels.get(c, 0) for c in ranked[:k]]
    ideal = sorted((g for g in qrels.values() if g > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg else None


def query_metrics(ranked: Sequence[str], qrels: Mapping[str, int]) -> dict[str, float | None]:
    return {
        "recall@5": recall_at_k(ranked, qrels, 5), "recall@10": recall_at_k(ranked, qrels, 10),
        "precision@5": precision_at_k(ranked, qrels, 5), "mrr": mrr(ranked, qrels),
        "ndcg@5": ndcg_at_k(ranked, qrels, 5), "ndcg@10": ndcg_at_k(ranked, qrels, 10),
    }


def numeric_evidence_metrics(top: Sequence[Mapping[str, Any]], expected: Mapping[str, Any] | None, k: int = 10,
                             ) -> dict[str, float | None]:
    """top: metadata dicts (+'chunk_id') of the ranked results. expected: dataset 'expected' block (any key optional).
    Each metric is None when the label it needs is absent (so it is excluded, not counted as a miss)."""
    e = expected or {}
    topk = list(top[:k])
    out: dict[str, float | None] = {}

    pages = {(d, p) for d, p in e.get("pages", [])}
    out["source_page_hit"] = (float(any((m.get("doc_id"), m.get("pdf_page_index")) in pages for m in topk))
                              if pages else None)
    comps, fy = e.get("companies") or [], e.get("fiscal_year")
    if len(comps) == 1:
        out["company_year_hit"] = float(any(m.get("company") == comps[0] and (fy is None or m.get("fiscal_year") == fy)
                                            for m in topk))
    else:
        out["company_year_hit"] = None
    tids = set(e.get("table_ids") or [])
    out["metric_table_hit"] = float(any(m.get("table_id") in tids for m in topk)) if tids else None
    rows = set(e.get("row_chunk_ids") or [])
    out["row_evidence_hit"] = float(any(m.get("chunk_id") in rows for m in topk)) if rows else None
    if len(comps) >= 2:
        got = {m.get("company") for m in topk}
        out["company_coverage"] = sum(c in got for c in comps) / len(comps)
        out["company_full_coverage"] = float(all(c in got for c in comps))
    else:
        out["company_coverage"] = out["company_full_coverage"] = None
    sc = e.get("scope")
    out["scope_match@1"] = float(bool(topk) and topk[0].get("scope") == sc) if sc else None
    return out


def mean_ignoring_none(values: Iterable[float | None]) -> tuple[float | None, int]:
    v = [x for x in values if x is not None]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


def percentile(values: Sequence[float], p: float, min_n: int = 20) -> float | None:
    """Linear-interpolated percentile; None when the sample is too small to report honestly."""
    if len(values) < min_n:
        return None
    import numpy as np
    return float(np.percentile(values, p))
