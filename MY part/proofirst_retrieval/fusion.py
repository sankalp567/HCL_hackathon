"""Reciprocal Rank Fusion + deduplication.

RRF(d) = sum over ranked lists L containing d of  w_L / (k + rank_L(d)),  rank is ONE-based (best = 1), k = 60 default
(Cormack et al. 2009). A chunk repeated inside one list counts once (its best/first rank). Missing from a list => no term.
Ties: higher fusion score, then better best-rank, then first appearance (list order, then position) => deterministic.
"""
from __future__ import annotations

from typing import Sequence

from .contracts import Candidate

RRF_K = 60


def dedupe_ranked(cands: Sequence[Candidate]) -> list[Candidate]:
    """Keep first occurrence of each chunk_id within one ranked list (preserves order)."""
    seen: set[str] = set()
    out = []
    for c in cands:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c)
    return out


def rrf_fuse(ranked_lists: Sequence[Sequence[Candidate]], names: Sequence[str] | None = None, k: int = RRF_K,
             weights: Sequence[float] | None = None, top_n: int | None = None) -> list[Candidate]:
    """Fuse ranked lists (each best-first). Returns new Candidates with fusion_score and merged ranks/scores.

    Rank within each list is recomputed from list position after in-list dedupe (so the fusion never depends on
    stale rank fields); the original retriever rank/score are preserved in `.ranks` / `.scores` under the list name.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    names = list(names) if names is not None else [f"list{i}" for i in range(len(ranked_lists))]
    weights = list(weights) if weights is not None else [1.0] * len(ranked_lists)
    if not (len(names) == len(weights) == len(ranked_lists)):
        raise ValueError("names/weights must match number of ranked lists")

    fused: dict[str, Candidate] = {}
    first_seen: dict[str, int] = {}
    best_rank: dict[str, int] = {}
    order = 0
    for lst, name, w in zip(ranked_lists, names, weights):
        for pos, c in enumerate(dedupe_ranked(lst), 1):
            if c.chunk_id not in fused:
                m = c.copy()
                m.ranks, m.scores, m.fusion_score = {}, {}, 0.0
                fused[c.chunk_id] = m
                first_seen[c.chunk_id] = order
                order += 1
                best_rank[c.chunk_id] = pos
            m = fused[c.chunk_id]
            m.fusion_score += w / (k + pos)
            m.ranks[name] = pos
            if c.scores:
                m.scores[name] = next(iter(c.scores.values()))
            best_rank[c.chunk_id] = min(best_rank[c.chunk_id], pos)
    out = sorted(fused.values(), key=lambda m: (-m.fusion_score, best_rank[m.chunk_id], first_seen[m.chunk_id]))
    return out if top_n is None else out[:top_n]
