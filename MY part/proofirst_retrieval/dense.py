"""Dense retrieval adapters.

* ChromaDenseRetriever: wraps Member 2's ChromaDB collection. Needs a query embedder that is the SAME model/revision/
  dimension/instruction as the indexing pipeline; we never embed documents and never create a collection.
* HashingDenseRetriever: deterministic char-n-gram hashing stand-in used ONLY for fixtures/benchmarks when the real
  index is unavailable. It is NOT Qwen3-Embedding and results from it say nothing about Qwen quality.
"""
from __future__ import annotations

import zlib
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import Candidate, FieldMap, FilterSpec, QueryAnalysis
from .filters import matches, to_chroma_where
from .text import tokenize

NAME = "dense"


def _distance_to_score(d: float) -> float:
    """Chroma returns a distance (smaller = closer). We keep the raw distance in scores['dense_distance']
    and expose similarity-like score = -distance so 'higher is better' holds for every retriever."""
    return -float(d)


class ChromaDenseRetriever:
    name = NAME
    stand_in = False

    def __init__(self, collection: Any, embed_query: Callable[[str], Sequence[float]],
                 field_map: FieldMap | None = None, text_key_is_document: bool = True):
        self.collection = collection
        self.embed_query = embed_query
        self.field_map = field_map or FieldMap()

    def search(self, analysis: QueryAnalysis | str, top_k: int = 20,
               filters: FilterSpec | None = None) -> list[Candidate]:
        q = analysis if isinstance(analysis, str) else analysis.original
        where = to_chroma_where(filters.hard, self.field_map) if filters and filters.hard else None
        kwargs = dict(query_embeddings=[list(self.embed_query(q))], n_results=top_k,
                      include=["documents", "metadatas", "distances"])
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out = []
        for r, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), 1):
            md = self.field_map.to_canonical(meta)
            c = Candidate(chunk_id=cid, text=doc or "", metadata=md,
                          ranks={NAME: r}, scores={NAME: _distance_to_score(dist), "dense_distance": float(dist)})
            out.append(c)
        return out


def load_corpus_from_chroma(collection: Any, field_map: FieldMap | None = None, page: int = 500) -> list[Candidate]:
    """Read every chunk (id, document, metadata) from the existing collection for the lexical index.
    Reads only; no re-parse, no re-chunk, no embeddings."""
    fm = field_map or FieldMap()
    out, offset = [], 0
    while True:
        res = collection.get(include=["documents", "metadatas"], limit=page, offset=offset)
        ids = res.get("ids") or []
        if not ids:
            break
        for cid, doc, meta in zip(ids, res.get("documents") or [], res.get("metadatas") or []):
            out.append(Candidate(chunk_id=cid, text=doc or "", metadata=fm.to_canonical(meta)))
        offset += len(ids)
        if len(ids) < page:
            break
    return out


class HashingDenseRetriever:
    """STAND-IN dense retriever (signed hashing of word + char-trigram features, cosine). Deterministic (crc32)."""
    name = NAME
    stand_in = True

    def __init__(self, chunks: Sequence[Candidate], dim: int = 1024):
        self.dim = dim
        self.chunks = list(chunks)
        self.mat = np.stack([self._embed(c.index_text()) for c in self.chunks]) if self.chunks else np.zeros((0, dim))

    def _embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for t in tokenize(text, bigrams=False):
            for feat, w in [(f"w:{t}", 1.0)] + [(f"c:{t[i:i+3]}", 0.3) for i in range(max(1, len(t) - 2))]:
                h = zlib.crc32(feat.encode())
                v[h % self.dim] += w * (1.0 if (h >> 16) & 1 else -1.0)
        n = np.linalg.norm(v)
        return v / n if n else v

    def search(self, analysis: QueryAnalysis | str, top_k: int = 20,
               filters: FilterSpec | None = None) -> list[Candidate]:
        q = analysis if isinstance(analysis, str) else analysis.original
        sims = self.mat @ self._embed(q)
        idx = range(len(self.chunks))
        if filters is not None and filters.hard:
            idx = [i for i in idx if matches(self.chunks[i].metadata, filters.hard)]
        order = sorted(idx, key=lambda i: (-sims[i], i))[:top_k]
        out = []
        for r, i in enumerate(order, 1):
            c = self.chunks[i].copy()
            c.ranks = {NAME: r}
            c.scores = {NAME: float(sims[i]), "dense_distance": float(1.0 - sims[i])}
            out.append(c)
        return out
