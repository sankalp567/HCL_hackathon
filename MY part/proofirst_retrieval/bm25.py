"""Pure-Python/numpy BM25 over the same chunks the dense retriever searches. No dependencies, no second parse/chunk."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np

from .contracts import Candidate, FieldMap, FilterSpec, QueryAnalysis
from .filters import matches
from .text import tokenize

NAME = "bm25"


class BM25Retriever:
    """Okapi BM25 (Lucene-style non-negative IDF: ln(1 + (N-n+0.5)/(n+0.5))), unigram + bigram terms.

    Query = original tokens (weight 1.0) + optional controlled alias expansion tokens (weight `alias_weight`).
    Filters: hard filters restrict the *scored set*; IDF/avgdl stay corpus-global so scores are comparable.
    """
    name = NAME

    def __init__(self, chunks: Sequence[Candidate], k1: float = 1.5, b: float = 0.75,
                 alias_weight: float = 0.5, use_alias: bool = False):
        self.k1, self.b, self.alias_weight, self.use_alias = k1, b, alias_weight, use_alias
        self.chunks = list(chunks)
        self.n = len(self.chunks)
        self._post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._len = np.zeros(self.n)
        for i, c in enumerate(self.chunks):
            toks = tokenize(c.index_text())
            self._len[i] = len(toks)
            for t, tf in Counter(toks).items():
                self._post[t].append((i, tf))
        self.avgdl = float(self._len.mean()) if self.n else 0.0
        self._idf = {t: math.log(1.0 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self._post.items()}

    def with_alias(self, use_alias: bool) -> "BM25Retriever":
        """Cheap view with a different alias setting sharing the same index."""
        o = object.__new__(BM25Retriever)
        o.__dict__.update(self.__dict__)
        o.use_alias = use_alias
        return o

    def query_terms(self, analysis: QueryAnalysis | str) -> dict[str, float]:
        text = analysis if isinstance(analysis, str) else analysis.original
        terms: dict[str, float] = {}
        for t in tokenize(text):
            terms[t] = 1.0
        if self.use_alias and not isinstance(analysis, str):
            for phrase in analysis.expansion_terms:
                for t in tokenize(phrase):
                    terms.setdefault(t, self.alias_weight)
        return terms

    def search(self, analysis: QueryAnalysis | str, top_k: int = 20,
               filters: FilterSpec | None = None) -> list[Candidate]:
        terms = self.query_terms(analysis)
        mask = None
        if filters is not None and filters.hard:
            mask = np.fromiter((matches(c.metadata, filters.hard) for c in self.chunks), bool, self.n)
        scores = np.zeros(self.n)
        for t, w in terms.items():
            post = self._post.get(t)
            if not post:
                continue
            idf = self._idf[t]
            for i, tf in post:
                norm = tf + self.k1 * (1 - self.b + self.b * self._len[i] / (self.avgdl or 1.0))
                scores[i] += w * idf * tf * (self.k1 + 1) / norm
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        # stable: by score desc, then corpus order
        order = sorted((i for i in np.nonzero(scores > 0)[0]), key=lambda i: (-scores[i], i))[:top_k]
        out = []
        for r, i in enumerate(order, 1):
            c = self.chunks[i].copy()
            c.ranks = {NAME: r}
            c.scores = {NAME: float(scores[i])}
            out.append(c)
        return out

    def corpus_values(self, fields: Iterable[str]) -> dict[str, set[str]]:
        from .filters import norm_value
        vals: dict[str, set[str]] = {f: set() for f in fields}
        for c in self.chunks:
            for f in vals:
                v = c.metadata.get(f)
                if v not in (None, ""):
                    vals[f].add(norm_value(f, v))
        return vals
