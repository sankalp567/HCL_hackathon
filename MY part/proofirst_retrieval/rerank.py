"""Rerankers over a limited candidate pool. Relevance scores are NOT evidence that a number is correct."""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

from .contracts import Candidate, QueryAnalysis
from .text import tokenize


def relevance_text(analysis: QueryAnalysis) -> str:
    """The question with the literal spans removed for constraints the pipeline ALREADY enforced as hard filters.
    A chunk must not be scored down for lacking 'FY2025-26' / the company name / 'consolidated' when retrieval guarantees them.
    Spans of constraints that were NOT enforced (filter dropped, or filtering off) are kept, so e.g. a historical year that could
    not be filtered still discriminates between chunks."""
    import re
    text = analysis.original
    for fld in analysis.enforced:
        for span in analysis.surface.get(fld, []):
            text = re.sub(re.escape(span), " ", text, flags=re.I)
    return text


class Reranker(Protocol):
    name: str
    def rerank(self, analysis: QueryAnalysis, cands: list[Candidate]) -> list[Candidate]: ...


class NoReranker:
    name = "none"
    def rerank(self, analysis, cands):
        return list(cands)


class LexicalFeatureReranker:
    """Heuristic baseline (NOT a learned cross-encoder): query-term / bigram coverage + metric-phrase and scope agreement.
    Deterministic, dependency-free; used to measure the pipeline's rerank stage when no cross-encoder is installed."""
    name = "lexical-feature-heuristic"

    def rerank(self, analysis, cands):
        rel = relevance_text(analysis)          # constraint terms already guaranteed by hard filters are not scored
        q_uni = {t for t in tokenize(rel, bigrams=False)}
        q_bi = {t for t in tokenize(rel) if "_" in t}
        out = []
        for c in cands:
            c = c.copy()
            toks = set(tokenize(c.index_text()))
            uni = len(q_uni & toks) / (len(q_uni) or 1)
            bi = len(q_bi & toks) / (len(q_bi) or 1)
            s = 0.6 * uni + 0.4 * bi
            # agreement bonuses only for constraints the retriever did NOT enforce (otherwise they cannot discriminate)
            if analysis.scope and "scope" not in analysis.enforced and c.metadata.get("scope"):
                s += 0.15 if str(c.metadata["scope"]).lower() == analysis.scope else -0.15
            if analysis.companies and "company" not in analysis.enforced and c.metadata.get("company"):
                s += 0.1 if c.metadata["company"] in analysis.companies else -0.1
            c.rerank_score = s
            out.append(c)
        return sorted(out, key=lambda c: (-c.rerank_score, -(c.fusion_score or 0.0)))


class CrossEncoderReranker:
    """Wraps any pair scorer: scorer(list[(query, passage)]) -> list[float].
    Default scorer lazily loads sentence_transformers.CrossEncoder(model_name); raises RerankerUnavailable if missing."""
    def __init__(self, scorer: Callable[[Sequence[tuple[str, str]]], Sequence[float]] | None = None,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", name: str | None = None):
        self.model_name = model_name
        self.name = name or f"cross-encoder:{model_name}"
        self._scorer = scorer

    def _get_scorer(self):
        if self._scorer is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore
            except Exception as e:  # ImportError or backend load failure
                raise RerankerUnavailable(f"sentence-transformers unavailable: {e}") from e
            model = CrossEncoder(self.model_name)
            self._scorer = lambda pairs: model.predict(list(pairs))
        return self._scorer

    def rerank(self, analysis, cands):
        if not cands:
            return []
        scores = list(self._get_scorer()([(analysis.original, c.index_text()) for c in cands]))
        if len(scores) != len(cands):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(cands)} candidates")
        out = []
        for c, s in zip(cands, scores):
            c = c.copy(); c.rerank_score = float(s); out.append(c)
        return sorted(out, key=lambda c: (-c.rerank_score, -(c.fusion_score or 0.0)))


class RerankerUnavailable(RuntimeError):
    pass


def cross_encoder_status(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> tuple[bool, str]:
    """Probe without downloading anything: is the package importable?"""
    import importlib.util
    for mod in ("torch", "sentence_transformers"):
        if importlib.util.find_spec(mod) is None:
            return False, f"python package '{mod}' not installed in this environment"
    return True, "packages importable (model weights not verified/downloaded)"
