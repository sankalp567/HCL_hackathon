"""Retrieval orchestration: analyze -> route -> (filter) -> dense + bm25 -> RRF -> (soft boost) -> (rerank) -> policy -> select -> evidence.

Result status (RetrievalResult.status), from most to least severe:
  error             every retriever call failed (infrastructure problem, NOT evidence about the reports)
  no_evidence       retrievers ran but returned nothing
  low_relevance     candidates existed but all fell below the configured RelevancePolicy
  partial_coverage  a multi-company question got evidence for only some of the companies
  degraded          candidates returned but a constraint/stage could not be honoured (filter dropped, reranker/retriever failed, routing impossible)
  ok                none of the above
`absence_is_proof` is always False: an empty or failed retrieval never proves a fact is absent from the reports.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .bm25 import BM25Retriever
from .contracts import Candidate, EvidenceRecord, FilterSpec, QueryAnalysis
from .evidence import CorpusIndex, assemble, company_coverage, select_with_company_quota
from .filters import FilterReport, normalize_fy, resolve_filters
from .fusion import RRF_K, rrf_fuse
from .query import QueryProcessor
from .rerank import NoReranker, Reranker, RerankerUnavailable

FILTER_FIELDS = ("company", "fiscal_year", "scope", "doc_id", "content_type", "table_id")

NO_EVIDENCE_NOTE = ("No candidate chunks were retrieved for this query. This is a retrieval outcome and does not prove "
                    "the fact is absent from the reports.")


@dataclass(frozen=True)
class RelevancePolicy:
    """Optional minimum-relevance policy. EVERY threshold defaults to None (disabled): useful values depend on the embedding
    model, distance metric and index, so they must be calibrated on labelled real data, never guessed.
    A candidate is kept if it meets AT LEAST ONE enabled threshold (supported by lexical OR dense OR rerank evidence).
    With no threshold enabled every candidate passes."""
    min_bm25: float | None = None
    max_dense_distance: float | None = None
    min_rerank: float | None = None

    def enabled(self) -> bool:
        return any(v is not None for v in (self.min_bm25, self.max_dense_distance, self.min_rerank))

    def passes(self, c: Candidate) -> bool:
        if not self.enabled():
            return True
        if self.min_bm25 is not None:
            bm = [v for k, v in c.scores.items() if k.startswith("bm25")]
            if bm and max(bm) >= self.min_bm25:
                return True
        if self.max_dense_distance is not None:
            dd = [v for k, v in c.scores.items() if k == "dense_distance"] or [-v for k, v in c.scores.items() if k.startswith("dense")]
            if dd and min(dd) <= self.max_dense_distance:
                return True
        if self.min_rerank is not None and c.rerank_score is not None and c.rerank_score >= self.min_rerank:
            return True
        return False


@dataclass
class PipelineConfig:
    name: str = "custom"
    use_dense: bool = True
    use_bm25: bool = True
    k_candidates: int = 20          # per retriever (and per entity when per_entity is on)
    rrf_k: int = RRF_K
    use_filters: bool = False       # apply user-stated hard filters (and routing, below)
    per_entity: bool = False        # multi-company queries: retrieve each company independently, fuse all lists
    implicit_routing: bool = True   # with use_filters: unnamed cross-company questions search all indexed companies; "latest" = newest indexed report year
    alias_expansion: bool = False   # controlled alias expansion in the lexical query
    reranker: Reranker | None = None
    rerank_pool: int = 20           # only the top-N fused candidates are reranked
    final_k: int = 10
    soft_boost: float = 0.0         # x(1+b) on fusion score for soft-filter matches; applies to fused (multi-list) results only
    company_quota: bool = False     # enforce per-company minimum slots in final top-k for multi-company queries
    min_relevance: RelevancePolicy | None = None
    assemble_tables: bool = True

    def describe(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("reranker", "min_relevance")}
        d["reranker"] = getattr(self.reranker, "name", "none") if self.reranker else "none"
        d["min_relevance"] = dict(self.min_relevance.__dict__) if self.min_relevance else None
        return d


@dataclass
class RetrievalResult:
    analysis: QueryAnalysis
    candidates: list[Candidate]
    evidence: list[EvidenceRecord]
    timings_ms: dict[str, float]
    counts: dict[str, int]
    filter_report: FilterReport | None
    limitations: list[str]
    coverage: dict[str, bool]
    reranker_status: str
    config: dict[str, Any]
    status: str = "ok"                 # ok | degraded | partial_coverage | low_relevance | no_evidence | error
    status_note: str = ""
    retrieval_errors: list[str] = field(default_factory=list)       # e.g. "dense: ConnectionError(...)"
    routing: dict[str, Any] = field(default_factory=dict)           # how companies / period were chosen (explicit vs implicit)
    top_scores: dict[str, float] = field(default_factory=dict)      # best native score per retriever + fusion + rerank
    below_threshold: list[dict[str, Any]] = field(default_factory=list)  # candidates removed by the relevance policy
    absence_is_proof: bool = False     # ALWAYS False


class RetrievalPipeline:
    def __init__(self, lexical: BM25Retriever, dense: Any | None, corpus_chunks: list[Candidate],
                 query_processor: QueryProcessor | None = None):
        self.lexical, self.dense = lexical, dense
        self.qp = query_processor or QueryProcessor()
        self.corpus = CorpusIndex(corpus_chunks)
        self._corpus_values = lexical.corpus_values(FILTER_FIELDS)
        # registered companies / report years come from the INDEX metadata (stored spelling), never from a hard-coded list
        self._company_names = sorted({str(c.metadata["company"]) for c in corpus_chunks if c.metadata.get("company")})
        years = {normalize_fy(c.metadata["fiscal_year"]) for c in corpus_chunks if c.metadata.get("fiscal_year")}
        self._report_years = sorted((y for y in years if y), key=lambda y: int(y[2:6]))

    # ------------------------------------------------------------------ routing
    def _route(self, a: QueryAnalysis, spec: FilterSpec, limitations: list[str], degraded: list[str]) -> tuple[FilterSpec, dict]:
        routing: dict[str, Any] = {"companies": "explicit" if a.companies else "none", "companies_used": list(a.companies),
                                   "period": "explicit" if a.periods else "none", "period_used": list(a.periods)}
        if not a.companies and a.cross_company:
            if len(self._company_names) >= 2:
                spec = spec.with_hard("company", tuple(self._company_names))
                routing.update(companies="implicit_all_registered", companies_used=list(self._company_names))
            else:
                msg = "cross-company question but the index has fewer than two companies in its metadata: cannot route to all companies"
                limitations.append(msg); degraded.append(msg)
        if not a.periods and a.relative_period == "latest":
            if self._report_years:
                latest = self._report_years[-1]
                spec = spec.with_hard("fiscal_year", (latest,))
                routing.update(period="latest_from_index", period_used=[latest])
                limitations.append(f"'latest' resolved to the newest report year present in the index ({latest}); not guessed")
            else:
                msg = "'latest' requested but the index has no fiscal_year metadata: latest period cannot be determined"
                limitations.append(msg); degraded.append(msg)
        return spec, routing

    # ------------------------------------------------------------------ run
    def run(self, question: str, cfg: PipelineConfig) -> RetrievalResult:
        t = {}
        T0 = time.perf_counter()
        limitations: list[str] = []
        degraded: list[str] = []
        errors: list[str] = []

        a = self.qp.analyze(question)
        t["analysis"] = _ms(T0)

        spec = self.qp.to_filters(a) if cfg.use_filters else FilterSpec()
        routing: dict[str, Any] = {}
        report = None
        if cfg.use_filters:
            if cfg.implicit_routing:
                spec, routing = self._route(a, spec, limitations, degraded)
            spec, report = resolve_filters(spec, self._corpus_values)
            limitations += report.limitations
            if report.unavailable:
                degraded.append("one or more requested filters could not be applied")
        a.enforced = dict(spec.hard)

        lex = self.lexical.with_alias(cfg.alias_expansion)
        companies = list(spec.hard.get("company", ()))
        if cfg.per_entity and len(companies) >= 2:
            scopes = [spec.with_hard("company", (c,)) for c in companies]
            labels = [f"[{c}]" for c in companies]
        else:
            scopes = [spec if spec.hard else None]
            labels = [""]

        lists, names = [], []
        td = tb = 0.0
        n_dense = n_bm25 = 0
        calls = ok_calls = 0
        for sc, lab in zip(scopes, labels):
            if cfg.use_dense and self.dense is not None:
                s = time.perf_counter(); calls += 1
                try:
                    r = self.dense.search(a, cfg.k_candidates, sc)
                    ok_calls += 1
                    lists.append(r); names.append("dense" + lab); n_dense += len(r)
                except Exception as e:  # infrastructure failure is reported, never mistaken for "no evidence"
                    errors.append(f"dense{lab}: {type(e).__name__}: {e}")
                td += time.perf_counter() - s
            if cfg.use_bm25:
                s = time.perf_counter(); calls += 1
                try:
                    r = lex.search(a, cfg.k_candidates, sc)
                    ok_calls += 1
                    lists.append(r); names.append("bm25" + lab); n_bm25 += len(r)
                except Exception as e:
                    errors.append(f"bm25{lab}: {type(e).__name__}: {e}")
                tb += time.perf_counter() - s
        t["dense"], t["bm25"] = td * 1000, tb * 1000
        if cfg.use_dense and self.dense is None:
            limitations.append("dense retriever not configured: dense lists omitted")
            degraded.append("dense retriever not configured")
        for e in errors:
            limitations.append(f"retriever failed ({e})")
        if errors and ok_calls:
            degraded.append("a retriever failed; results come from the remaining retrievers only")

        s = time.perf_counter()
        if len(lists) == 1:
            fused = []
            for pos, c in enumerate(lists[0], 1):
                m = c.copy(); m.fusion_score = None
                m.ranks = {names[0]: pos}
                fused.append(m)
        else:
            fused = rrf_fuse(lists, names, k=cfg.rrf_k)
        n_union = sum(len(x) for x in lists)
        n_dedup = len(fused)
        soft = spec.soft if cfg.use_filters else {}
        if cfg.soft_boost and soft and len(lists) > 1:
            for c in fused:
                if all(str(c.metadata.get(f, "")).lower() in {str(v).lower() for v in vs} for f, vs in soft.items()):
                    c.soft_boost = cfg.soft_boost
                    c.fusion_score = (c.fusion_score or 0.0) * (1.0 + cfg.soft_boost)
            fused.sort(key=lambda c: -(c.fusion_score or 0.0))
        t["fusion"] = _ms(s)

        status_rr = "none"
        s = time.perf_counter()
        pool = fused[: cfg.rerank_pool] if cfg.reranker else fused
        n_rerank = 0
        if cfg.reranker is not None and pool:
            try:
                reranked = cfg.reranker.rerank(a, [c.copy() for c in pool])   # copies: a failing reranker cannot leak partial edits
                n_rerank = len(pool)
                fused = reranked + fused[cfg.rerank_pool:]
                status_rr = f"ok:{getattr(cfg.reranker, 'name', 'reranker')}"
            except Exception as e:  # failure => keep fusion order, say so
                status_rr = f"failed:{type(e).__name__}"
                limitations.append(f"reranker failed ({e}); fusion order kept")
                degraded.append("reranker failed")
        t["rerank"] = _ms(s)

        below: list[dict[str, Any]] = []
        n_before_policy = len(fused)
        if cfg.min_relevance is not None and cfg.min_relevance.enabled():
            kept = []
            for c in fused:
                if cfg.min_relevance.passes(c):
                    kept.append(c)
                else:
                    below.append({"chunk_id": c.chunk_id, "scores": dict(c.scores), "rerank_score": c.rerank_score})
            fused = kept

        if cfg.company_quota and len(companies) >= 2:
            final = select_with_company_quota(fused, cfg.final_k, companies)
        else:
            final = fused[: cfg.final_k]
        evidence = assemble(final, self.corpus, a, cfg.assemble_tables)

        cov_companies = companies if len(companies) >= 2 else (a.companies if len(a.companies) >= 2 else [])
        cov = company_coverage(final, cov_companies) if cov_companies else {}
        missing = [c for c, ok in cov.items() if not ok]
        for comp in missing:
            limitations.append(f"no evidence retrieved for requested company '{comp}'")

        # ---- status ----
        if calls and not ok_calls:
            status, note = "error", "All retriever calls failed; this is an infrastructure failure, not evidence about the reports."
        elif not final:
            if below and n_before_policy:
                status, note = "low_relevance", ("Candidates were retrieved but none met the configured relevance policy. "
                                                 "This does not prove the fact is absent from the reports.")
            else:
                status, note = "no_evidence", NO_EVIDENCE_NOTE
            limitations.append(note)
        elif missing:
            status, note = "partial_coverage", f"Evidence found for only part of the requested companies; missing: {', '.join(missing)}."
        elif degraded:
            status, note = "degraded", "; ".join(dict.fromkeys(degraded))
        else:
            status, note = "ok", ""

        top = {}
        for prefix in ("bm25", "dense"):
            vals = [v for c in fused for k, v in c.scores.items() if k.startswith(prefix) and k != "dense_distance"]
            if vals:
                top[prefix] = float(max(vals))
        if any(c.fusion_score is not None for c in final):
            top["fusion"] = float(max(c.fusion_score or 0.0 for c in final))
        if any(c.rerank_score is not None for c in final):
            top["rerank"] = float(max(c.rerank_score for c in final if c.rerank_score is not None))
        t["total"] = _ms(T0)

        return RetrievalResult(
            analysis=a, candidates=final, evidence=evidence, timings_ms=t,
            counts={"dense_hits": n_dense, "bm25_hits": n_bm25, "union_before_dedupe": n_union,
                    "after_dedupe": n_dedup, "reranked_pool": n_rerank, "below_threshold": len(below), "returned": len(final)},
            filter_report=report, limitations=limitations, coverage=cov, reranker_status=status_rr, config=cfg.describe(),
            status=status, status_note=note, retrieval_errors=errors, routing=routing, top_scores=top, below_threshold=below)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000
