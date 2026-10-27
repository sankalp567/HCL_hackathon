"""Rule-based query analysis. Extracts only what the user actually wrote: never invents a company/year filter."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .contracts import FilterSpec, QueryAnalysis
from .filters import find_fiscal_year_spans

DEFAULT_COMPANY_ALIASES: dict[str, list[str]] = {
    "TCS": ["tcs", "tata consultancy services", "tata consultancy"],
    "Infosys": ["infosys", "infy"],
    "HCLTech": ["hcltech", "hcl technologies", "hcl tech", "hcl"],
}

# canonical metric -> surface forms that identify it in a question, and controlled expansions for lexical search.
# `generic` surface forms are ambiguous (flagged); expansions never cross into a confusable metric.
DEFAULT_METRICS: dict[str, dict] = {
    "revenue_from_operations": {
        "surface": ["revenue from operations", "operating revenue", "net sales", "core revenue", "top line", "topline"],
        "generic": ["revenue", "revenues", "turnover", "sales"],
        "expand": ["revenue from operations", "operating revenue", "net sales"],
        "confusable_with": ["total_income", "other_income"],
    },
    "total_income": {
        "surface": ["total income", "gross income"],
        "generic": [],
        "expand": ["total income"],
        "confusable_with": ["revenue_from_operations"],
    },
    "other_income": {
        "surface": ["other income", "non operating income"],
        "generic": [],
        "expand": ["other income"],
        "confusable_with": ["total_income"],
    },
    "net_profit": {
        "surface": ["profit after tax", "net profit", "profit for the year", "net income", "bottom line", "pat"],
        "generic": ["profit", "earnings"],
        "expand": ["profit for the year", "profit after tax", "net profit"],
        "confusable_with": ["operating_margin"],
    },
    "operating_margin": {
        "surface": ["operating margin", "ebit margin", "operating profit margin"],
        "generic": ["margin"],
        "expand": ["operating margin", "ebit margin"],
        "confusable_with": ["net_profit"],
    },
    "headcount": {
        "surface": ["headcount", "employee strength", "number of employees", "workforce", "employee count",
                    "people strength"],
        "generic": ["employees"],
        "expand": ["headcount", "employee strength", "workforce", "number of employees"],
        "confusable_with": [],
    },
    "attrition": {
        "surface": ["attrition", "employee turnover", "voluntary exits", "staff churn"],
        "generic": [],
        "expand": ["attrition", "employee turnover", "voluntary exits"],
        "confusable_with": [],
    },
    "dividend": {
        "surface": ["dividend", "payout"],
        "generic": [],
        "expand": ["dividend", "payout"],
        "confusable_with": [],
    },
    "order_book": {
        "surface": ["order book", "total contract value", "tcv", "deal wins", "deal pipeline"],
        "generic": [],
        "expand": ["order book", "total contract value", "deal wins"],
        "confusable_with": [],
    },
}

_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference|differ|higher than|lower than|larger than|"
                      r"relative to|rank|which (?:company|one))\b", re.I)
_TREND = re.compile(r"\b(trend|growth|grew|yoy|year[- ]on[- ]year|over the years|historical|history|change[sd]?|"
                    r"compared (?:to|with) (?:the )?(?:previous|last|prior))\b", re.I)
_QUAL = re.compile(r"\b(why|explain|describe|discuss|strategy|strategic|risk|risks|outlook|approach|initiative|"
                   r"initiatives|focus|plans?|policy|commentary|according to|mention)\b", re.I)
_STRONG_QUAL = re.compile(r"\b(why|explain|describe|discuss|strategy|strategic|risks?|outlook|approach|initiatives?|polic(?:y|ies))\b", re.I)
_NUMQ = re.compile(r"\b(how much|how many|what (?:was|is|were)|value|amount|number|figure|percent(?:age)?|rate|"
                   r"%|crore|million|total)\b", re.I)
_REL_PREV = re.compile(r"\b(last|previous|prior)\s+(?:fiscal|financial)?\s*(year|fy)\b", re.I)
_REL_LATEST = re.compile(r"\b(latest|most recent|newest)\b|\b(current|this)\s+(?:fiscal|financial)?\s*(year|fy|period)\b", re.I)
# wording that asks about the companies collectively, so an unnamed question means "search all registered companies"
_CROSS_COMPANY = re.compile(r"\b(?:which|what)\s+(?:company|companies|firm|firms)\b|\ball\s+(?:three|3|the)?\s*(?:companies|firms)\b|"
                            r"\b(?:each|every)\s+(?:company|firm)\b|\bacross\s+(?:the\s+|all\s+)?(?:three\s+)?(?:companies|firms)\b|"
                            r"\b(?:among|between|of)\s+the\s+(?:three|3)\b|\bthree\s+companies\b|\ball\s+three\b", re.I)


@dataclass
class QueryProcessor:
    company_aliases: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_COMPANY_ALIASES))
    metrics: dict[str, dict] = field(default_factory=lambda: dict(DEFAULT_METRICS))

    def analyze(self, question: str) -> QueryAnalysis:
        q = question.strip()
        low = q.lower()
        a = QueryAnalysis(original=q)

        for canon, names in self.company_aliases.items():
            for n in sorted(names, key=len, reverse=True):
                m = re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", low)
                if m:
                    a.companies.append(canon)
                    a.surface.setdefault("company", []).append(q[m.start():m.end()])
                    break

        spans = find_fiscal_year_spans(q)
        a.periods = [fy for fy, _ in spans]
        if spans:
            a.surface["fiscal_year"] = [lit for _, lit in spans]
        if _REL_PREV.search(q):
            a.relative_period = "previous"
            a.notes.append("relative period 'previous year' noted; no fiscal-year filter invented")
        elif _REL_LATEST.search(q):
            a.relative_period = "latest"

        # metrics: longest specific surface forms first; generic words flagged ambiguous
        consumed = low
        for canon, spec in self.metrics.items():
            for s in sorted(spec["surface"], key=len, reverse=True):
                if re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", consumed):
                    if canon not in a.metrics:
                        a.metrics.append(canon)
                    consumed = consumed.replace(s, " ")
        for canon, spec in self.metrics.items():
            if canon in a.metrics:
                continue
            for g in spec["generic"]:
                if re.search(rf"(?<![a-z0-9]){re.escape(g)}(?![a-z0-9])", consumed):
                    a.metrics.append(canon)
                    a.ambiguous_terms.append(g)
                    break
        if "revenue_from_operations" in a.metrics and "total_income" in a.metrics:
            a.notes.append("query mentions both revenue-from-operations and total income: kept as distinct metrics")

        if re.search(r"\bconsolidated\b", low):
            a.scope = "consolidated"
        if re.search(r"\bstandalone\b", low):
            a.scope = "standalone" if a.scope is None else None  # both mentioned -> no single scope
            if a.scope is None:
                a.notes.append("both consolidated and standalone mentioned: no scope constraint")

        if a.scope:
            a.surface["scope"] = [a.scope]
        a.cross_company = bool(_CROSS_COMPANY.search(q)) and not a.companies
        a.narrative_cue = bool(_STRONG_QUAL.search(q))
        exp: list[str] = []
        for canon in a.metrics:
            exp.extend(self.metrics[canon]["expand"])
        a.expansion_terms = list(dict.fromkeys(exp))

        if len(a.companies) >= 2 or _COMPARE.search(q):
            a.intent = "comparison"
        elif _STRONG_QUAL.search(q):
            a.intent = "qualitative"
        elif _TREND.search(q) or len(a.periods) >= 2:
            a.intent = "trend"
        elif a.metrics:
            a.intent = "numeric_lookup"
        elif _QUAL.search(q):
            a.intent = "qualitative"
        return a

    @staticmethod
    def to_filters(a: QueryAnalysis) -> FilterSpec:
        """Hard = explicitly stated by the user. Soft = preferences implied by intent. Nothing is invented."""
        hard: dict[str, tuple] = {}
        soft: dict[str, tuple] = {}
        if a.companies:
            hard["company"] = tuple(a.companies)
        if a.periods:
            hard["fiscal_year"] = tuple(a.periods)
        if a.scope:
            hard["scope"] = (a.scope,)
        # table preference only when the wording does not signal narrative content (policy / strategy / risk / why ...)
        if a.intent in ("numeric_lookup", "comparison", "trend") and a.metrics and not a.narrative_cue:
            soft["content_type"] = ("table",)
        return FilterSpec(hard, soft)
