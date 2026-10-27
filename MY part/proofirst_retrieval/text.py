"""Shared tokenisation for the lexical index, the dense stand-in and the query processor."""
from __future__ import annotations
import re

_STOP = frozenset("a an the of in on for to and or is are was were be been by with at as it its this that "
                  "from what which who how did do does has have had their there than into per s".split())
_NUM_COMMA = re.compile(r"(?<=\d),(?=\d)")
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def stem(tok: str) -> str:
    """Crude plural stemmer (documented limitation): employees->employee, operations->operation."""
    if tok.isdigit() or len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, bigrams: bool = True) -> list[str]:
    """Lowercase, strip thousands commas (2,30,908 -> 230908), drop stopwords, stem; add adjacent bigrams
    over the stopword-free sequence ('revenue from operations' -> revenue, operation, revenue_operation)."""
    text = _NUM_COMMA.sub("", text.lower())
    toks = [stem(t) for t in _TOKEN.findall(text) if t not in _STOP]
    if bigrams:
        toks = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    return toks
