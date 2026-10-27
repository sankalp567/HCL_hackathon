"""Numbers, units and fiscal years as the three annual reports print them.

Infosys and TCS use Indian digit grouping (1,78,650), HCLTech Western grouping (130,144),
negatives are printed in brackets, and TCS's fonts put the letters 'H' or 'J' in the text layer
where the rupee sign is printed. Everything here is a pure function so it can be unit-tested.
"""
from __future__ import annotations

import re

_CURRENCY = re.compile(r"^(?:₹|rs\.?|inr|us\s?\$|usd|\$|[HJ](?=\s*[\d(]))\s*", re.IGNORECASE)
_SCALE_SUFFIX = re.compile(r"\s*(?:crores?|cr|lakhs?|million|mn|billion|bn)\.?$", re.IGNORECASE)
_NUMBER = re.compile(r"^-?\(?-?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?\)?$")
_NUMBER_IN_TEXT = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")


def parse_number(text: str) -> tuple[float, bool] | None:
    """Return (value, is_percent) for a cell such as '1,78,650', '(24)' or '21.0%', else None."""
    s = text.strip().replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.rstrip("*#†‡^¹²³⁴⁵").strip()
    is_percent = s.endswith("%")
    s = _CURRENCY.sub("", s.rstrip("%").strip())
    s = _SCALE_SUFFIX.sub("", s).replace(" ", "")
    if not _NUMBER.match(s):
        return None
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    value = float(s.strip("()-").replace(",", ""))
    return (-value if negative else value), is_percent


def printed_number(text: str) -> str | None:
    """The number as printed inside a cell ('₹ 43,075 cr' -> '43,075'), used for text-layer checks."""
    match = _NUMBER_IN_TEXT.search(text)
    return match.group(0).strip("()") if match else None


def fy_label(end_year: int) -> str:
    """2026 -> 'FY2025-26' (Indian fiscal years end on 31 March)."""
    return f"FY{end_year - 1}-{end_year % 100:02d}"


_PART_YEAR = re.compile(r"\bq[1-4]\b|quarter|\bh[12]\b|half[- ]year|nine months|\b9m\b", re.IGNORECASE)
_YEAR_RANGE = re.compile(r"\b(20\d{2})\s*[-–/]\s*(20\d{2}|\d{2})\b")
_FY = re.compile(r"\bfy\s*'?\s*(20\d{2}|\d{2})\b", re.IGNORECASE)
_MARCH_31 = re.compile(
    r"(?:march|mar)\.?\s*31(?:st)?,?\s*(20\d{2})|31(?:st)?\s*(?:march|mar)\.?,?\s*(20\d{2})", re.IGNORECASE)
_FISCAL = re.compile(r"\bfiscal\s*(?:year\s*)?(20\d{2})\b", re.IGNORECASE)
_BARE_YEAR = re.compile(r"^\s*(20\d{2})\s*$")


def parse_fy(text: str) -> str | None:
    """'FY 2026', 'FY26', '2025-26', '31 March 2026', 'fiscal 2026' -> 'FY2025-26'; quarters -> None."""
    if not text or _PART_YEAR.search(text):
        return None
    if match := _YEAR_RANGE.search(text):
        start, end = int(match.group(1)), int(match.group(2))
        end = end if end > 100 else 2000 + end
        if end == start + 1:
            return fy_label(end)
    for pattern in (_FY, _MARCH_31, _FISCAL, _BARE_YEAR):
        if match := pattern.search(text):
            year = int(next(g for g in match.groups() if g))
            return fy_label(year if year > 100 else 2000 + year)
    return None


_USD_MILLION = re.compile(r"(?:us\s?\$|usd)\s*(?:in\s*)?(?:million|mn)|(?:million|mn)\s*(?:of\s*)?(?:us\s?\$|usd)")
_USD_BILLION = re.compile(r"(?:us\s?\$|usd)\s*(?:in\s*)?(?:billion|bn)|(?:billion|bn)\s*(?:of\s*)?(?:us\s?\$|usd)")
_INR_CRORE = re.compile(r"\bcrores?\b|\bcr\b")
_INR_LAKH = re.compile(r"\blakhs?\b")
_INR_MILLION = re.compile(r"(?:₹|rs\.?|inr)\s*(?:in\s*)?(?:million|mn)")


def detect_unit(text: str, allow_percent: bool = True) -> str | None:
    """Unit named in a label or heading: '%', 'USD million', 'USD billion', 'INR crore', 'INR lakh', 'INR million'.

    allow_percent=False for table-wide context, where a stray '%' says nothing about the values.
    A bare 'crore' is taken as rupees: the reports never use crore for anything else.
    """
    t = (text or "").lower()
    if allow_percent and ("%" in t or "per cent" in t or "percentage" in t):
        return "%"
    for pattern, unit in ((_USD_MILLION, "USD million"), (_USD_BILLION, "USD billion"), (_INR_CRORE, "INR crore"),
                          (_INR_LAKH, "INR lakh"), (_INR_MILLION, "INR million")):
        if pattern.search(t):
            return unit
    return None


_FOOTNOTE_MARKS = re.compile(r"[*#†‡^¹²³⁴⁵]+|\(\d\)$|\s\d$")


def normalise_label(text: str) -> str:
    """Lower-case, '&' -> 'and', footnote marks removed, spaces collapsed: for matching row labels."""
    t = (text or "").replace("&", " and ").replace("’", "'")
    t = _FOOTNOTE_MARKS.sub("", t.strip())
    return re.sub(r"\s+", " ", t).strip(" :-–.,").lower()
