import pytest

from extract.numbers import (detect_unit, format_value, fy_from_header, normalise_label, parse_fy, parse_number,
                             printed_number)


@pytest.mark.parametrize("text, expected", [
    ("1,78,650", (178650.0, False)),          # Indian grouping (Infosys, TCS)
    ("130,144", (130144.0, False)),           # Western grouping (HCLTech)
    ("(24)", (-24.0, False)),                 # bracket negative
    ("(1,234)", (-1234.0, False)),
    ("-12.5", (-12.5, False)),
    ("21.0%", (21.0, True)),
    ("₹ 43,075 cr", (43075.0, False)),
    ("29,440#", (29440.0, False)),            # footnote mark
    ("H 2,67,021", (267021.0, False)),        # TCS font: rupee sign extracted as 'H'
    ("0.81", (0.81, False)),
    ("3.19", (3.19, False)),
    ("6, 17 437", (617437.0, False)),         # TCS letter-spaced digits, repaired because commas mark one number
    ("6, 02, 101", (602101.0, False)),
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["-", "—", "", "Revenue", "1.2x", "Nil", "FY 2026 vs FY 2025",
                                  "97 5", "13."])   # TCS text layer lost the decimal point / a digit
def test_parse_number_rejects_non_numbers(text):
    assert parse_number(text) is None


def test_printed_number():
    assert printed_number("₹ 43,075 cr") == "43,075"
    assert printed_number("(1,234)") == "1,234"
    assert printed_number("n/a") is None


@pytest.mark.parametrize("text, expected", [
    ("FY 2026", "FY2025-26"),
    ("FY26", "FY2025-26"),
    ("FY 2025-26", "FY2025-26"),
    ("2025-26", "FY2025-26"),
    ("Year ended - 31 March 2026", "FY2025-26"),
    ("As at March 31, 2025", "FY2024-25"),
    ("ear ended March 31, 2026 - Intrinsic value", "FY2025-26"),   # TCS text layer drops the 'Y'
    ("fiscal 2026", "FY2025-26"),
    ("2026", "FY2025-26"),
])
def test_parse_fy(text, expected):
    assert parse_fy(text) == expected


@pytest.mark.parametrize("text", ["Q4 FY26", "Quarter ended March 31, 2026", "Note No.", "2021-2025", ""])
def test_parse_fy_rejects(text):
    assert parse_fy(text) is None


@pytest.mark.parametrize("label, expected", [
    ("Consolidated - 2026", "FY2025-26"),                                # Infosys p. 81 ratios table
    ("(In ` crore) - Year ended March 31, - 2025", "FY2024-25"),         # Infosys p. 333
    ("FY 2026 - Total (A)", "FY2025-26"),
    ("Particulars", None),
])
def test_fy_from_header(label, expected):
    assert fy_from_header(label) == expected


@pytest.mark.parametrize("text, expected", [
    ("In ₹ crore, except per equity share data", "INR crore"),
    ("(All amounts in crores of ₹, except share data)", "INR crore"),
    ("Fair value (J crore)", "INR crore"),                 # TCS: rupee sign extracted as 'J'
    ("In US$ million, except per equity share data", "USD million"),
    ("Operating margin (%)", "%"),
    ("Revenue from operations", None),
])
def test_detect_unit(text, expected):
    assert detect_unit(text) == expected


def test_detect_unit_without_percent():
    assert detect_unit("Revenue growth % (In ₹ crore)", allow_percent=False) == "INR crore"


@pytest.mark.parametrize("value, unit, expected", [
    (267021, "INR crore", "₹2,67,021 crore"),
    (130144, "INR crore", "₹1,30,144 crore"),
    (25.0, "%", "25.0%"),
    (12.51, "%", "12.51%"),
    (617437, "count", "6,17,437 employees"),
    (20158, "USD million", "US$20,158 million"),
])
def test_format_value(value, unit, expected):
    assert format_value(value, unit) == expected


def test_normalise_label():
    assert normalise_label("Net profit #") == "net profit"
    assert normalise_label("Revenue from operations (2)") == "revenue from operations"
    assert normalise_label("  Research &  Development ") == "research and development"
