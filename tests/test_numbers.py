import pytest

from extract.numbers import detect_unit, normalise_label, parse_fy, parse_number, printed_number


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
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["-", "—", "", "Revenue", "1.2x", "Nil", "FY 2026 vs FY 2025"])
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


def test_normalise_label():
    assert normalise_label("Net profit #") == "net profit"
    assert normalise_label("Revenue from operations (2)") == "revenue from operations"
    assert normalise_label("  Research &  Development ") == "research and development"
