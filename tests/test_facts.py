"""Checks on the built data/facts.db against values read by hand from the PDFs (skipped if it is not built)."""
import sqlite3
from pathlib import Path

import pytest

DB = Path(__file__).resolve().parent.parent / "data" / "facts.db"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="data/facts.db not built (python -m extract.build_facts)")


@pytest.fixture(scope="module")
def con():
    con = sqlite3.connect(DB)
    yield con
    con.close()


def best(con, company, metric, fy):
    return con.execute("SELECT value, unit, basis, page_no, in_text_layer FROM best_facts "
                       "WHERE company = ? AND metric = ? AND fy = ?", (company, metric, fy)).fetchone()


@pytest.mark.parametrize("company, fy, value, page", [
    ("TCS", "FY2025-26", 267021, 167),       # consolidated statement of profit and loss
    ("TCS", "FY2024-25", 255324, 167),
    ("Infosys", "FY2025-26", 178650, 333),
    ("Infosys", "FY2024-25", 162990, 333),   # not the standalone 1,36,592 on p. 209
    ("HCLTech", "FY2025-26", 130144, 375),
    ("HCLTech", "FY2024-25", 117055, 375),
])
def test_revenue_is_consolidated_and_cited(con, company, fy, value, page):
    v, unit, basis, page_no, in_text = best(con, company, "revenue", fy)
    assert (v, unit, basis, page_no, in_text) == (value, "INR crore", "consolidated", page, 1)


def test_tcs_headcount_repaired_from_letter_spaced_digits(con):
    value, unit, _, _, in_text = best(con, "TCS", "headcount", "FY2025-26")
    assert (value, unit, in_text) == (617437, "count", 1)   # printed '6, 17 437' = 5,84,519 + 32,918


def test_margins_come_from_each_companys_own_wording(con):
    assert best(con, "TCS", "operating_margin", "FY2025-26")[0] == 25.0       # 'operating margin of 25.0%'
    assert best(con, "HCLTech", "operating_margin", "FY2025-26")[0] == 17.2   # 'EBIT ... at 17.2% of revenue'
    # p. 81 'Key financial ratios': Standalone 21.9 | Consolidated 20.3 under one unflagged header row
    assert best(con, "Infosys", "operating_margin", "FY2025-26")[:3] == (20.3, "%", "consolidated")


def test_section_rows_do_not_leak_into_other_rows(con):
    leaked = con.execute("SELECT COUNT(*) FROM facts WHERE metric = 'operating_margin' "
                         "AND label_in_report LIKE '%discount%'").fetchone()[0]
    assert leaked == 0


def test_every_fact_has_provenance(con):
    missing = con.execute("SELECT COUNT(*) FROM facts WHERE page_no IS NULL OR quote = '' OR element_id IS NULL "
                          "OR x0 IS NULL").fetchone()[0]
    assert missing == 0


def test_search_index_finds_report_text(con):
    hits = con.execute("SELECT company FROM passages_fts JOIN passages ON passages.passage_id = passages_fts.rowid "
                       "WHERE passages_fts MATCH 'attrition' GROUP BY company").fetchall()
    assert {h[0] for h in hits} == {"TCS", "Infosys", "HCLTech"}
