-- facts.db: everything extracted from the three annual reports, built once by extract/build_facts.py.
-- Coordinates (x0, y0, x1, y1) are PDF points with the origin at the TOP-LEFT of the page (PyMuPDF convention).
-- page_no is the physical PDF page, 1-based (what a PDF viewer shows), not the printed page number.

CREATE TABLE documents (
    doc_id      TEXT PRIMARY KEY,        -- e.g. 'TCS_FY2025-26', also the PDF file stem
    company     TEXT NOT NULL,           -- 'TCS' | 'Infosys' | 'HCLTech'
    title       TEXT NOT NULL,
    fy          TEXT NOT NULL,           -- 'FY2025-26'
    file        TEXT NOT NULL,           -- PDF file name in data/reports/
    sha256      TEXT NOT NULL,
    n_pages     INTEGER NOT NULL,
    url         TEXT,
    downloaded  TEXT
);

CREATE TABLE pages (
    doc_id        TEXT NOT NULL REFERENCES documents(doc_id),
    page_no       INTEGER NOT NULL,
    width         REAL NOT NULL,
    height        REAL NOT NULL,
    text          TEXT NOT NULL,         -- PyMuPDF text layer, used to verify numbers
    text_quality  TEXT NOT NULL,         -- 'ok' | 'letter_spaced' | 'image_only'
    PRIMARY KEY (doc_id, page_no)
);

-- Every Docling item: headings, paragraphs, list items, captions, tables, pictures.
CREATE TABLE elements (
    element_id  TEXT PRIMARY KEY,        -- '<doc_id>#/tables/12'
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    page_no     INTEGER NOT NULL,
    seq         INTEGER NOT NULL,        -- reading order within the document
    kind        TEXT NOT NULL,           -- 'section_header' | 'text' | 'list_item' | 'caption' | 'footnote' | 'table' | 'picture'
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    section     TEXT,                    -- nearest heading above the item
    text        TEXT,                    -- text; for tables the table context (heading, caption, unit line)
    extra       TEXT                     -- JSON: table size, picture class, ...
);

CREATE TABLE table_cells (
    element_id  TEXT NOT NULL REFERENCES elements(element_id),
    row         INTEGER NOT NULL,
    col         INTEGER NOT NULL,
    row_span    INTEGER NOT NULL,
    col_span    INTEGER NOT NULL,
    is_header   INTEGER NOT NULL,        -- 1 for column-header cells
    row_label   TEXT,                    -- 'section > label' of the row
    col_label   TEXT,                    -- column header text, top to bottom, joined with ' - '
    text        TEXT NOT NULL,           -- the cell exactly as extracted
    value       REAL,                    -- parsed number, or NULL
    unit        TEXT,                    -- 'INR crore' | 'USD million' | '%' | ... or NULL
    fy          TEXT,                    -- 'FY2025-26' from the column header, or NULL
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    PRIMARY KEY (element_id, row, col)
);

-- Values read from charts (optional chart pass); empty until then.
CREATE TABLE chart_points (
    element_id  TEXT NOT NULL REFERENCES elements(element_id),
    series      TEXT,
    category    TEXT,
    raw_label   TEXT,
    value       REAL,
    unit        TEXT,
    read_from   TEXT                     -- 'data_label' | 'axis_estimate'
);

-- One row per number for a canonical metric (see extract/metrics.yaml).
CREATE TABLE facts (
    fact_id          INTEGER PRIMARY KEY,
    company          TEXT NOT NULL,
    metric           TEXT NOT NULL,      -- canonical id, e.g. 'revenue'
    fy               TEXT NOT NULL,
    dimension        TEXT,               -- e.g. 'geography=Americas'; NULL for totals
    value            REAL NOT NULL,
    unit             TEXT NOT NULL,
    basis            TEXT NOT NULL,      -- 'consolidated' | 'standalone' | 'unknown'
    label_in_report  TEXT NOT NULL,      -- the row label as printed
    source_kind      TEXT NOT NULL,      -- 'table' | 'chart' | 'text'
    source_rank      INTEGER NOT NULL,   -- 1 financial statement, 2 highlights table, 3 other table, 4 text
    element_id       TEXT NOT NULL REFERENCES elements(element_id),
    cell_row         INTEGER,
    cell_col         INTEGER,
    doc_id           TEXT NOT NULL REFERENCES documents(doc_id),
    page_no          INTEGER NOT NULL,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    quote            TEXT NOT NULL,      -- 'row label | column label: cell text'
    in_text_layer    INTEGER NOT NULL,   -- 1 if the printed number is on the page's text layer
    agrees_with      INTEGER NOT NULL DEFAULT 0,  -- other sources with the same value
    conflicts_with   INTEGER NOT NULL DEFAULT 0,  -- other sources with a different value
    status           TEXT NOT NULL,      -- 'corroborated' | 'single_source' | 'conflict' | 'not_in_text' | 'verified'
    verified_by      TEXT,
    extractor        TEXT NOT NULL
);

-- Retrieval units: paragraphs grouped under their heading, and tables linearised row by row.
CREATE TABLE passages (
    passage_id  INTEGER PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    company     TEXT NOT NULL,
    page_no     INTEGER NOT NULL,
    kind        TEXT NOT NULL,           -- 'text' | 'table'
    element_id  TEXT,                    -- the table, or the first text item of the passage
    section     TEXT,
    text        TEXT NOT NULL
);

CREATE VIRTUAL TABLE passages_fts USING fts5(
    text, section, content='passages', content_rowid='passage_id', tokenize='porter unicode61'
);

CREATE TABLE metric_dictionary (
    metric  TEXT PRIMARY KEY,
    label   TEXT NOT NULL,
    unit    TEXT NOT NULL,
    spec    TEXT NOT NULL                -- the metric's entry from metrics.yaml, as JSON
);

-- The one fact the chart and the Q&A use per company × metric × year × unit: consolidated before unknown before
-- standalone, then checked status, then the most authoritative source, then the most agreement.
CREATE VIEW best_facts AS
SELECT * FROM (
    SELECT f.*, ROW_NUMBER() OVER (
        PARTITION BY company, metric, fy, unit, COALESCE(dimension, '')
        ORDER BY CASE basis WHEN 'consolidated' THEN 0 WHEN 'unknown' THEN 1 ELSE 2 END,
                 CASE status WHEN 'verified' THEN 0 WHEN 'corroborated' THEN 1 WHEN 'single_source' THEN 2
                             WHEN 'conflict' THEN 3 ELSE 4 END,
                 source_rank, agrees_with DESC, fact_id) AS choice
    FROM facts f)
WHERE choice = 1;

CREATE INDEX idx_elements_page ON elements(doc_id, page_no);
CREATE INDEX idx_cells_value ON table_cells(element_id) WHERE value IS NOT NULL;
CREATE INDEX idx_facts_lookup ON facts(company, metric, fy);
CREATE INDEX idx_passages_page ON passages(doc_id, page_no);
