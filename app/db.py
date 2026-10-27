"""Read-only access to facts.db."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    """Open facts.db read-only; one connection per request keeps threads apart."""
    if not path.is_file():
        raise FileNotFoundError(f"facts.db not found at {path}; build it with: python -m extract.build_facts")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
