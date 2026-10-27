"""A small on-disk cache for what costs a model call: router plans and checked model answers.

Keys are exact: the question with case, spacing and end punctuation normalised, plus everything else that changes
the result (company filter, facts.db version, models, prompt). Nothing is matched by similarity: 'TCS revenue FY25'
and 'TCS revenue FY26' look alike but need different answers.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path


def normalise(question: str) -> str:
    return " ".join(question.lower().split()).rstrip(" ?.!")


class Cache:
    """Key-value pairs in memory, appended to a JSON-lines file so they survive a restart."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, object] | None = None

    @staticmethod
    def key(parts) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:32]

    def _loaded(self) -> dict[str, object]:
        if self._data is None:
            self._data = {}
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []   # no cache yet
            for line in lines:
                try:
                    entry = json.loads(line)
                    self._data[entry["key"]] = entry["value"]
                except (ValueError, KeyError, TypeError):
                    continue  # a damaged line costs only its own entry
        return self._data

    def get(self, parts) -> object | None:
        with self._lock:
            return self._loaded().get(self.key(parts))

    def put(self, parts, value: object) -> None:
        key = self.key(parts)
        with self._lock:
            self._loaded()[key] = value
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
            except OSError:
                pass  # the cache must never break an answer; this entry then lives in memory only
