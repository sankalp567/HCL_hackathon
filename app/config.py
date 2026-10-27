"""Settings, read from the environment once when the app is created."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLM_BACKENDS = ("extractive", "transformers", "vllm")


@dataclass(frozen=True)
class Settings:
    facts_db: Path
    reports_dir: Path
    log_dir: Path
    llm_backend: str            # 'extractive' | 'transformers' | 'vllm'
    llm_model_path: Path        # transformers: local model directory
    llm_device: str             # transformers: torch device
    llm_base_url: str           # vllm: OpenAI-compatible endpoint
    llm_model: str              # vllm: served model name
    llm_api_key: str | None     # vllm: optional bearer token
    llm_timeout_s: float        # vllm: connect/read timeout


def _repo_path(name: str, default: str) -> Path:
    path = Path(os.environ.get(name, default)).expanduser()
    return path if path.is_absolute() else ROOT / path


def get_settings() -> Settings:
    backend = os.environ.get("LLM_BACKEND", "extractive")
    if backend not in LLM_BACKENDS:
        raise ValueError(f"LLM_BACKEND={backend!r} is not valid; use one of {', '.join(LLM_BACKENDS)}")
    return Settings(
        facts_db=_repo_path("FACTS_DB", "data/facts.db"),
        reports_dir=_repo_path("REPORTS_DIR", "data/reports"),
        log_dir=_repo_path("LOG_DIR", "data/logs"),
        llm_backend=backend,
        llm_model_path=Path(os.environ.get("LLM_MODEL_PATH", "~/work/hcl_extract/llm/Qwen3-4B-Instruct-2507")).expanduser(),
        llm_device=os.environ.get("LLM_DEVICE", "cuda:0"),
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
        llm_model=os.environ.get("LLM_MODEL", "qwen3-4b"),
        llm_api_key=os.environ.get("LLM_API_KEY") or None,
        llm_timeout_s=float(os.environ.get("LLM_TIMEOUT_S", "60")),
    )
