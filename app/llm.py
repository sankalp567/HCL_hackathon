"""Answer writers. Every backend has `name` and `stream(system, user) -> Iterator[str]`.

- extractive:   no model; copies the source sentences that best overlap the question (laptop, tests, fallback)
- transformers: the self-hosted model loaded in-process on a GPU (Qwen3-4B-Instruct-2507 on sustain)
- vllm:         an OpenAI-compatible vLLM server, e.g. on 'ramanujan' through an SSH tunnel
"""
from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import httpx

from app.config import Settings
from app.retrieval import STOPWORDS

MAX_NEW_TOKENS = 400


class LLM(Protocol):
    name: str

    def stream(self, system: str, user: str) -> Iterator[str]: ...


class LLMUnavailable(RuntimeError):
    """The model could not be reached or loaded; the caller falls back to the extractive backend."""


# ---------------------------------------------------------------- extractive
SOURCE_LINE = re.compile(r"^\[(\d+)\] \([^)]*\) (.*)$", re.M)


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z0-9]+", text.lower())
    return {w.rstrip("s") for w in words if len(w) > 2 and w not in STOPWORDS}


class ExtractiveLLM:
    """Streams the 2-3 source sentences with the most question words, each with its [n] marker."""
    name = "extractive"

    def stream(self, system: str, user: str) -> Iterator[str]:
        question = user.split("\n", 1)[0].removeprefix("Question:").strip()
        wanted = _content_words(question)
        scored = []
        for m in SOURCE_LINE.finditer(user):
            n = int(m[1])
            for sentence in re.split(r"(?<=[.!?])\s+", m[2]):
                overlap = len(wanted & _content_words(sentence))
                if overlap:
                    scored.append((overlap, -n, sentence.strip(), n))
        if not scored:
            yield "NOT_FOUND"
            return
        picked: list[str] = []
        for _, _, sentence, n in sorted(scored, reverse=True):
            line = f"{sentence} [{n}]"
            if line not in picked:
                picked.append(line)
            if len(picked) == 3:
                break
        text = " ".join(picked)
        for i in range(0, len(text), 24):  # small chunks, like a model's tokens
            yield text[i:i + 24]


# ---------------------------------------------------------------- transformers (in-process GPU)
class TransformersLLM:
    """Loads the model once (lazily), then streams with TextIteratorStreamer; one generation at a time."""
    name = "transformers"

    def __init__(self, model_path: Path, device: str):
        self.model_path, self.device = model_path, device
        self._model = self._tokenizer = None
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            if not self.model_path.is_dir():
                raise LLMUnavailable(f"model directory not found: {self.model_path}")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=torch.bfloat16)
            self._model = model.to(self.device).eval()

    def stream(self, system: str, user: str) -> Iterator[str]:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        self.load()
        stop = threading.Event()

        class _StopWhenAsked(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return torch.full((input_ids.shape[0],), stop.is_set(), dtype=torch.bool, device=input_ids.device)

        with self._gen_lock:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            inputs = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
            ).to(self.device)
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
            failure: list[BaseException] = []

            def generate() -> None:
                try:
                    self._model.generate(**inputs, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         stopping_criteria=StoppingCriteriaList([_StopWhenAsked()]))
                except BaseException as exc:  # surfaced to the caller below
                    failure.append(exc)
                    streamer.end()

            worker = threading.Thread(target=generate, daemon=True)
            worker.start()
            try:
                for piece in streamer:
                    if piece:
                        yield piece
            finally:
                stop.set()          # the client went away or we finished: stop generating
                worker.join()
            if failure:
                raise LLMUnavailable(f"generation failed: {failure[0]!r}")


# ---------------------------------------------------------------- vLLM (OpenAI-compatible HTTP)
class OpenAICompatLLM:
    """Streams /chat/completions from a vLLM (or any OpenAI-compatible) server."""
    name = "vllm"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout_s: float = 60.0,
                 transport: httpx.BaseTransport | None = None):
        # one URL, or a comma-separated list tried in order (primary sustain, then the ramanujan safety net)
        self.base_urls = [u.strip().rstrip("/") for u in base_url.split(",") if u.strip()]
        self.base_url = self.base_urls[0]
        self.model, self.api_key = model, api_key
        self.timeout, self.transport = timeout_s, transport
        self.last_url: str | None = None  # the server that answered last

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Try each server in order; move on only if one fails before sending any text."""
        errors = []
        for url in self.base_urls:
            started = False
            try:
                for piece in self._stream_from(url, system, user):
                    started = True
                    yield piece
                self.last_url = url
                return
            except LLMUnavailable as exc:
                if started:
                    raise
                errors.append(str(exc))
        raise LLMUnavailable("; ".join(errors))

    def _stream_from(self, base_url: str, system: str, user: str) -> Iterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": self.model, "stream": True, "temperature": 0, "max_tokens": MAX_NEW_TOKENS,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                with client.stream("POST", f"{base_url}/chat/completions", json=body, headers=headers) as resp:
                    if resp.status_code != 200:
                        raise LLMUnavailable(f"{base_url} answered HTTP {resp.status_code}")
                    for line in resp.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        choices = json.loads(data).get("choices") or []
                        content = (choices[0].get("delta") or {}).get("content") if choices else None
                        if content:
                            yield content
        except httpx.HTTPError as exc:  # connection refused, timeout, broken stream
            raise LLMUnavailable(f"{base_url} unreachable: {exc!r}") from exc


_transformers_cache: dict[tuple[str, str], TransformersLLM] = {}


def get_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "transformers":
        key = (str(settings.llm_model_path), settings.llm_device)
        if key not in _transformers_cache:
            _transformers_cache[key] = TransformersLLM(settings.llm_model_path, settings.llm_device)
        return _transformers_cache[key]
    if settings.llm_backend == "vllm":
        return OpenAICompatLLM(settings.llm_base_url, settings.llm_model, settings.llm_api_key, settings.llm_timeout_s)
    return ExtractiveLLM()
