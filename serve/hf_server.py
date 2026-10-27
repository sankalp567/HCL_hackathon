"""Primary LLM host (sustain): an OpenAI-compatible chat endpoint for Qwen3-4B-Instruct-2507 using transformers.

Implements only what the backend calls: POST /v1/chat/completions (streaming or not), GET /health and /v1/models.
Uses the standard-library HTTP server because the GPU server has no internet to install a web framework.

    CUDA_VISIBLE_DEVICES=0 python hf_server.py --model llm/Qwen3-4B-Instruct-2507 --port 8002

The backend reaches it through an SSH tunnel (ssh -N -L 8002:127.0.0.1:8002 sustain) with
LLM_BACKEND=vllm LLM_BASE_URL=http://127.0.0.1:8002/v1 LLM_MODEL=qwen3-4b, the same client as the vLLM backup.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class Model:
    def __init__(self, path: str, name: str) -> None:
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")
        self.lock = threading.Lock()  # one generation at a time on one GPU

    def stream(self, messages: list[dict], max_tokens: int, temperature: float) -> Iterator[str]:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_tokens, do_sample=temperature > 0)
        if temperature > 0:
            kwargs["temperature"] = temperature
        with self.lock:
            worker = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
            worker.start()
            for piece in streamer:
                if piece:
                    yield piece
            worker.join()


def make_handler(model: Model, api_key: str | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(time.strftime("%H:%M:%S"), self.address_string(), fmt % args, flush=True)

        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                return self.send_json(200, {"status": "ok", "model": model.name})
            if self.path == "/v1/models":
                return self.send_json(200, {"object": "list", "data": [{"id": model.name, "object": "model"}]})
            self.send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                return self.send_json(404, {"error": "not found"})
            if api_key and self.headers.get("Authorization") != f"Bearer {api_key}":
                return self.send_json(401, {"error": "missing or wrong API key"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            messages = body.get("messages") or []
            max_tokens = min(int(body.get("max_tokens") or 400), 1024)
            temperature = float(body.get("temperature") or 0)
            completion_id = "chatcmpl-" + uuid.uuid4().hex[:12]
            if not body.get("stream"):
                text = "".join(model.stream(messages, max_tokens, temperature))
                return self.send_json(200, {"id": completion_id, "object": "chat.completion", "model": model.name,
                                            "choices": [{"index": 0, "finish_reason": "stop",
                                                         "message": {"role": "assistant", "content": text}}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(delta: dict, finish: str | None) -> None:
                chunk = {"id": completion_id, "object": "chat.completion.chunk", "model": model.name,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            try:
                for piece in model.stream(messages, max_tokens, temperature):
                    send({"content": piece}, None)
                send({}, "stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # the client went away; the lock is released when the generator closes

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local model directory")
    parser.add_argument("--served-name", default="qwen3-4b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    if args.host != "127.0.0.1" and not args.api_key:
        parser.error("listening beyond 127.0.0.1 needs --api-key (the server is shared)")
    started = time.time()
    model = Model(args.model, args.served_name)
    print(f"loaded {args.model} in {time.time() - started:.0f}s; serving http://{args.host}:{args.port}/v1", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(model, args.api_key)).serve_forever()


if __name__ == "__main__":
    main()
