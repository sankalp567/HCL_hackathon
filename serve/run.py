#!/usr/bin/env python3
"""Backup LLM host: serve the Q&A model with vLLM (OpenAI-compatible API) on one GPU of ramanujan.

    python3 run.py                  # first run creates .venv (Python 3.12 + vLLM), then serves on GPU 3
    python run.py                   # the same, from inside the activated .venv
    python3 run.py --check          # health check + one short completion against a running server
    python3 run.py --setup-only     # only create .venv and install vLLM

Reach it from the machine running the backend through an SSH tunnel (the server listens on 127.0.0.1):
    ssh -N -L 8001:127.0.0.1:8001 rama
    LLM_BACKEND=vllm LLM_BASE_URL=http://127.0.0.1:8001/v1 LLM_MODEL=qwen3-4b uvicorn app.main:app

Runs on the server's system Python 3.8 (standard library only) so it can bootstrap its own environment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
VENV_PY = VENV / "bin" / "python"
LOG = HERE / "vllm.log"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"))
    p.add_argument("--served-name", default="qwen3-4b", help="model name clients send (LLM_MODEL in the backend)")
    p.add_argument("--gpu", default=os.environ.get("VLLM_GPU", "3"), help="CUDA device index")
    p.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and use an SSH tunnel")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.30,
                   help="share of the card's memory to take; the GPU is shared with other users")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"),
                   help="bearer token clients must send; required when --host is not 127.0.0.1")
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--check", action="store_true")
    return p.parse_args()


def in_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV.resolve()


def ensure_venv() -> None:
    """Create .venv with vLLM on first use, then re-run this script inside it."""
    if in_venv():
        return
    ready = VENV_PY.exists() and subprocess.run([str(VENV_PY), "-c", "import vllm"],
                                                capture_output=True).returncode == 0
    if not ready:
        uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        if not Path(uv).exists():
            sys.exit("[run.py] uv not found: put the uv binary in ~/.local/bin/uv, then run again")
        print("[run.py] creating .venv and installing vLLM (first run only, a few minutes)", flush=True)
        subprocess.run([uv, "venv", "--python", "3.12", str(VENV)], check=True)
        subprocess.run([uv, "pip", "install", "--python", str(VENV_PY), "vllm"], check=True)
    os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve())] + sys.argv[1:])


def request(url: str, payload: dict | None = None, api_key: str | None = None, timeout: float = 5) -> tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as resp:
        return resp.status, resp.read().decode()


def check(args: argparse.Namespace) -> int:
    base = "http://127.0.0.1:{}".format(args.port)
    try:
        status, _ = request(base + "/health")
        print("[run.py] health:", status)
        body = {"model": args.served_name, "max_tokens": 5, "temperature": 0,
                "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}
        _, text = request(base + "/v1/chat/completions", body, args.api_key, timeout=120)
        print("[run.py] completion:", json.loads(text)["choices"][0]["message"]["content"].strip())
        return 0
    except Exception as exc:  # any failure means "not usable" for this check
        print("[run.py] not reachable:", exc)
        return 1


def tail(path: Path, lines: int = 30) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-lines:])
    except OSError:
        return ""


def serve(args: argparse.Namespace) -> int:
    if args.host != "127.0.0.1" and not args.api_key:
        sys.exit("[run.py] refusing to listen on {} without --api-key: the server is shared".format(args.host))
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", args.model, "--served-model-name", args.served_name,
           "--host", args.host, "--port", str(args.port), "--dtype", "bfloat16",
           "--max-model-len", str(args.max_model_len),
           "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.api_key:
        cmd += ["--api-key", args.api_key]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    print("[run.py] GPU {} | {} as '{}' | http://{}:{}/v1 | log {}".format(
        args.gpu, args.model, args.served_name, args.host, args.port, LOG), flush=True)

    with open(LOG, "a") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 30 * 60  # the first start downloads the model
        while time.time() < deadline:
            if proc.poll() is not None:
                print("[run.py] vLLM exited with code {}. Last log lines:\n{}".format(proc.returncode, tail(LOG)))
                return proc.returncode or 1
            try:
                if request("http://127.0.0.1:{}/health".format(args.port))[0] == 200:
                    break
            except Exception:  # not listening yet
                pass
            time.sleep(5)
        else:
            print("[run.py] not healthy after 30 minutes; see", LOG)
            proc.terminate()
            return 1
        print("[run.py] READY. Tunnel from the backend host: ssh -N -L {0}:127.0.0.1:{0} rama".format(args.port),
              flush=True)
        return proc.wait()
    except KeyboardInterrupt:
        print("[run.py] stopping vLLM", flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


def main() -> int:
    args = parse_args()
    if args.check:
        return check(args)
    ensure_venv()
    if args.setup_only:
        print("[run.py] .venv ready:", VENV)
        return 0
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())
