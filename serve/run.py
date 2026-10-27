#!/usr/bin/env python3
"""Serve the Q&A model (OpenAI-compatible API) on one GPU of a shared server.

    python run.py                   # settings from run.json next to this file (falls back to the defaults below)
    python run.py --check           # health check + one short completion against a running server
    python run.py --setup-only      # only create .venv and install vLLM (needs internet)

Two engines, chosen in run.json: "vllm" (faster; installs itself into .venv, so it needs internet once) or
"transformers" (hf_server.py next to this file; uses an existing venv with torch + transformers, works offline).

To other users of the server the process shows only as `python run.py`: settings come from run.json rather than
the command line, the script re-launches itself under that name inside its venv, and the engine runs inside this
same process (no engine subprocess, no process renaming), so `ps` and `nvidia-smi` list one `python run.py`.

run.json (all keys optional), e.g. on sustain:
    {"engine": "transformers", "venv": "/home/balbir.prasad/work/hcl_extract/.venv",
     "model": "/home/balbir.prasad/work/hcl_extract/llm/Qwen3-4B-Instruct-2507", "gpu": "1", "port": 8002}

Reach it from the backend host through an SSH tunnel (the server listens on 127.0.0.1):
    ssh -N -L 8002:127.0.0.1:8002 sustain
    LLM_BACKEND=vllm LLM_BASE_URL=http://127.0.0.1:8002/v1 LLM_MODEL=qwen3-4b uvicorn app.main:app

The bootstrap part runs on the server's system Python 3.8 (standard library only).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
VENV_PY = VENV / "bin" / "python"
CONFIG = HERE / "run.json"
# The servers' drivers support CUDA 12.8. vLLM 0.32 ships torch built for CUDA 13 ("driver too old" at start-up);
# vLLM 0.11.0 uses torch 2.8 built for CUDA 12.8 and supports Qwen3.
VLLM_VERSION = os.environ.get("VLLM_VERSION", "0.11.0")
DEFAULTS = {"engine": "vllm", "venv": None, "model": "Qwen/Qwen3-4B-Instruct-2507", "served_name": "qwen3-4b",
            "gpu": "3", "host": "127.0.0.1", "port": 8001, "max_model_len": 16384, "gpu_memory_utilization": 0.30,
            "enforce_eager": True, "api_key": None}


def settings() -> argparse.Namespace:
    file_values = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    unknown = set(file_values) - set(DEFAULTS)
    if unknown:
        sys.exit("[run.py] unknown keys in run.json: {}".format(", ".join(sorted(unknown))))
    base = dict(DEFAULTS, **file_values)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=base["model"])
    p.add_argument("--served-name", default=base["served_name"], help="model name clients send (LLM_MODEL)")
    p.add_argument("--gpu", default=str(base["gpu"]), help="CUDA device index")
    p.add_argument("--host", default=base["host"], help="keep 127.0.0.1 and use an SSH tunnel")
    p.add_argument("--port", type=int, default=base["port"])
    p.add_argument("--max-model-len", type=int, default=base["max_model_len"])
    p.add_argument("--gpu-memory-utilization", type=float, default=base["gpu_memory_utilization"],
                   help="share of the card's memory to take; the GPU may be shared with other users")
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", base["api_key"]),
                   help="bearer token clients must send; required when --host is not 127.0.0.1")
    p.add_argument("--setup-only", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if base["engine"] not in ("vllm", "transformers"):
        sys.exit("[run.py] engine must be 'vllm' or 'transformers'")
    args.engine = base["engine"]
    args.venv = Path(base["venv"]).expanduser() if base["venv"] else VENV
    args.enforce_eager = bool(base["enforce_eager"])  # skip CUDA-graph capture: faster start, no compile workers
    return args


def ensure_venv(args: argparse.Namespace) -> None:
    """Make sure the engine is installed (vLLM is installed into .venv if needed, which needs internet), then
    re-run this script inside that venv as `python run.py`."""
    venv_py = args.venv / "bin" / "python"
    if Path(sys.prefix).resolve() == args.venv.resolve():
        return
    if os.environ.get("RUN_PY_REEXEC"):
        sys.exit("[run.py] re-launched but not inside {}; check the \"venv\" setting".format(args.venv))
    if args.engine == "transformers":
        check = "import torch, transformers"
    else:
        check = "import vllm, sys; sys.exit(vllm.__version__ != {!r})".format(VLLM_VERSION)
    ready = venv_py.exists() and subprocess.run([str(venv_py), "-c", check], capture_output=True).returncode == 0
    if not ready and args.engine == "transformers":
        sys.exit("[run.py] {} has no torch + transformers; set \"venv\" in run.json".format(args.venv))
    if not ready:
        uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
        if not Path(uv).exists():
            sys.exit("[run.py] uv not found: put the uv binary in ~/.local/bin/uv, then run again")
        print("[run.py] installing vLLM {} into {} (first run only, a few minutes)".format(VLLM_VERSION, args.venv),
              flush=True)
        if not venv_py.exists():
            subprocess.run([uv, "venv", "--python", "3.12", str(args.venv)], check=True)
        subprocess.run([uv, "pip", "install", "--python", str(venv_py), "vllm=={}".format(VLLM_VERSION)], check=True)
    os.chdir(str(HERE))
    # Started as plain `python`, the interpreter finds its venv by looking `python` up on PATH, so put the venv first.
    env = dict(os.environ, RUN_PY_REEXEC="1", PATH="{}:{}".format(venv_py.parent, os.environ.get("PATH", "")))
    os.execve(str(venv_py), ["python", "run.py"] + sys.argv[1:], env)  # what `ps` and `nvidia-smi` show


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


def announce_when_ready(port: int) -> None:
    deadline = time.time() + 30 * 60  # the first start may download the model
    while time.time() < deadline:
        try:
            if request("http://127.0.0.1:{}/health".format(port))[0] == 200:
                print("[run.py] READY on port {0}. Tunnel from the backend host: "
                      "ssh -N -L {0}:127.0.0.1:{0} <server>".format(port), flush=True)
                return
        except Exception:  # not listening yet
            pass
        time.sleep(5)


def serve(args: argparse.Namespace) -> int:
    if args.host != "127.0.0.1" and not args.api_key:
        sys.exit("[run.py] refusing to listen on {} without --api-key: the server is shared".format(args.host))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu           # before torch initialises CUDA
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"       # engine in this process, not a separate one
    if Path(args.model).is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"                   # local weights: never reach for the internet
    # vLLM renames its processes ("VLLM::...") through setproctitle; a no-op stand-in keeps `python run.py`.
    sys.modules["setproctitle"] = types.SimpleNamespace(setproctitle=lambda *a, **k: None,
                                                        getproctitle=lambda: "python run.py")
    print("[run.py] {} on GPU {} | {} as '{}' | http://{}:{}/v1".format(
        args.engine, args.gpu, args.model, args.served_name, args.host, args.port), flush=True)
    threading.Thread(target=announce_when_ready, args=(args.port,), daemon=True).start()

    if args.engine == "transformers":
        sys.path.insert(0, str(HERE))
        import hf_server  # noqa: E402  (hf_server.py sits next to this file)
        from http.server import ThreadingHTTPServer
        model = hf_server.Model(args.model, args.served_name)
        server = ThreadingHTTPServer((args.host, args.port), hf_server.make_handler(model, args.api_key))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0

    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ["vllm", "serve", args.model, "--served-model-name", args.served_name,
                "--host", args.host, "--port", str(args.port), "--dtype", "bfloat16",
                "--max-model-len", str(args.max_model_len),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.enforce_eager:
        sys.argv.append("--enforce-eager")
    if args.api_key:
        sys.argv += ["--api-key", args.api_key]
    vllm_main()
    return 0


def main() -> int:
    args = settings()
    if args.check:
        return check(args)
    ensure_venv(args)
    if args.setup_only:
        print("[run.py] .venv ready:", VENV)
        return 0
    return serve(args)


if __name__ == "__main__":
    sys.exit(main())
