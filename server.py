"""cuda-jupyter notebook server.

    .venv/bin/python server.py            # http://127.0.0.1:8888

Serves the notebook UI, executes cells through engine.Engine and exposes a
real shell over a WebSocket for the terminal pane.
"""
import asyncio
import fcntl
import hashlib
import json
import os
import pty
import re
import shutil
import subprocess
import tempfile
import signal
import struct
import sys
import termios
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engine import Engine

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
NOTEBOOK = ROOT / "notebook.ipynb"
DEFAULT_NOTEBOOK = ROOT / "examples" / "cuda_jupyter_tour.ipynb"

app = FastAPI()
engine = Engine()
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def no_stale_ui(request: Request, call_next):
    # Revalidate the UI on every load so a redeploy shows up on a plain reload.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = "mps (Apple GPU); traced on cpu"
    else:
        device = "cpu"
    return {"torch": torch.__version__, "python": sys.version.split()[0], "device": device,
            "notebook": NOTEBOOK.name, "shell": Path(login_shell()).name, "gpu": gpu_spec(),
            "nvcc": bool(find_nvcc())}


# Per-architecture SM layout that torch does not report. fp32/int32/tc are per SM;
# partitions = warp schedulers (processing blocks) per SM.
ARCH = {
    (6, 0): dict(family="Pascal", fp32=64, int32=0, tc=0, partitions=2, max_warps=64, max_blocks=32, smem_kb=64),
    (6, 1): dict(family="Pascal", fp32=128, int32=0, tc=0, partitions=4, max_warps=64, max_blocks=32, smem_kb=96),
    (7, 0): dict(family="Volta", fp32=64, int32=64, tc=8, partitions=4, max_warps=64, max_blocks=32, smem_kb=96),
    (7, 5): dict(family="Turing", fp32=64, int32=64, tc=8, partitions=4, max_warps=32, max_blocks=16, smem_kb=64),
    (8, 0): dict(family="Ampere", fp32=64, int32=64, tc=4, partitions=4, max_warps=64, max_blocks=32, smem_kb=164),
    (8, 6): dict(family="Ampere", fp32=128, int32=64, tc=4, partitions=4, max_warps=48, max_blocks=16, smem_kb=100),
    (8, 7): dict(family="Ampere", fp32=128, int32=64, tc=4, partitions=4, max_warps=48, max_blocks=16, smem_kb=164),
    (8, 9): dict(family="Ada", fp32=128, int32=64, tc=4, partitions=4, max_warps=48, max_blocks=24, smem_kb=100),
    (9, 0): dict(family="Hopper", fp32=128, int32=64, tc=4, partitions=4, max_warps=64, max_blocks=32, smem_kb=228),
    (10, 0): dict(family="Blackwell", fp32=128, int32=64, tc=4, partitions=4, max_warps=64, max_blocks=32, smem_kb=228),
    (12, 0): dict(family="Blackwell", fp32=128, int32=64, tc=4, partitions=4, max_warps=48, max_blocks=24, smem_kb=100),
}
# Shown when there is no CUDA device (e.g. on a Mac): a Tesla T4, the Colab GPU.
T4 = dict(name="Tesla T4", major=7, minor=5, sms=40, total_memory=15 * 2**30, l2=4 * 2**20)


def gpu_spec():
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        dev = dict(name=p.name, major=p.major, minor=p.minor, sms=p.multi_processor_count,
                   total_memory=p.total_memory, l2=getattr(p, "L2_cache_size", 0), assumed=False)
    else:
        dev = dict(T4, assumed=True)
    arch = ARCH.get((dev["major"], dev["minor"])) or ARCH.get((dev["major"], 0)) or ARCH[(8, 0)]
    spec = dict(dev, **arch, warp=32, regs_per_sm=65536, max_regs_thread=255, max_threads_block=1024,
                sm=f"sm_{dev['major']}{dev['minor']}")
    spec["max_threads_sm"] = spec["max_warps"] * 32
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        spec["regs_per_sm"] = getattr(p, "regs_per_multiprocessor", 0) or spec["regs_per_sm"]
        spec["max_threads_sm"] = getattr(p, "max_threads_per_multi_processor", 0) or spec["max_threads_sm"]
        spec["max_warps"] = spec["max_threads_sm"] // 32
        smem = getattr(p, "shared_memory_per_multiprocessor", 0)
        if smem:
            spec["smem_kb"] = smem // 1024
    return spec


def find_nvcc():
    return shutil.which("nvcc") or next((p for p in ("/usr/local/cuda/bin/nvcc",) if os.path.exists(p)), None)


_resource_cache: dict = {}
_nvcc_lock = asyncio.Lock()


def _demangle(sym):
    m = re.match(r"_Z(\d+)(\w+)", sym)
    return m.group(2)[:int(m.group(1))] if m else sym


def nvcc_resources(code, arch):
    """Per-kernel registers / shared memory / spills, as reported by ptxas."""
    nvcc = find_nvcc()
    if not nvcc:
        return {"error": "nvcc not found"}
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "k.cu"
        src.write_text(code)
        try:
            r = subprocess.run([nvcc, f"-arch={arch}", "--resource-usage", "-cubin", "-o", str(Path(d) / "k.cubin"),
                                str(src)], capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return {"error": "nvcc timed out"}
    kernels, cur = {}, None
    for ln in (r.stdout + r.stderr).split("\n"):
        m = re.search(r"(?:Compiling entry function|Function properties for) '?([\w$]+)'?", ln)
        if m:
            cur = kernels.setdefault(_demangle(m.group(1)), {"regs": None, "smem": 0, "spill": 0})
            continue
        if cur is None:
            continue
        if m := re.search(r"Used (\d+) registers", ln):
            cur["regs"] = int(m.group(1))
        if m := re.search(r"(\d+) bytes smem", ln):
            cur["smem"] = int(m.group(1))
        if m := re.search(r"(\d+) bytes spill stores", ln):
            cur["spill"] = int(m.group(1))
    kernels = {k: v for k, v in kernels.items() if v["regs"] is not None}
    if r.returncode and not kernels:
        return {"error": (r.stderr or r.stdout).strip().split("\n")[-1][:300]}
    return {"kernels": kernels}


@app.post("/api/resources")
async def resources(request: Request):
    # Compiles off the kernel thread, so it never delays a running cell.
    body = await request.json()
    code, arch = body.get("code", ""), body.get("arch", "sm_75")
    key = hashlib.sha1(f"{arch}\0{code}".encode()).hexdigest()
    async with _nvcc_lock:
        if key not in _resource_cache:
            _resource_cache[key] = await asyncio.to_thread(nvcc_resources, code, arch)
    return _resource_cache[key]


@app.get("/api/notebook")
def load_notebook():
    path = NOTEBOOK if NOTEBOOK.exists() else DEFAULT_NOTEBOOK
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/example")
def example_notebook():
    return JSONResponse(json.loads(DEFAULT_NOTEBOOK.read_text()))


@app.put("/api/notebook")
async def save_notebook(request: Request):
    nb = await request.json()
    NOTEBOOK.write_text(json.dumps(nb, indent=1))
    return {"ok": True}


# Cells run as jobs: POST starts one (idempotent per client-chosen id, so the
# browser can retry after a dropped connection), GET long-polls for the result.
# No single HTTP request has to stay open for the whole cell, which matters for
# long cells, browsers that time out slow requests, and flaky tunnels.
executor = ThreadPoolExecutor(max_workers=1)  # FIFO, one cell at a time
jobs: dict = {}


@app.post("/api/execute")
async def execute(request: Request):
    body = await request.json()
    job_id = str(body.get("id") or uuid.uuid4())
    now = time.time()
    for k in [k for k, j in jobs.items() if j["done_at"] and now - j["done_at"] > 900]:
        del jobs[k]
    if job_id not in jobs:
        loop = asyncio.get_running_loop()
        job = {"done_at": None, "result": None, "event": asyncio.Event()}
        jobs[job_id] = job

        def finished(fut):
            try:
                job["result"] = fut.result()
            except Exception as e:  # engine.run reports cell errors itself; this is a bug path
                job["result"] = {"error": {"ename": type(e).__name__, "evalue": str(e),
                                           "traceback": [f"{type(e).__name__}: {e}\n"]}}
            job["done_at"] = time.time()
            loop.call_soon_threadsafe(job["event"].set)

        executor.submit(engine.run, body.get("code", "")).add_done_callback(finished)
    return {"id": job_id}


@app.get("/api/job/{job_id}")
async def job_status(job_id: str, wait: float = 20.0):
    job = jobs.get(job_id)
    if job is None:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    try:
        await asyncio.wait_for(job["event"].wait(), timeout=max(0.0, min(wait, 25.0)))
    except asyncio.TimeoutError:
        return {"done": False}
    return {"done": True, "result": job["result"]}


@app.post("/api/restart")
async def restart():
    # Queued behind any running cell, like a kernel restart after the current cell.
    await asyncio.wrap_future(executor.submit(engine.reset))
    return {"ok": True}


def login_shell():
    return os.environ.get("SHELL") or ("/bin/zsh" if os.path.exists("/bin/zsh") else "/bin/bash")


@app.websocket("/ws/terminal")
async def terminal(ws: WebSocket):
    await ws.accept()
    shell = login_shell()
    env = dict(os.environ)
    env.update(TERM="xterm-256color", COLORTERM="truecolor")
    if sys.prefix != sys.base_prefix:  # running from a virtualenv: activate it in the shell too
        env.update(PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
                   VIRTUAL_ENV=sys.prefix)
    cuda_bin = "/usr/local/cuda/bin"  # nvcc, so generated .cu files can be built from the terminal
    if os.path.isdir(cuda_bin) and cuda_bin not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = env.get("PATH", "") + os.pathsep + cuda_bin
    pid, fd = pty.fork()
    if pid == 0:  # child: become the shell
        os.chdir(ROOT)
        os.execvpe(shell, [shell, "-i"], env)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_readable():
        try:
            data = os.read(fd, 65536)
        except OSError:
            data = b""
        queue.put_nowait(data)
        if not data:
            loop.remove_reader(fd)

    loop.add_reader(fd, on_readable)

    async def pump():
        while True:
            data = await queue.get()
            if not data:
                await ws.close()
                return
            await ws.send_bytes(data)

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg["type"] == "input":
                os.write(fd, msg["data"].encode())
            elif msg["type"] == "resize":
                fcntl.ioctl(fd, termios.TIOCSWINSZ,
                            struct.pack("HHHH", int(msg["rows"]), int(msg["cols"]), 0, 0))
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass
    finally:
        pump_task.cancel()
        try:
            loop.remove_reader(fd)
        except Exception:
            pass
        try:
            os.kill(pid, signal.SIGHUP)
            os.waitpid(pid, os.WNOHANG)
        except (ProcessLookupError, ChildProcessError):
            pass
        os.close(fd)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8888"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
