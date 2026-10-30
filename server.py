"""cuda-jupyter notebook server.

    .venv/bin/python server.py            # http://127.0.0.1:8888

Serves the notebook UI, executes cells through engine.Engine and exposes a
real shell over a WebSocket for the terminal pane.
"""
import asyncio
import fcntl
import json
import os
import pty
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
            "notebook": NOTEBOOK.name, "shell": Path(login_shell()).name}


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
