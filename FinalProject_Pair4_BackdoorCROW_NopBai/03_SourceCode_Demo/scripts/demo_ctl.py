#!/usr/bin/env python3
"""Supervisor daemon for the Pair 4 GPU service (demo_api.py :8100).

The Docker backend container cannot start host processes directly, so this
small control server (port 8101, Bearer token = PAIR4_GPU_TOKEN) starts /
stops the uvicorn GPU worker as a detached child process (survives SSH and
Hermes sessions; pidfile + logs under /tmp and pair4/logs/).

Endpoints (all require `Authorization: Bearer <PAIR4_GPU_TOKEN>`):
  GET  /status  -> {running, online, pid, detail}
  POST /start   -> spawn uvicorn child, wait for startup, {ok, detail}
  POST /stop    -> SIGTERM (SIGKILL fallback) the child, {ok, detail}

Run (detached, once):
  PAIR4_GPU_TOKEN=... nohup python3 \
      demo_ctl.py > pair4/logs/demo_api_ctl_stdout.log 2>&1 &
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

BASE = "<REPO_ROOT>"
PAIR = os.path.join(BASE, "Final_Project_Attack_Defense", "pair4")
SCRIPTS = os.path.join(PAIR, "scripts")
LOGS = os.path.join(PAIR, "logs")
VENV_PY = "python3"
PIDFILE = "/tmp/pair4_gpu.pid"
CHILD_LOG = os.path.join(LOGS, "demo_api_child.log")
CTL_PORT = int(os.environ.get("PAIR4_CTL_PORT", "8101"))
TARGET_PORT = 8100
TOKEN = os.environ.get("PAIR4_GPU_TOKEN", "pair4-local-demo")

if not os.path.exists(VENV_PY):
    print(f"[fatal] venv python not found: {VENV_PY}", flush=True)
    sys.exit(1)


def log(msg: str) -> None:
    print(f"[ctl {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def read_pid() -> int | None:
    try:
        with open(PIDFILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def _probe_online() -> bool:
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{TARGET_PORT}/health")
        req.add_header("Authorization", f"Bearer {TOKEN}")
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _status() -> dict:
    pid = read_pid()
    running = pid is not None and is_alive(pid)
    online = _probe_online()
    return {"running": running, "online": online,
            "pid": pid if running else None,
            "detail": ("child alive" if running else "no child")
                       + (f", port {TARGET_PORT} responding" if online
                          else f", port {TARGET_PORT} not responding")}


def _start() -> dict:
    st = _status()
    if st["running"] or st["online"]:
        return {"ok": False, "msg": f"already running (pid {st['pid']})"}
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, PAIR4_GPU_TOKEN=TOKEN)
    with open(CHILD_LOG, "a") as lf:
        proc = subprocess.Popen(
            [VENV_PY, "-m", "uvicorn", "demo_api:app",
             "--host", "0.0.0.0", "--port", str(TARGET_PORT),
             "--log-level", "info"],
            cwd=SCRIPTS, env=env, stdout=lf, stderr=subprocess.STDOUT,
            start_new_session=True)
    with open(PIDFILE, "w") as f:
        f.write(str(proc.pid))
    log(f"spawned uvicorn pid={proc.pid}")
    for _ in range(60):  # up to ~60s for startup
        if _probe_online():
            return {"ok": True, "msg": f"started pid {proc.pid}",
                    "status": _status()}
        if not is_alive(proc.pid):
            return {"ok": False, "msg": "child exited early — see "
                    f"{CHILD_LOG}"}
        time.sleep(1)
    return {"ok": False, "msg": "startup timeout (model loads later — "
            "service is up), port still probing", "status": _status()}


def _stop() -> dict:
    pid = read_pid()
    if pid is None or not is_alive(pid):
        return {"ok": False, "msg": "no running child (pidfile stale?)"}
    os.kill(pid, signal.SIGTERM)
    for _ in range(10):
        if not is_alive(pid):
            break
        time.sleep(0.5)
    else:
        os.kill(pid, signal.SIGKILL)
    try:
        os.remove(PIDFILE)
    except OSError:
        pass
    log(f"stopped pid {pid}")
    return {"ok": True, "msg": f"stopped pid {pid}"}


def _handle(method: str, path: str, headers, body: bytes) -> tuple[int, dict]:
    auth = headers.get("authorization", "")
    if auth != f"Bearer {TOKEN}":
        return 401, {"error": "missing or invalid PAIR4_GPU_TOKEN"}
    if path == "/status" and method == "GET":
        return 200, _status()
    if path == "/start" and method == "POST":
        return 200, _start()
    if path == "/stop" and method == "POST":
        return 200, _stop()
    return 404, {"error": "not found"}


def main() -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # silence default logging
            pass

        def _respond(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._respond(*_handle("GET", self.path.split("?")[0],
                                   self.headers, b""))

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            self._respond(*_handle("POST", self.path.split("?")[0],
                                   self.headers, body))

    srv = ThreadingHTTPServer(("0.0.0.0", CTL_PORT), H)
    log(f"control server listening on 0.0.0.0:{CTL_PORT} (pid {os.getpid()})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
