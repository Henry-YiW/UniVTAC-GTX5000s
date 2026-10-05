"""HTTP inference server for tactileVLA deploy-time evaluation.

Run this file with the RTac1 repo's ``.venv/bin/python``. The IsaacLab eval
process talks to it over HTTP so the simulator and model dependency stacks
can stay in separate Python environments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# JAX on the GPU would preallocate most of the memory Isaac Sim needs; the PyTorch model does not use it.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("USE_SWANLAB", "false")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def serve(host: str, port: int, authkey: str) -> None:
    root = _repo_root()
    os.chdir(root)
    sys.path.insert(0, str(root))
    from policy.TactileVLA.deploy_policy import (
        _from_wire_payload,
        _TactileVLALocalPolicy,
        _to_wire_payload,
    )

    state = {"model": None}
    model_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        timeout = 5

        def log_message(self, format, *args):
            pass

        def _reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def handle_error(self, request, client_address):
            exc = sys.exc_info()[1]
            if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
                return
            super().handle_error(request, client_address)

        def _authorized(self, payload: dict | None = None) -> bool:
            if not authkey:
                return True
            return authkey in (self.headers.get("X-TactileVLA-Auth"), (payload or {}).get("authkey"))

        def do_GET(self):
            if self.path != "/health":
                self._reply(404, {"detail": "not found"})
            elif not self._authorized():
                self._reply(401, {"detail": "authentication failed"})
            else:
                self._reply(200, {"status": "ok", "model_loaded": state["model"] is not None})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not self._authorized(payload):
                self._reply(401, {"detail": "authentication failed"})
                return
            try:
                if self.path == "/init":
                    with model_lock:
                        if state["model"] is None:
                            state["model"] = _TactileVLALocalPolicy(_from_wire_payload(payload["args"]))
                    self._reply(200, {"type": "ok"})
                elif self.path == "/act":
                    with model_lock:
                        if state["model"] is None:
                            self._reply(409, {"detail": "model is not initialized"})
                            return
                        action = state["model"].get_action(_from_wire_payload(payload["observation"]))
                    self._reply(200, {"type": "ok", "action": _to_wire_payload(action)})
                elif self.path == "/reset":
                    with model_lock:
                        if state["model"] is not None:
                            state["model"].reset()
                    self._reply(200, {"type": "ok"})
                elif self.path == "/shutdown":
                    self._reply(200, {"type": "ok"})
                    threading.Thread(target=server.shutdown, daemon=True).start()
                else:
                    self._reply(404, {"detail": "not found"})
            except BaseException as exc:
                self._reply(200, {"type": "error", "error": repr(exc), "traceback": traceback.format_exc()})

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"[TactileVLA] Server listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="tactileVLA HTTP inference server")
    parser.add_argument("--host", default=os.environ.get("TACTILEVLA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=os.environ.get("TACTILEVLA_PORT"))
    parser.add_argument("--authkey", default=os.environ.get("TACTILEVLA_AUTHKEY", ""))
    args = parser.parse_args()

    if args.port is None:
        raise ValueError("TACTILEVLA_PORT or --port is required")
    serve(args.host, int(args.port), args.authkey)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
