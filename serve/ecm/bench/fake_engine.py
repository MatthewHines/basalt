"""Fake engines for the policy bench: real HTTP on an ephemeral loopback port, no GPU, no
podman, no card reads. They are fake ENGINES, not fake ECMs — the ECM (and the bench) talk
to them through serve/ecm/client.py, so the seam's laws (415, verbatim refusal, no faked
status) are exercised over the wire, not asserted against a mock that shares the caller's
assumptions. Every request is recorded for post-hoc assertion.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

JSON_TYPE = "application/json"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # silence the per-request stderr chatter
        pass

    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", JSON_TYPE)
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802 - stdlib naming
        eng = self.server.engine
        eng.seen.append({"method": "GET", "path": self.path, "content_type": None,
                         "payload": None})
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": eng.name, "object": "model"}]})
        elif self.path == "/v1/status":
            self._send(200, dict(eng.status))
        else:
            self._send(404, {"error": {"message": f"no such route {self.path}"}})

    def do_POST(self):  # noqa: N802
        eng = self.server.engine
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        ctype = (self.headers.get("content-type") or "").split(";")[0].strip().lower()
        eng.seen.append({"method": "POST", "path": self.path, "content_type": ctype or None,
                         "payload": raw.decode("utf-8", "replace") or None})
        if eng.enforce_json and ctype != JSON_TYPE:
            # the 11-27 law, modeled faithfully: no JSON type -> 415, that exact body
            self._send(415, {"error": {"message": "send application/json"}})
            return
        if self.path in ("/v1/load", "/v1/unload"):
            if eng.refuse_wake:
                self._send(503, {"error": {"message": eng.refusal_reason}})
            else:
                eng.status["loaded"] = self.path == "/v1/load"
                self._send(200, {"ok": True})
        elif self.path == "/v1/chat/completions":
            if eng.refuse_wake:
                self._send(503, {"error": {"message": eng.refusal_reason}})
            else:
                self._send(200, {"id": "fake", "object": "chat.completion",
                                 "choices": [{"index": 0, "message": {"role": "assistant",
                                                                      "content": "ok"},
                                              "finish_reason": "stop"}],
                                 "usage": {"completion_tokens": 1, "prompt_tokens": 1,
                                           "total_tokens": 2}})
        else:
            self._send(404, {"error": {"message": f"no such route {self.path}"}})


class _Server(ThreadingHTTPServer):
    """The handler reaches the engine through this typed attribute."""
    engine: "FakeEngine"


class FakeEngine:
    """A live HTTP engine with switchable behavior. Mutable fields (status, refuse_wake,
    refusal_reason) let a branch flip reality mid-test; seen[] is the truth of what arrived.
    """

    def __init__(self, *, name: str = "fake-engine", refuse_wake: bool = False,
                 refusal_reason: str = "card is full, refusing to load",
                 enforce_json: bool = True):
        self.name = name
        self.refuse_wake = refuse_wake
        self.refusal_reason = refusal_reason
        self.enforce_json = enforce_json
        self.seen: list[dict] = []
        self.status: dict = {"model_loaded": True, "in_flight": 0}
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.engine = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def last_post(self, path: str = "") -> dict | None:
        for rec in reversed(self.seen):
            if rec["method"] == "POST" and rec["path"].startswith(path):
                return rec
        return None


class WorkingEngine(FakeEngine):
    """Answers everything: wake loads, completions complete."""


class RefusingEngine(FakeEngine):
    """Every wake-shaped call (load, completions) is a 503 with one fixed, verbatim reason —
    the B-2 shape: refusal carries the engine's own words, and the bench asserts them
    character-for-character."""

    def __init__(self, **kw):
        kw.setdefault("refuse_wake", True)
        super().__init__(**kw)
