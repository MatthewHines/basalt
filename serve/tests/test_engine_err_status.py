"""The engine's ERR reaches the client by cause: 400 when the request caused it, 500 when the engine failed;
a GPU held elsewhere is a 503 that also says when to come back (Retry-After)."""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.engine.errors import EngineError, GpuBusy  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.engine.process import request_err  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CTX = 4096
FAILED = "verify: a device wait timed out (mid chain: kind 2, line 69, block 3); the window is discarded; restart the engine"
REFUSED = "prompt (5000 tokens) + max_new (20) exceeds the context (4096)"
BODIES = [("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}),
          ("/v1/messages", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}),
          ("/v1/responses", {"input": "hi", "max_output_tokens": 20})]


class ErrEngine(MockEngine):
    err = None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        yield None
        raise request_err(self.err)


class GrammarEndEngine(MockEngine):
    """A reply the engine ends itself: tokens, then a DONE whose finish is "error" (the grammar allowed no token)."""
    last = None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, session=None):
        yield from self.tok.encode("{\"a\":")
        self.last = {"generated": 3, "prompt_tokens": len(ids), "prompt_ms": 1.0, "decode_ms": 1.0, "finish": "error"}


class BusyService(Service):
    """The card belongs to another program: loading refuses before anything starts."""

    def ensure_loaded(self):
        raise GpuBusy("the GPU is in use by another program: 10 MiB of VRAM free, the model needs "
                      "4096 (min_free_vram_mib) - it stays unloaded until that is free")


class EngineErrStatus(unittest.TestCase):
    def test_request_err_classifies_the_text(self):
        self.assertIsInstance(request_err(FAILED), EngineError)
        self.assertIsInstance(request_err("verify: launch: an illegal instruction was encountered"), EngineError)
        for text in (REFUSED, "grammar: bad rule", "logit_bias: id out of range", "bad request: max_new"):
            e = request_err(text)
            self.assertIs(type(e), ValueError, text)

    def statuses(self, err):
        tok = ByteTokenizer()
        eng = ErrEngine(tok, "xy", max_context=CTX)
        eng.err = err
        return self.serve_statuses(tok, eng)

    def serve_statuses(self, tok, eng):
        svc = Service(eng, tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        out = []
        try:
            for path, body in BODIES:
                req = urllib.request.Request(base + path, data=json.dumps({"model": "basalt", **body}).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        out.append((path, r.status, r.read().decode()))
                except urllib.error.HTTPError as e:
                    out.append((path, e.code, e.read().decode()))
        finally:
            httpd.shutdown()
            httpd.server_close()
        return out

    def test_an_engine_failure_is_a_500(self):
        for path, code, text in self.statuses(FAILED):
            self.assertEqual(code, 500, (path, text))
            self.assertIn("device wait timed out", text, path)
            self.assertIn("server_error", text, path)

    def test_a_reply_the_engine_ends_as_an_error_is_a_500(self):
        tok = ByteTokenizer()
        for path, code, text in self.serve_statuses(tok, GrammarEndEngine(tok, "xy", max_context=CTX)):
            self.assertEqual(code, 500, (path, text))
            self.assertIn("grammar allowed no token", text, path)

    def test_a_request_error_is_a_400(self):
        for path, code, text in self.statuses(REFUSED):
            self.assertEqual(code, 400, (path, text))
            self.assertIn("exceeds the context", text, path)

    def test_a_busy_gpu_is_a_503_that_says_when_to_come_back(self):
        tok = ByteTokenizer()
        svc = BusyService(MockEngine(tok, "xy", max_context=CTX), tok,
                          ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            for path, body in BODIES:
                req = urllib.request.Request(base + path, data=json.dumps({"model": "basalt", **body}).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=30) as r:
                        self.fail((path, r.status, "expected a 503"))
                except urllib.error.HTTPError as e:
                    self.assertEqual(e.code, 503, (path, e.read().decode()))
                    self.assertEqual(e.headers.get("Retry-After"), "30", path)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
