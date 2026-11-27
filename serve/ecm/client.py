"""The ECM's single HTTP seam. Every engine request goes through this file, so three laws
live in exactly one place: a POST always sends application/json with a body (a header-less
POST is refused 415 "send application/json" — this seam makes that shape unreachable);
a refusal's reason is surfaced VERBATIM, never paraphrased; and a status is never faked —
no answer is an error, not an optimistic default.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class EngineError(Exception):
    """A failed engine call. status is the HTTP code (0 when nothing answered at all) and
    reason is the engine's own words. Both ride into the decision log untouched."""

    def __init__(self, status: int, reason: str, url: str):
        super().__init__(f"{url} -> {status}: {reason}")
        self.status = status
        self.reason = reason
        self.url = url


def _reason_of(body: bytes, fallback: str) -> str:
    """The refusal reason exactly as the engine wrote it. A JSON error body contributes its
    error.message; anything else contributes its own first 200 characters."""
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return body.decode("utf-8", "replace")[:200] or fallback
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(err, str):
            return err
    return fallback


def request_json(url: str, payload: dict | None = None, method: str | None = None,
                 timeout: float = 5.0) -> object:
    """The choke point. With a payload the request is POST (or an explicit method) and the
    JSON content type is ALWAYS attached — no caller can forget it. Without one it is a GET.
    Non-2xx raises EngineError carrying the engine's verbatim reason; a dead endpoint raises
    EngineError(0, ...) — silence is reported as silence."""
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raise EngineError(exc.code, _reason_of(exc.read(), str(exc.reason)), url) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", None) or exc
        raise EngineError(0, f"no answer ({reason})", url) from None
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise EngineError(200, f"answered with non-JSON body: {raw[:120]!r}", url) from exc


def get_json(url: str, timeout: float = 5.0) -> object:
    """A read. Never fakes: any non-2xx or non-JSON answer raises EngineError."""
    return request_json(url, timeout=timeout)


def post_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> object:
    """A write. An empty body is sent as {} so the request is still a JSON POST — the 415
    law is about the content type AND a body, and both are this function's problem, not the
    caller's."""
    return request_json(url, payload={} if payload is None else payload, method="POST",
                        timeout=timeout)
