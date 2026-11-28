"""The ping: one read that answers "is the hypervisor healthy?" (PING-CONTRACT.md).

Strictly read-only by construction: it calls READERS and never a verb — no load, no wake,
no POST, no sleep, and every field is re-derived at call time (no cache, no script memory).
Fail-closed law: a state the primary cannot confirm is never printed — an unreadable
primary is the REF-4 503 shape, never an optimistic default. A refusal is a first-class
answer: reason verbatim + decision-log id + lease tuple + ONE literal, executable remedy
for exactly the state just read (PC-11).

Readers are injected callables, so the same builder runs in-process (live) and against
fake engines (bench) — the HTTP shape asserted here is the shape the route serves.

Guest knowledge arrives with the lifecycle slice (W2-4): until then the guest reader
returns None and the truth is reported as `reaped` — no guest exists, the card is home.
The decision-log path likewise stays None until the ECM owns a writable home for it
(that is a real decision, not a repeat of the /tmp modefile mistake).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable

from . import client, decision_log

GUEST_STATES = frozenset(("standby", "loaded", "starved", "reaped", "unknown"))
# literal remedies per failure shape; PC-11: only what is executable for the state just read
NEXT_FOR = {
    "unknown": "curl -sS -m 5 http://127.0.0.1:8003/v1/status",
    "starved": "bash /var/home/mhines/basalt-test/window/inkling/swarm-ctl.sh status",
}
REF4_REASON = "primary unreachable — refuse, never assume free"
GUEST_UNREACHABLE_REASON = "engine unreachable — fail-closed"


def _primary_view(status: dict) -> dict:
    activity = status.get("activity") or {}
    vram = status.get("vram") or {}
    return {"loaded": status.get("loaded"), "loading": status.get("loading"),
            "in_flight": activity.get("in_flight"), "requests": activity.get("requests"),
            "vram_free_mib": vram.get("free_mib", vram.get("free"))}


def _log_view(path: str | None) -> dict:
    if not path:
        return {"latest_id": None, "entries_24h": None, "home": "unassigned"}
    entries = decision_log.read(path)
    day_ago = datetime.now(timezone.utc) - timedelta(hours=24)
    recent = 0
    for e in entries:
        try:
            if datetime.fromisoformat(e["ts"]) >= day_ago:
                recent += 1
        except (KeyError, ValueError, TypeError):
            recent += 1  # an entry we cannot date counts as recent: never under-report
    return {"latest_id": entries[-1]["id"] if entries else None, "entries_24h": recent}


def build(engine: str, primary_status: Callable[[], dict],
          guest: Callable[[], dict | None] | None = None,
          log_path: str | None = None, *, now: Callable[[], float] = time.time,
          ) -> tuple[int, dict]:
    """Returns (http_status, body). 200 only when the primary is readable and the guest
    state is a confirmed one; every refusal carries the full shape (PC-4)."""
    log = _log_view(log_path)
    body: dict = {"ok": False, "engine": engine, "checked_at": int(now()),
                  "primary": None, "guest": None, "decision_log": log, "degraded": None}

    try:
        status = primary_status()
        if not isinstance(status, dict):
            raise ValueError(f"status reader returned {type(status).__name__}, not an object")
    except client.EngineError as exc:
        # an answered HTTP error is the engine's own words; silence is REF-4
        reason = exc.reason if exc.status else REF4_REASON
        body.update({"state": "unknown", **_refusal(reason)})
        body["next"] = NEXT_FOR["unknown"]
        return 503, body
    except Exception as exc:  # noqa: BLE001 - fail-closed: any read failure refuses
        body.update({"state": "unknown", **_refusal(f"{REF4_REASON} ({exc})")})
        body["next"] = NEXT_FOR["unknown"]
        return 503, body

    body["primary"] = _primary_view(status)

    try:
        guest_state = (guest() if guest else None)
    except Exception as exc:  # noqa: BLE001 - the guest reader failing is a guest truth
        guest_state = {"state": "unknown", "reason": f"guest state unreadable: {exc}"}
    if guest_state is None:
        guest_state = {"state": "reaped", "name": None, "mode": None,
                       "lease_asked_mib": None, "lease_applied_mib": None,
                       "note": "no guest reader wired: lifecycle lands in W2-4"}
    if guest_state.get("state") not in GUEST_STATES:
        guest_state = {"state": "unknown",
                       "reason": f"guest state {guest_state.get('state')!r} outside the "
                                 f"contract; {GUEST_UNREACHABLE_REASON}"}
    body["guest"] = guest_state

    state = guest_state["state"]
    body["state"] = state  # PING-CONTRACT §2: state is top-level on 200s and 503s alike
    if state == "unknown":
        reason = guest_state.get("reason") or GUEST_UNREACHABLE_REASON
        body.update(_refusal(reason))
        body["next"] = NEXT_FOR["unknown"]
        return 503, body
    if state == "starved":
        reason = guest_state.get("reason")
        if not reason:
            reason = GUEST_UNREACHABLE_REASON  # a starved with no reason is unreadable honesty
        body.update(_refusal(reason))
        held = {"asked_mib": guest_state.get("lease_asked_mib"),
                "applied_mib": guest_state.get("lease_applied_mib")}
        body["lease"] = held
        body["next"] = NEXT_FOR["starved"]
        return 503, body

    body["ok"] = True
    return 200, body


def table(engine: str, primary_status: Callable[[], dict],
          guest: Callable[[], dict | None] | None = None,
          log_path: str | None = None, *, cap: int = 20,
          ) -> dict:
    """The drill view (GET /v1/ecm): the same live reads as the ping, plus the last
    `cap` decision-log entries, oldest first. Read-only, like the ping - a capped tail,
    never the whole file on the wire."""
    code, body = build(engine, primary_status, guest, log_path)
    entries: list[dict] = []
    if log_path:
        entries = decision_log.read(log_path)[-cap:]
    return {"ok": body["ok"], "http": code, "engine": engine,
            "state": body.get("state"), "primary": body.get("primary"),
            "guest": body.get("guest"), "reason": body.get("reason"),
            "decisions": entries}


def _refusal(reason: str) -> dict:
    """The refusal shape (PC-4): reason verbatim + lease tuple; the caller adds the
    decision-log id (already in the body) and the literal remedy. Never an empty body."""
    return {"reason": reason, "lease": {"asked_mib": None, "applied_mib": None}}
