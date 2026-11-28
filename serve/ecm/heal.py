"""ONE heal policy for the primary's card (ECM-W2-DESIGN §4). The trigger table lives in
the plan() decision; every call answers with a decision-log entry, so heal is auditable
without archaeology.

Absorbed (guard list proven by swarm-ctl heal-replan, rows 26/28 - absorbed, not
duplicated): a live window refuses, a labeled guest container refuses, SILENCE IS REQUIRED
before any reload, and plan facts are never guessed. The R1 anchor here is driver truth at
the quiet gate against a CONFIG-derived floor - never a status key the installed build
happens to keep (0.5.7 dropped expert_cache_mib; the drill confirmed no free-VRAM number
either, so free comes from the driver, not the engine).

The decision is a pure function (plan); verbs only run inside replan(). Doctrine:
NO surprise loads - heal is operator-visible or gated, never fires mid-traffic, never
fires while a guest is starting (a starting guest is guard territory, W2-4).
"""
from __future__ import annotations

import time
from typing import Callable, Sequence

from . import client, decision_log

# R1's observable is UNPLANNED HEADROOM at the quiet gate, measured by driver truth.
# Measured live 11-27: a healthy qwen steady state runs the card FULL (nvidia-smi free
# 1,904 MiB inside the loaded container) - a crowd-MADE plan is the one that leaves
# GiB unplanned. So the predicate is a CEILING on idle free, not a floor: free above it
# means capacity went unreserved. (The 11-22 R1 wording, re-read: "plan BELOW the floor"
# spoke of the CACHE plan - the inverse of free. My first pass inverted it; the container
# fact caught it.) Calibrate this number against measured boots in W3's p3-grid work.
HEADROOM_CEILING_MIB = 6_144  # steady state observed ~1.9 GiB; 3x + slack, calibrate in drills
# Absorbed from heal-replan's PROVEN guard (60 s of request silence). §4 drafted 90 as an
# aspiration; measured discipline wins, and a live drill may still raise this, not lower it.
QUIET_SILENCE_SECONDS = 60


def quiet_check(status: dict, now: float) -> tuple[bool, str]:
    """Fail-closed: a status that cannot PROVE silence is not silent."""
    activity = status.get("activity") or {}
    inflight = activity.get("in_flight")
    if inflight is None:
        return False, "status has no activity.in_flight - fail-closed as not quiet"
    if inflight:
        return False, f"{inflight} request(s) in flight"
    last = activity.get("last_request_at")
    if last is None:
        return True, "no request has ever landed"
    gap = now - last
    if gap >= QUIET_SILENCE_SECONDS:
        return True, f"{int(gap)} s of request silence"
    return False, f"only {int(gap)} s of silence (needs {QUIET_SILENCE_SECONDS})"


def r1_verdict(loaded: bool | None, free_mib: int | None,
               ceiling_mib: int = HEADROOM_CEILING_MIB) -> tuple[str, str]:
    if not loaded:
        return "not-applicable", "primary not loaded: there is no card plan to heal"
    if free_mib is None:
        return "unreadable", "no driver-truth free-VRAM reading - refusing to guess (fail-closed)"
    if free_mib > ceiling_mib:
        return "required", (f"{free_mib} MiB free at the quiet gate - the plan left "
                            f"{free_mib - ceiling_mib}+ MiB unplanned (crowd-made?)")
    return "healthy", f"{free_mib} MiB free at the quiet gate - the card plan runs full"


def plan(*, status: dict, free_mib: int | None, guards: list[Callable[[], str | None]],
         now: float) -> tuple[str, str]:
    """The whole decision, pure: guards first (policy), then silence (timing), then the
    verdict (fact). Returns (action, reason); action is refused|replan|healthy|
    not-applicable|unreadable."""
    for guard in guards:
        reason = guard()
        if reason:
            return "refused", reason
    verdict, detail = r1_verdict(bool(status.get("loaded")), free_mib)
    if verdict != "required":
        return verdict, detail
    quiet, why = quiet_check(status, now)
    if not quiet:
        return "refused", f"re-plan needs a quiet card: {why}"
    return "replan", detail


VERB_TIMEOUT = 10.0


def replan(*, status_reader: Callable[[], dict], free_reader: Callable[[], int | None],
           unload: Callable[[], None], load: Callable[[], None],
           wait_loaded: Callable[[], None], log_path: str | None = None,
           guards: Sequence[Callable[[], str | None]] = (),
           spawn: Callable[[Callable[[], None]], None] | None = None,
           source: str = "ecm") -> dict:
    """Decide, and when told to replan: run exactly one unload+load cycle. With spawn,
    the work runs detached (the route answers 202 and the log carries completion); with
    None (bench, drills) it runs inline. Status that cannot be read refuses verbatim -
    heal never assumes the card is fine because a read failed."""
    now = time.time()
    try:
        status = status_reader()
        if not isinstance(status, dict):
            raise ValueError(f"status reader returned {type(status).__name__}, not an object")
        free = free_reader()
    except client.EngineError as exc:
        reason = exc.reason if exc.status else f"engine unreadable: {exc.reason}"
        return _log(log_path, "heal.refused", "refuse", reason,
                    "curl -sS -m 5 http://127.0.0.1:8003/v1/status", source=source)
    except Exception as exc:  # noqa: BLE001 - fail-closed
        return _log(log_path, "heal.refused", "refuse", f"heal inputs unreadable: {exc}",
                    "curl -sS -m 5 http://127.0.0.1:8003/v1/status", source=source)

    action, reason = plan(status=status, free_mib=free, guards=list(guards), now=now)
    if action == "refused":
        return _log(log_path, "heal.refused", "refuse", reason,
                    "bash /var/home/mhines/basalt-test/window/inkling/swarm-ctl.sh status",
                    source=source)
    if action in ("healthy", "not-applicable"):
        return _log(log_path, "heal.checked", "already", reason, source=source)
    if action == "unreadable":
        return _log(log_path, "heal.refused", "degraded", reason,
                    "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
                    source=source)

    required = _log(log_path, "heal.required", "allow", reason, source=source)

    def work() -> None:
        try:
            unload()
            load()
            wait_loaded()
        except Exception as exc:  # noqa: BLE001 - one failed cycle is a ledger fact, not a crash
            _log(log_path, "heal.failed", "rollback", f"replan cycle failed: {exc}",
                 "the primary may be unloaded or mid-load - check /v1/status, then POST /v1/load",
                 source=source)
            return
        _log(log_path, "heal.replanned", "allow",
             f"one unload+load cycle done at the quiet gate (was: {reason})", source=source)

    if spawn is None:
        work()
    else:
        spawn(work)
    return required


def _log(path: str | None, event: str, decision: str, reason: str,
         hint: str | None = None, *, source: str) -> dict:
    """Persist the decision when a home exists; without one, return an EPHEMERAL entry -
    validated, honestly flagged persisted:false, never quietly dropped."""
    if path:
        return decision_log.append(path, event=event, decision=decision, source=source,
                                   reason=reason, operator_hint=hint)
    from datetime import datetime, timezone
    entry = {"id": decision_log.new_id(None), "ts": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "event": event, "decision": decision, "source": source,
        "reason": reason, "operator_hint": hint, "persisted": False}
    decision_log.validate(entry)
    return entry
