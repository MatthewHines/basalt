"""The ECM policy bench (W2-0). Run: python3 -m serve.ecm.bench.run_bench   (repo root).

Rules this driver enforces about ITSELF, inherited from the suite laws learned the hard
way this month:
- A branch with no implementation reports SKIP with its owning slice — never a silent
  PASS. Passing on emptiness is the failure mode this bench exists to avoid.
- If EVERY registered branch is SKIP, the run exits nonzero: an empty bench proves nothing.
- Log assertions are by id + event + decision, never substring (markers ending in hex
  cross-joined plain substrings once; that law applies here too).
- Each branch gets fresh fake reality; nothing reads a card, a GPU, or podman.

W2-0 proves the machinery every later branch rides: the HTTP client seam (415 law,
verbatim refusals, no faked status), the decision log (grammar, monotonic ids, honest
reader), and the calibrated fit-points. B-1..B-9 register as pending with owners.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .. import client, decision_log, heal, home, lifecycle, ping
from ..calibration import CALIBRATION_V1, demand_fits
from .fake_engine import FakeEngine, RefusingEngine, WorkingEngine

# ---------------------------------------------------------------- implemented branches


def seam_415_law() -> None:
    """A header-less POST gets the real 415 shape (the fake models the 11-27 engine);
    through the client seam that shape is unreachable, and the fake SAW the json type."""
    eng = FakeEngine()
    try:
        raw = urllib.request.Request(f"{eng.url}/v1/load", data=b"{}", method="POST")
        try:  # deliberately bypassing the seam: reality must show the old law
            urllib.request.urlopen(raw, timeout=5)
            raise AssertionError("header-less POST was not 415 — fake unfaithful")
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read())
            assert exc.code == 415, f"expected 415, got {exc.code}"
            assert body["error"]["message"] == "send application/json", body
        out = client.post_json(f"{eng.url}/v1/load")
        assert out == {"ok": True}, out
        post = eng.last_post("/v1/load")
        assert post and post["content_type"] == "application/json", post
    finally:
        eng.stop()


def seam_refusal_verbatim() -> None:
    """A 503 refusal arrives with the engine's words EXACTLY — character-for-character,
    not paraphrased, not wrapped. (The '(busy?)' paraphrase lived for weeks once.)"""
    reason = "card is full, refusing to load"
    eng = RefusingEngine(refusal_reason=reason)
    try:
        try:
            client.post_json(f"{eng.url}/v1/load")
            raise AssertionError("refusing engine accepted a load")
        except client.EngineError as exc:
            assert exc.status == 503, exc.status
            assert exc.reason == reason, f"paraphrased: {exc.reason!r} != {reason!r}"
    finally:
        eng.stop()


def seam_no_faked_status() -> None:
    """Unknown route -> EngineError(404); dead endpoint -> EngineError(0). Neither becomes
    an optimistic default anywhere inside the seam."""
    eng = WorkingEngine()
    try:
        try:
            client.get_json(f"{eng.url}/v1/nope")
            raise AssertionError("404 answered as success")
        except client.EngineError as exc:
            assert exc.status == 404, exc.status
    finally:
        eng.stop()
    try:
        client.get_json("http://127.0.0.1:1/v1/status", timeout=2.0)
        raise AssertionError("dead endpoint answered")
    except client.EngineError as exc:
        assert exc.status == 0, exc.status
        assert "no answer" in exc.reason, exc.reason


def log_grammar() -> None:
    """Validator runs BEFORE the write: a refusal without reason or remedy raises and the
    file keeps its old length; a well-formed refusal round-trips with reason verbatim."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "decisions.jsonl")
        assert decision_log.read(path) == [], "fresh log must read empty, not raise"
        try:
            decision_log.append(path, event="guest.wake.refused", decision="refuse")
            raise AssertionError("refusal without reason was accepted")
        except ValueError:
            pass
        assert decision_log.read(path) == [], "failed append must not write a line"
        try:
            decision_log.append(path, event="melt.down", decision="observe",
                                operator_hint="x", reason="r")
            raise AssertionError("event outside the family was accepted")
        except ValueError:
            pass
        e = decision_log.append(path, event="guest.wake.refused", decision="refuse",
                                reason="card is full, refusing to load",
                                operator_hint="bash swarm-ctl.sh heal-replan")
        back = decision_log.read(path)
        assert len(back) == 1 and back[0]["id"] == e["id"], back
        assert back[0]["reason"] == "card is full, refusing to load"
        # W2-4a grew the family with `window.*` (§6.1): in-family must validate,
        # out-of-family must still be refused (the enum keeps machine checks honest).
        decision_log.append(path, event="window.empty", decision="degraded",
                            reason="open with zero children", operator_hint="x")
        assert decision_log.read(path)[-1]["event"] == "window.empty"


def log_monotonic() -> None:
    """60 same-second appends: ids strictly increase, the reader accepts the file."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "decisions.jsonl")
        for i in range(60):
            decision_log.append(path, event="lease.set", lease={"asked_mib": i})
        entries = decision_log.read(path)
        assert len(entries) == 60
        ids = [e["id"] for e in entries]
        assert ids == sorted(set(ids), key=lambda x: (int(x.split("-")[1]),
                                                      int(x.split("-")[2], 16))), ids


def log_corrupt_raises() -> None:
    """The reader never silently skips: a garbage line raises with its number."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "decisions.jsonl")
        decision_log.append(path, event="guest.wake.ok")
        with open(path, "a") as fh:
            fh.write("not json at all\n")
        try:
            decision_log.read(path)
            raise AssertionError("corrupt line was skipped")
        except ValueError as exc:
            assert ":2" in str(exc), exc


def calibration_fit_points() -> None:
    """The v1 point: demand (25.2 GiB) EXCEEDS the lease that carries it (21,504 MiB) —
    the fixture itself proves plan != ceiling. Naive-sum reasoning must refuse."""
    assert CALIBRATION_V1.demand_mib > CALIBRATION_V1.lease_floor_mib, \
        "the point only matters because demand > lease"
    assert demand_fits(21_504) is True
    assert demand_fits(21_503) is False


def _status(eng: FakeEngine):
    """A primary_status reader that goes over the wire - the ping is tested through the
    same seam the live route uses, not against an in-process shortcut."""
    def read() -> dict:
        out = client.get_json(f"{eng.url}/v1/status")
        assert isinstance(out, dict), out
        return out
    return read


def ecm_ping_shape() -> None:
    """PC-1: healthy primary, no guest -> 200, ok:true, state reaped, full body shape."""
    eng = WorkingEngine()
    try:
        code, body = ping.build("fake-qwen", _status(eng))
        assert code == 200 and body["ok"] is True, body
        assert body["engine"] == "fake-qwen"
        assert body["state"] == "reaped" and body["guest"]["state"] == "reaped", body
        assert body["primary"]["loaded"] is True, body["primary"]
        assert body["degraded"] is None
        assert body["decision_log"]["latest_id"] is None
    finally:
        eng.stop()


def ecm_ping_readonly() -> None:
    """PC-2, machine-checked: the fake engine's own request log shows the ping issued
    GETs only - a POST anywhere is a contract violation, no audit tool needed."""
    eng = WorkingEngine()
    try:
        ping.build("fake-qwen", _status(eng))
        ping.build("fake-qwen", _status(eng))
        assert eng.seen, "ping derived nothing live - it read no status at all"
        posts = [r for r in eng.seen if r["method"] != "GET"]
        assert not posts, f"ping is not read-only: {posts}"
    finally:
        eng.stop()


def ecm_ping_no_cache() -> None:
    """PC-7: two pings, reality flipped between them - the second says the new truth."""
    eng = WorkingEngine()
    try:
        _, b1 = ping.build("fake-qwen", _status(eng))
        assert b1["primary"]["loaded"] is True
        eng.status["loaded"] = False
        _, b2 = ping.build("fake-qwen", _status(eng))
        assert b2["primary"]["loaded"] is False, "ping served a cached status"
    finally:
        eng.stop()


def ecm_ping_unreachable() -> None:
    """REF-4 / PC-5: primary dark -> 503, ok:false, state unknown, full refusal shape.
    Silence gets the REF-4 words; never ok:true, never an empty body."""
    eng = WorkingEngine()
    eng.stop()  # free the port: now nothing answers
    code, body = ping.build("fake-qwen", _status(eng))
    assert code == 503 and body["ok"] is False, body
    assert body["state"] == "unknown"
    assert ping.REF4_REASON in body["reason"], body["reason"]
    assert body["lease"] == {"asked_mib": None, "applied_mib": None}
    assert body["next"].startswith("curl"), body["next"]
    assert "decision_log" in body


def ecm_ping_starved() -> None:
    """REF-2 class / PC-4: starved guest -> 503 with the reason VERBATIM, the lease tuple
    from the guest reader, and the decision-log id of the refusal that caused it."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "decisions.jsonl")
        e = decision_log.append(path, event="guest.wake.refused", decision="refuse",
                                reason="experts cannot fit",
                                operator_hint="bash swarm-ctl.sh status")
        eng = WorkingEngine()
        try:
            def guest() -> dict:
                return {"state": "starved", "name": "inkling", "reason": "experts cannot fit",
                        "lease_asked_mib": 12800, "lease_applied_mib": 2048}
            code, body = ping.build("fake-qwen", _status(eng), guest=guest, log_path=path)
            assert code == 503 and body["ok"] is False
            assert body["state"] == "starved"
            assert body["reason"] == "experts cannot fit", body["reason"]  # verbatim
            assert body["lease"] == {"asked_mib": 12800, "applied_mib": 2048}, body["lease"]
            assert body["decision_log"]["latest_id"] == e["id"]
            assert body["next"] == ping.NEXT_FOR["starved"]
        finally:
            eng.stop()


def ecm_ping_speed() -> None:
    """The <1 s idle budget (PC-1), averaged over three loopback pings - generous on
    purpose; this catches an accidental sleep or a retry storm, not jitter."""
    eng = WorkingEngine()
    try:
        t0 = time.perf_counter()
        for _ in range(3):
            ping.build("fake-qwen", _status(eng))
        avg = (time.perf_counter() - t0) / 3
        assert avg < 1.0, f"{avg:.3f}s average - over the idle budget"
    finally:
        eng.stop()


def ecm_ping_log_view() -> None:
    """The drill view: capped, ordered tail of the log; count and latest id agree."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "decisions.jsonl")
        ids = [decision_log.append(path, event="lease.set", lease={"asked_mib": i})["id"]
               for i in range(25)]
        eng = WorkingEngine()
        try:
            view = ping.table("fake-qwen", _status(eng), log_path=path, cap=20)
            assert [e["id"] for e in view["decisions"]] == ids[-20:]
            assert view["ok"] is True
            _, body = ping.build("fake-qwen", _status(eng), log_path=path)
            assert body["decision_log"]["latest_id"] == ids[-1]
            assert body["decision_log"]["entries_24h"] == 25
        finally:
            eng.stop()


def heal_quiet_gate() -> None:
    """Silence must be PROVEN (fail-closed): in-flight, too-recent, or unreadable all
    answer 'not quiet'; >= QUIET seconds of silence answers quiet."""
    now = 1_000_000.0
    cases = [
        ({"activity": {"in_flight": 2, "last_request_at": now}}, False, "in flight"),
        ({"activity": {"in_flight": 0, "last_request_at": now - 5}}, False, "silence"),
        ({"activity": {"in_flight": 0, "last_request_at": now - heal.QUIET_SILENCE_SECONDS - 1}}, True, ""),
        ({}, False, "fail-closed"),
    ]
    for status, want, needle in cases:
        quiet, why = heal.quiet_check(status, now)
        assert quiet is want, f"{status} -> {quiet} ({why})"
        if needle:
            assert needle in why, why


def heal_verdict() -> None:
    """The predicate is a CEILING on idle free (healthy card runs full; a crowd-made
    plan leaves headroom): above ceiling -> required, at-or-below -> healthy; no reading
    is unreadable, never a guess; an unloaded primary has no plan to heal."""
    assert heal.r1_verdict(True, heal.HEADROOM_CEILING_MIB + 1)[0] == "required"
    assert heal.r1_verdict(True, heal.HEADROOM_CEILING_MIB)[0] == "healthy"
    assert heal.r1_verdict(True, 1_904)[0] == "healthy"  # the measured live steady state
    assert heal.r1_verdict(True, None)[0] == "unreadable"
    assert heal.r1_verdict(False, 1_000)[0] == "not-applicable"


def heal_plan_guards_first() -> None:
    """Policy beats fact: with a required verdict AND an open window, the answer is the
    guard's reason, not a replan."""
    now = 1_000_000.0
    status = {"loaded": True,
              "activity": {"in_flight": 0, "last_request_at": now - 400}}
    action, reason = heal.plan(status=status, free_mib=10_000,
                               guards=[lambda: "a window is OPEN (w1) - re-plan AFTER hand-back"],
                               now=now)
    assert action == "refused" and "OPEN" in reason, (action, reason)
    action2, _ = heal.plan(status=status, free_mib=10_000, guards=[], now=now)
    assert action2 == "replan"


def heal_replan_flow() -> None:
    """Required -> exactly one unload/load/wait cycle, logged required->replanned;
    immediately asking again on a healed card answers already with NO verbs."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "decisions.jsonl")
        calls: list[str] = []
        frees = iter([26_000, 2_000])  # crowd-made (headroom unplanned) now, card-full after the cycle

        def status_reader() -> dict:
            return {"loaded": True, "activity": {"in_flight": 0,
                                                 "last_request_at": time.time() - 400}}
        e1 = heal.replan(status_reader=status_reader, free_reader=lambda: next(frees),
                         unload=lambda: calls.append("unload"),
                         load=lambda: calls.append("load"),
                         wait_loaded=lambda: calls.append("wait"),
                         log_path=path)
        assert e1["event"] == "heal.required", e1
        assert calls == ["unload", "load", "wait"], calls
        e2 = heal.replan(status_reader=status_reader, free_reader=lambda: 2_000,
                         unload=lambda: calls.append("unload2"),
                         load=lambda: calls.append("load2"),
                         wait_loaded=lambda: None, log_path=path)
        assert e2["event"] == "heal.checked" and e2["decision"] == "already", e2
        assert calls == ["unload", "load", "wait"], f"second opinion moved verbs: {calls}"
        events = [e["event"] for e in decision_log.read(path)]
        assert events == ["heal.required", "heal.replanned", "heal.checked"], events


def heal_guard_refuses_without_verbs() -> None:
    """A guarded refusal touches NOTHING: the fake engine's request log stays POST-free
    and the ledger carries reason verbatim + remedy."""
    eng = WorkingEngine()
    try:
        eng.status["activity"]["last_request_at"] = time.time() - 400
        entry = heal.replan(
            status_reader=_status(eng),
            free_reader=lambda: 10_000,
            unload=lambda: eng.seen.append({"method": "BUG"}),
            load=lambda: eng.seen.append({"method": "BUG"}),
            wait_loaded=lambda: None,
            guards=[lambda: "a labeled guest container exists - refusing (hand its window back first)"])
        assert entry["event"] == "heal.refused" and entry["decision"] == "refuse", entry
        assert "guest container exists" in entry["reason"]
        assert entry["operator_hint"], "a refusal without a remedy is noise"
        assert not [r for r in eng.seen if r["method"] != "GET"], "guarded refusal ran verbs"
    finally:
        eng.stop()


def heal_unreadable_driver() -> None:
    """No driver truth = degraded refusal with the nvidia-smi remedy - never a silent
    pass, never a guess about the card."""
    entry = heal.replan(status_reader=lambda: {"loaded": True, "activity": {"in_flight": 0,
                                                                            "last_request_at": 0}},
                        free_reader=lambda: None,
                        unload=lambda: None, load=lambda: None, wait_loaded=lambda: None)
    assert entry["decision"] == "degraded", entry
    assert "nvidia-smi" in entry["operator_hint"], entry["operator_hint"]
    assert entry["persisted"] is False, "no-home entries must say so"


def home_resolution() -> None:
    """ECM_STATE_DIR wins; without it, no /out mount (host/bench side) means UNASSIGNED -
    honestly None, never a guessed writable path."""
    saved = dict(os.environ)
    try:
        os.environ["ECM_STATE_DIR"] = "/tmp/ecm-test-home"
        assert home.decision_log_path() == "/tmp/ecm-test-home/decisions.jsonl"
        del os.environ["ECM_STATE_DIR"]
        if not os.path.isdir("/out/window/inkling"):  # true on the host; container side is the drill's job
            assert home.decision_log_path() is None
    finally:
        os.environ.clear()
        os.environ.update(saved)


# ---------------------------------------------------------------- W2-4a watch predicates
#
# Fixtures are temp-dir window state + fake port truth - the same incident shapes the
# card showed on 11-28, reproducible without a GPU: an OPEN window holding the card
# with zero children, and a guest whose port answers while its model is unloaded.

WATCH_NOW = 1_000_000.0


def _write_window(root: str, name: str, *, returned: str = "0", abandoned: str = "0",
                  yielded_at: int | None = None, delegs: str | None = "") -> str:
    wdir = os.path.join(root, name)
    os.makedirs(wdir, exist_ok=True)
    state: dict = {"returned": returned, "abandoned": abandoned}
    if yielded_at is not None:
        state["yielded_at"] = str(yielded_at)
    if delegs is not None:
        state["delegs"] = delegs
    with open(os.path.join(wdir, "state.json"), "w") as fh:
        json.dump(state, fh)
    return wdir


_GUEST_LOADED = lambda: {"data": [{"id": "inkling", "status": {"value": "loaded"}}]}


def _guest_unloaded() -> object:
    return {"data": [{"id": "inkling", "status": {"value": "unloaded"}}]}


def _guest_dark() -> object:
    raise client.EngineError(0, "no answer (connection refused)",
                             "http://127.0.0.1:8004/v1/models")


def _watch(state_dir: str, path: str, *, loaded: bool = True, guest=_GUEST_LOADED,
           free: int | None = 1_904):
    return lifecycle.observe(
        status_reader=lambda: {"loaded": loaded}, models_reader=guest,
        free_reader=lambda: free, state_dir=state_dir, log_path=path,
        now=lambda: WATCH_NOW)


def watch_window_empty_fires_once() -> None:
    """The incident #1 shape: OPEN, zero children, 400 s past grace -> window.empty +
    the literal return --force line; a second read of the SAME episode logs NOTHING
    new (episode law, §1)."""
    lifecycle.reset_episodes()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        _write_window(d, "w1", yielded_at=int(WATCH_NOW) - 400, delegs="")
        obs = _watch(d, log)
        assert obs["degraded"] == "window.empty", obs
        alarm = obs["alarms"][0]
        assert "zero registered children" in alarm["reason"], alarm
        assert "return --window" in alarm["next"] and "/w1 --force" in alarm["next"], alarm
        ids_before = [e["id"] for e in decision_log.read(log)]
        assert len(ids_before) == 1 and ids_before[0] == obs["episode_ids"]["window.empty:w1"]
        _watch(d, log)  # same episode, second ping
        assert len(decision_log.read(log)) == 1, "episode re-logged while held"
        # and the entry is a REAL degraded family entry (grammar + verbatim reason)
        e = decision_log.read(log)[0]
        assert e["event"] == "window.empty" and e["decision"] == "degraded", e
        assert e["operator_hint"] == alarm["next"]


def watch_empty_grace_and_children() -> None:
    """No false alarms: inside grace the window is OPENING, not empty; a window WITH
    registered children is running, not empty - neither may alarm."""
    lifecycle.reset_episodes()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        _write_window(d, "young", yielded_at=int(WATCH_NOW) - 60, delegs="")
        obs = _watch(d, log)
        assert obs["alarms"] == [], obs
        _write_window(d, "busy", yielded_at=int(WATCH_NOW) - 900, delegs="deleg_1")
        obs2 = _watch(d, log)
        assert obs2["alarms"] == [], obs2
        assert set(obs2["holders"]["open_windows"]) == {"young", "busy"}
        assert decision_log.read(log) == [], "false alarms reached the ledger"


def watch_delegs_unreadable_not_quiet() -> None:
    """A state.json without a provable delegs field cannot CLAIM zero children: the
    alarm says unreadable (naming the field), never window.empty off a guess."""
    lifecycle.reset_episodes()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        _write_window(d, "opaque", yielded_at=int(WATCH_NOW) - 900, delegs=None)
        obs = _watch(d, log)
        assert obs["degraded"] == "window.unreadable", obs
        assert "delegs" in obs["alarms"][0]["reason"], obs
        assert all(a["name"] != "window.empty" for a in obs["alarms"])


def watch_starved_and_parking_lot() -> None:
    """§1 predicate, all three legs: primary loaded + window OPEN + port dark ->
    guest.starved. Same dark port with the primary UNLOADED is the intended PARKED
    posture - inside grace, zero alarms (the parking lot is not an incident)."""
    lifecycle.reset_episodes()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        _write_window(d, "w1", yielded_at=int(WATCH_NOW) - 60, delegs="d1")
        obs = _watch(d, log, loaded=True, guest=_guest_dark)
        assert obs["degraded"] == "guest.starved", obs
        reason = obs["alarms"][0]["reason"]
        for leg in ("window w1 OPEN", "primary loaded", "port dark"):
            assert leg in reason, f"leg missing: {leg} in {reason!r}"
        lifecycle.reset_episodes()
        obs2 = _watch(d, log, loaded=False, guest=_guest_dark)
        assert obs2["alarms"] == [], f"parked posture alarmed: {obs2}"


def watch_guest_dark_names_the_leg() -> None:
    """The incident #2 trap: port answers, model unloaded, window open -> guest.dark
    saying 'port-up is not model-up' out loud. REVERSED LAW (ecmw24f F5): with an EMPTY
    round both alarms fire and window.empty leads the degraded token — closing the
    window is the cure for the dark symptom; the dark alarm still NAMES itself."""
    lifecycle.reset_episodes()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        _write_window(d, "w1", yielded_at=int(WATCH_NOW) - 900, delegs="")
        obs = _watch(d, log, loaded=False, guest=_guest_unloaded)
        assert obs["degraded"] == "window.empty", obs
        names = [a["name"] for a in obs["alarms"]]
        assert "guest.dark" in names and "window.empty" in names, names
        dark = next(a for a in obs["alarms"] if a["name"] == "guest.dark")
        assert "port-up is not model-up" in dark["reason"], dark
        assert "unloaded" in dark["reason"], dark  # names its own reading
        # and with CHILDREN present (not empty), dark is the headline again
        lifecycle.reset_episodes()
        _write_window(d, "w1", yielded_at=int(WATCH_NOW) - 900, delegs="deleg_1")
        obs2 = _watch(d, log, loaded=False, guest=_guest_unloaded)
        assert obs2["degraded"] == "guest.dark", obs2


def watch_parked_no_window() -> None:
    """No window + dark guest port is HOME, not an incident: zero alarms, and the
    holders story still reads the truth (§6.3 - one read, no archaeology)."""
    lifecycle.reset_episodes()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        obs = _watch(d, log, loaded=True, guest=_guest_dark)
        assert obs["alarms"] == [] and obs["degraded"] is None, obs
        assert obs["holders"]["guest_port"] == "dark"
        assert obs["holders"]["primary_loaded"] is True
        assert obs["holders"]["open_window"] is None
        assert obs["holders"]["driver_free_mib"] == 1_904  # the measured healthy card


def watch_holders_fail_closed() -> None:
    """No driver truth is a NULL in the story, never a zero and never a guess."""
    lifecycle.reset_episodes()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        obs = _watch(d, log, free=None)
        assert obs["holders"]["driver_free_mib"] is None, obs


def watch_episode_refires_after_resolution() -> None:
    """Episode law's other half: resolved -> memory clears -> a RE-CURRENCE earns a
    fresh entry (an honest duplicate beats lost truth)."""
    lifecycle.reset_episodes()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        _write_window(d, "w1", yielded_at=int(WATCH_NOW) - 900, delegs="")
        assert _watch(d, log)["degraded"] == "window.empty"
        _write_window(d, "w1", yielded_at=int(WATCH_NOW) - 900, delegs="deleg_1")
        obs2 = _watch(d, log)
        assert obs2["alarms"] == [] and obs2["episode_ids"] == {}, obs2
        _write_window(d, "w1", yielded_at=int(WATCH_NOW) - 900, delegs="")
        assert _watch(d, log)["degraded"] == "window.empty"
        events = [(e["event"], e["id"]) for e in decision_log.read(log)]
        assert len(events) == 2 and events[0][0] == events[1][0] == "window.empty", events
        assert events[0][1] != events[1][1], "re-fire reused the old id"


def watch_numeric_returned_reads_open() -> None:
    """REVIEW F1 fixture: a hand-edited state.json with numeric returned:0 (and
    abandoned:0) must read OPEN — same alarm, same literal — never silently closed."""
    lifecycle.reset_episodes()
    # REVIEW F2 beside its own law: the known bind translates to the HOST path so the
    # printed next-command is paste-ready; an unknown state_dir passes through untouched.
    assert lifecycle._host_path("/out/window/inkling/wX", lifecycle.STATE_DIR) == \
        os.path.join(lifecycle.HOST_STATE_DIR, "wX")
    assert lifecycle._host_path("/tmp/other/wX", "/tmp/other") == "/tmp/other/wX"
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        wd = os.path.join(d, "numwin")
        os.makedirs(wd)
        with open(os.path.join(wd, "state.json"), "w", encoding="utf-8") as fh:
            json.dump({"returned": 0, "abandoned": 0,
                       "yielded_at": str(int(WATCH_NOW) - 400), "delegs": ""}, fh)
        obs = _watch(d, log)
        assert obs["degraded"] == "window.empty", obs
        assert "numwin" in obs["holders"]["open_windows"], obs


def watch_wrong_engine_list_absent() -> None:
    """REVIEW F3 fixture: a port that answers with SOMEONE ELSE'S model list (no id
    matching GUEST_MODEL_ID) reports model 'absent' naming what it saw — and inside an
    OPEN window that is guest.dark, never a borrowed status from the wrong engine."""
    lifecycle.reset_episodes()
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "decisions.jsonl")
        _write_window(d, "w1", yielded_at=int(WATCH_NOW) - 60, delegs="d1")
        obs = _watch(d, log, loaded=False,
                     guest=lambda: {"data": [{"id": "qwen3.8-flash-next-basalt",
                                              "status": {"value": "loaded"}}]})
        assert obs["holders"]["guest_model"] == "absent", obs
        assert obs["degraded"] == "guest.dark", obs
        assert "no 'inkling' in the list" in obs["alarms"][0]["reason"], obs


def log_flock_concurrent_appends() -> None:
    """REVIEW F6 fixture: 8 threads appending at once — the file must READ back clean
    (monotonic ids, 8 entries). Before flock this raced into duplicate ids."""
    import threading as _th
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "decisions.jsonl")
        threads = [_th.Thread(target=lambda i=i: decision_log.append(
            path, event="heal.checked", decision="observe", reason=f"t{i}",
            operator_hint="none")) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        entries = decision_log.read(path)     # raises on any duplicate/interleave
        assert len(entries) == 8, len(entries)
        ids = [e["id"] for e in entries]
        assert len(set(ids)) == 8, f"duplicate ids: {ids}"


def ping_carries_watch() -> None:
    """Armed: holders/alarms/degraded ride the ping body. Unwired: the body SAYS it was
    not read - an unwired watch is never an all-clear."""
    lifecycle.reset_episodes()
    eng = WorkingEngine()
    try:
        code0, b0 = ping.build("fake-qwen", _status(eng))
        assert b0["watch"].startswith("not wired") and b0["holders"] is None, b0
        obs = {"holders": {"driver_free_mib": 1904, "open_window": "w1"},
               "alarms": [{"name": "window.empty", "reason": "r", "next": "n"}],
               "degraded": "window.empty"}
        code, body = ping.build("fake-qwen", _status(eng), watch=obs)
        assert code == 200 and body["watch"] == "armed", body
        assert body["degraded"] == "window.empty"
        assert body["holders"]["open_window"] == "w1"
        assert len(body["alarms"]) == 1
    finally:
        eng.stop()


def registry_honest() -> None:
    """Every pending branch names an owning slice, and the runner's own data shows no
    pending branch can report PASS (fn is None iff owner is set)."""
    for br in BRANCHES:
        if br.fn is None:
            assert br.owner, f"{br.name} is pending with no owning slice"
    implemented = [b for b in BRANCHES if b.fn is not None]
    assert implemented, "bench has zero implemented branches"


# ---------------------------------------------------------------- pending branches (B-1..B-9)

@dataclass(frozen=True)
class Branch:
    name: str
    fn: Callable[[], None] | None      # None => registered, not yet implemented
    owner: str | None = None           # slice that lands it


BRANCHES: list[Branch] = [
    Branch("seam.415-law", seam_415_law),
    Branch("seam.refusal-verbatim", seam_refusal_verbatim),
    Branch("seam.no-faked-status", seam_no_faked_status),
    Branch("log.grammar", log_grammar),
    Branch("log.monotonic", log_monotonic),
    Branch("log.corrupt-raises", log_corrupt_raises),
    Branch("calibration.fit-points", calibration_fit_points),
    Branch("ecm-ping.shape-pc1", ecm_ping_shape),
    Branch("ecm-ping.readonly-pc2", ecm_ping_readonly),
    Branch("ecm-ping.no-cache-pc7", ecm_ping_no_cache),
    Branch("ecm-ping.unreachable-ref4", ecm_ping_unreachable),
    Branch("ecm-ping.starved-ref2", ecm_ping_starved),
    Branch("ecm-ping.speed-budget", ecm_ping_speed),
    Branch("ecm-ping.drill-view", ecm_ping_log_view),
    Branch("heal.quiet-gate", heal_quiet_gate),
    Branch("heal.r1-verdict", heal_verdict),
    Branch("heal.plan-guards-first", heal_plan_guards_first),
    Branch("heal.replan-flow", heal_replan_flow),
    Branch("heal.guard-no-verbs", heal_guard_refuses_without_verbs),
    Branch("heal.unreadable-driver", heal_unreadable_driver),
    Branch("home.resolution", home_resolution),
    Branch("registry.honest", registry_honest),
    # W2-4a lifecycle OBSERVATION (ECM-W2-4-DESIGN §1/§6 - the 11-28 incidents):
    Branch("watch.window-empty-fires-once", watch_window_empty_fires_once),
    Branch("watch.empty-grace-and-children", watch_empty_grace_and_children),
    Branch("watch.delegs-unreadable-not-quiet", watch_delegs_unreadable_not_quiet),
    Branch("watch.starved-and-parking-lot", watch_starved_and_parking_lot),
    Branch("watch.guest-dark-names-the-leg", watch_guest_dark_names_the_leg),
    Branch("watch.parked-no-window", watch_parked_no_window),
    Branch("watch.holders-fail-closed", watch_holders_fail_closed),
    Branch("watch.episode-refires", watch_episode_refires_after_resolution),
    Branch("watch.numeric-returned-reads-open", watch_numeric_returned_reads_open),
    Branch("watch.wrong-engine-list-absent", watch_wrong_engine_list_absent),
    Branch("log.flock-concurrent-appends", log_flock_concurrent_appends),
    Branch("ecm-ping.carries-watch", ping_carries_watch),
    # BENCH-BRANCHES.md minimum list, owned per ECM-W2-DESIGN §7 slicing:
    Branch("B-1.wake-happy-path", None, "W2-4"),
    Branch("B-2.refusal-rollback", None, "W2-3"),
    Branch("B-3.self-unload-unrequested", None, "W2-4"),
    Branch("B-4.container-vanished", None, "W2-4"),
    Branch("B-5.primary-starvation-r1", None, "W2-3"),
    Branch("B-6.socket-degrades-manual", None, "W2-4"),
    Branch("B-7.mode-switch-persisted", None, "W2-5"),
    Branch("B-8.mode-rehydration", None, "W2-5"),
    Branch("B-9.handback-checklist-failure", None, "W2-4"),
]


def main() -> int:
    passed = failed = pending = 0
    for br in BRANCHES:
        if br.fn is None:
            pending += 1
            print(f"SKIP {br.name} (owner {br.owner}, awaiting implementation)")
            continue
        try:
            br.fn()
            passed += 1
            print(f"PASS {br.name}")
        except Exception as exc:  # noqa: BLE001 - a branch failing must not stop the run
            failed += 1
            print(f"FAIL {br.name}: {type(exc).__name__}: {exc}")
    print(f"bench: {passed} PASS / {failed} FAIL / {pending} pending-registered")
    if failed:
        return 1
    if passed == 0:
        print("bench: NOTHING RAN — an empty bench proves nothing")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
