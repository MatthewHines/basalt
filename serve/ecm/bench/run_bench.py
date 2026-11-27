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
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .. import client, decision_log
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
    Branch("registry.honest", registry_honest),
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
