"""W2-4a: the ECM owns lifecycle OBSERVATION. The predicates here are the alarms that
were missing during the 11-28 incidents (ECM-W2-4-DESIGN §6): an OPEN window with zero
children holding the card, and a guest whose PORT is up while its MODEL is not.

Pure reads + injected readers (bench pattern, same as heal/ping): nothing here loads,
unloads, boots, or touches podman — swarm-ctl keeps the verbs. Two laws bind every
predicate: unreadable is NOT quiet (a field we cannot read never reads as healthy, and
never reads as empty either), and every alarm carries the literal next command (PC-11).

Episode law (§1): an entry logs ONCE per episode. Episode memory is in-process; after an
ECM restart an ongoing condition re-enters fresh — an honest duplicate beats lost truth.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib
from typing import Callable

from . import client, decision_log

STATE_DIR = "/out/window/inkling"          # the window bind's CONTAINER side
GUEST_MODELS_URL = "http://127.0.0.1:8004/v1/models"
# REVIEW F3: whose model list counts as the guest. Overridable for benches/alt models.
GUEST_MODEL_ID = os.environ.get("ECM_GUEST_MODEL_ID", "inkling")
SWARMCTL = "/var/home/mhines/basalt-test/window/inkling/swarm-ctl.sh"
HOST_STATE_DIR = "/var/home/mhines/basalt-test/window/inkling"   # the bind's HOST side
# The card deserves a shorter answer than the defer flag's 4 h TTL (§6.1): five minutes
# of an OPEN window with zero children is an alarm, not a wait.
EMPTY_WINDOW_S = 300
# degraded = ONE token on the ping. REVIEW F5 (ecmw24f): window.empty LEADS — the one
# action that frees a card held by a stale round is closing the window; dark/starved are
# then symptoms of exactly that round, and must not outrank their own cause.
ALARM_PRIORITY = ("window.empty", "guest.starved", "guest.dark", "window.unreadable")

_episodes: dict[str, str] = {}             # episode key -> first entry id (in-process)
_lock = threading.Lock()


def _host_path(window_dir: str, state_dir: str) -> str:
    """REVIEW F2: a paste-ready command must name HOST paths. The same directory reads
    as /out/window/inkling/<name> from inside the container; the operator's shell (and
    host-side swarm-ctl) cannot use that. Translate only the known bind."""
    if state_dir == STATE_DIR and window_dir.startswith(state_dir):
        return HOST_STATE_DIR + window_dir[len(state_dir):]
    return window_dir


def window_facts(state_dir: str = STATE_DIR) -> tuple[list[dict], list[str]]:
    """OPEN windows (swarm-ctl's C5/D4 law: returned=="0" AND abandoned!="1") plus
    problems — unreadable state named, never swallowed. A missing dir is genuinely no
    windows; a present-but-garbled state.json is a problem, not an absence."""
    opens: list[dict] = []
    problems: list[str] = []
    try:
        names = sorted(os.listdir(state_dir))
    except FileNotFoundError:
        return [], []
    except OSError as exc:
        return [], [f"window state unreadable at {state_dir} ({exc})"]
    for name in names:
        sj = os.path.join(state_dir, name, "state.json")
        if not os.path.isfile(sj):
            continue
        try:
            with open(sj, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError) as exc:
            problems.append(f"window {name}: state.json unreadable ({exc})")
            continue
        if not isinstance(d, dict):
            problems.append(f"window {name}: state.json is not an object")
            continue
        # REVIEW F1 (ecmw24f): swarm-ctl writes strings, but a hand-edited state.json can
        # carry a JSON NUMBER 0 — which must still read OPEN. Coerce; never let a type
        # difference silently close a card-holding window.
        returned = str(d.get("returned", "0")).strip()
        abandoned = str(d.get("abandoned", "0")).strip()
        if returned in ("0", "0.0") and abandoned != "1":
            try:
                yielded = int(d.get("yielded_at", ""))
            except (TypeError, ValueError):
                yielded = None
            delegs = d.get("delegs")
            if delegs is not None and not isinstance(delegs, str):
                delegs = None  # wrong-typed registry: unprovable, never guessed empty
            opens.append({"name": name, "yielded_at": yielded, "delegs": delegs})
    return opens, problems


def guest_truth(models_reader: Callable[[], object]) -> dict:
    """Port truth for the guest, honest about what a port CAN say (B-4 law): a GET that
    fails says 'dark' — absent and unloaded are indistinguishable from the wire and the
    detail says so. An answered list says what the model itself reports, or that the
    shape was unrecognizable — never a guessed 'loaded'."""
    try:
        data = models_reader()
    except client.EngineError as exc:
        return {"port": "dark", "model": None,
                "detail": f"no usable answer at {exc.url} ({exc.status}: {exc.reason}) — "
                          f"port truth cannot tell 'absent' from 'unloaded'"}
    except Exception as exc:  # noqa: BLE001 - fail-closed: an unreadable port is dark
        return {"port": "dark", "model": None, "detail": f"guest read failed ({exc})"}
    models = data.get("data") if isinstance(data, dict) else data
    if not isinstance(models, list) or not models:
        return {"port": "up", "model": "unknown-shape",
                "detail": f"answered but not a model list: {str(data)[:120]}"}
    # REVIEW F3 (ecmw24f): the list must be the GUEST's list. Match the model by the id
    # the guest serves; an absent id is a wrong-engine answer, never a status read off
    # someone else's model (first-loaded-model-wins was the 11-28 trap, inverted).
    named = [m for m in models if isinstance(m, dict) and str(m.get("id")) == GUEST_MODEL_ID]
    if not named:
        seen = [str(m.get("id")) for m in models if isinstance(m, dict)][:4]
        return {"port": "up", "model": "absent",
                "detail": f"port answers with {seen} — no {GUEST_MODEL_ID!r} in the list "
                          f"(wrong engine on this port?) — port-up is not guest-up"}
    for m in named:
        status = m.get("status")
        value = status.get("value") if isinstance(status, dict) else None
        if isinstance(value, str):
            return {"port": "up", "model": value,
                    "detail": f"port answers, model {m.get('id')!r} reports {value!r}"}
    return {"port": "up", "model": "unknown-shape",
            "detail": f"{GUEST_MODEL_ID!r} listed with no status.value"}


def driver_free(gpu_index: int = 0, timeout: float = 2.0) -> int | None:
    """Driver truth from inside the container (heal's R1 reader, same law): a number, or
    None. Never a guess, never an optimistic default."""
    try:
        run = subprocess.run(
            ["nvidia-smi", "-i", str(int(gpu_index)), "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout)
        if run.returncode != 0:
            return None
        return int(float(run.stdout.strip().splitlines()[0]))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def evaluate(*, primary_loaded: bool | None, opens: list[dict], problems: list[str],
             guest: dict, driver_free_mib: int | None, now: float,
             state_dir: str = STATE_DIR) -> dict:
    """The pure predicates. Returns the ONE-READ story (§6.3): holders + named alarms.
    Alarm reasons always carry every leg with its reading, and each alarm its literal."""
    open_names = [w["name"] for w in opens]
    holders = {
        "driver_free_mib": driver_free_mib,
        "primary_loaded": primary_loaded,
        "guest_port": guest.get("port"),
        "guest_model": guest.get("model"),
        "open_window": open_names[0] if open_names else None,
        "open_windows": open_names,
    }
    alarms: list[dict] = []
    next_status = f"bash {SWARMCTL} status"
    for problem in problems:
        alarms.append({"name": "window.unreadable", "window": None,
                       "reason": problem + " — unreadable is not quiet",
                       "next": next_status})
    for w in opens:
        name = w["name"]
        # B-3/B-4 family, window-open leg: port dark while a window is OPEN and the
        # primary sits loaded = the §1 starved predicate. An UNloaded primary with a
        # dark port is the intended PARKED posture — no alarm.
        if guest.get("port") != "up":
            if primary_loaded:
                alarms.append({
                    "name": "guest.starved", "window": name,
                    "reason": f"window {name} OPEN, primary loaded, guest port dark "
                              f"({guest.get('detail')}) — children cannot run",
                    "next": next_status})
        elif guest.get("model") != "loaded":
            alarms.append({
                "name": "guest.dark", "window": name,
                "reason": f"window {name} OPEN and {guest.get('detail')} — port-up is "
                          f"not model-up (restart woke the container, not the model; "
                          f"auto-recovery does exactly this)",
                "next": next_status})
        # empty-window predicate (§6.1): OPEN + zero REGISTERED children + grace elapsed.
        # A missing/wrong-typed delegs field is NOT zero children (unreadable != quiet).
        if w.get("delegs") is None:
            alarms.append({
                "name": "window.unreadable", "window": name,
                "reason": f"window {name}: delegs field unprovable — cannot claim zero "
                          f"children",
                "next": next_status})
        elif not w["delegs"].strip():
            if w.get("yielded_at") is None:
                alarms.append({
                    "name": "window.unreadable", "window": name,
                    "reason": f"window {name}: yielded_at unparseable — cannot date the "
                              f"open", "next": next_status})
            elif now - w["yielded_at"] >= EMPTY_WINDOW_S:
                alarms.append({
                    "name": "window.empty", "window": name,
                    "reason": f"window {name} OPEN for "
                              f"{int(now - w['yielded_at'])} s with zero registered "
                              f"children and nothing to run — the card is held by an "
                              f"empty round",
                    "next": f"bash {SWARMCTL} return --window "
                            f"{_host_path(os.path.join(state_dir, name), state_dir)} --force"})
    degraded = None
    for name in ALARM_PRIORITY:
        if any(a["name"] == name for a in alarms):
            degraded = name
            break
    return {"holders": holders, "alarms": alarms, "degraded": degraded}


def observe(*, status_reader: Callable[[], dict],
            models_reader: Callable[[], object],
            free_reader: Callable[[], int | None] = driver_free,
            state_dir: str = STATE_DIR, log_path: str | None = None,
            now: Callable[[], float] = time.time) -> dict:
    """Read once, evaluate, and run episode memory (once per episode into the ledger).
    Returns the observation for the ping body — the SAME dict whether or not a log home
    exists (without one, episodes still fire in-process and simply do not persist)."""
    try:
        status = status_reader()
        primary_loaded = status.get("loaded") if isinstance(status, dict) else None
    except Exception:  # noqa: BLE001 - the ping itself reports primary truth; watch stays honest
        primary_loaded = None
    opens, problems = window_facts(state_dir)
    guest = guest_truth(models_reader)
    obs = evaluate(primary_loaded=primary_loaded, opens=opens, problems=problems,
                   guest=guest, driver_free_mib=free_reader(), now=now(),
                   state_dir=state_dir)
    live = {f"{a['name']}:{a['window']}": a for a in obs["alarms"]}
    with _lock:
        for key in [k for k in _episodes if k not in live]:
            del _episodes[key]           # resolved episodes may fire again later
        for key, a in live.items():
            if key in _episodes:
                continue                  # already logged this episode: idempotent
            if log_path:
                entry = decision_log.append(
                    log_path, event=a["name"], decision="degraded",
                    reason=a["reason"], operator_hint=a["next"],
                    primary={"loaded": primary_loaded},
                    guest={"port": guest.get("port"), "model": guest.get("model")})
                _episodes[key] = entry["id"]
            else:
                _episodes[key] = "unpersisted"
    obs["episode_ids"] = {k: v for k, v in _episodes.items() if k in live}
    return obs


def reset_episodes() -> None:
    """Bench/test seam: episode memory is process truth, and tests run in one."""
    with _lock:
        _episodes.clear()
