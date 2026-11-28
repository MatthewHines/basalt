"""The ECM decision log: append-only JSONL, one entry per transition or decision.

The log is readable WITHOUT the ECM (a human and a bench parse the same file), ids are
monotonic, corrections are NEW entries — history is never rewritten. A refusal entry
always carries the engine's reason verbatim and one literal operator remedy, because an
alarm nobody can act on is noise. Grammar is ECM-W2-DESIGN.md §6; the event namespace is
an owning-subsystem ENUM FAMILY (the grammar grows with the slices; the family keeps
machine checks honest).
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

EVENT_RE = re.compile(
    r"^(guest|primary|lease|mode|socket|window|heal|handback|wake_detector)"
    r"\.[a-z][a-z_.]*$")
DECISIONS = frozenset(("observe", "allow", "refuse", "rollback", "degraded", "already"))
SOURCES = frozenset(("ecm", "bench", "manual"))
# decisions that are failures-of-some-kind: they MUST carry a verbatim reason and a literal remedy
NEEDS_REASON = frozenset(("refuse", "rollback", "degraded"))
ID_RE = re.compile(r"^ecm-(\d+)-([0-9a-f]{4})$")

REQUIRED_TYPES = {"event": str, "decision": str, "source": str}


def new_id(previous: str | None) -> str:
    """Monotonic id: same-second appends cannot collide, because the counter component comes
    from the previous entry, not from the clock."""
    epoch = int(time.time())
    seq = 0
    if previous:
        m = ID_RE.match(previous)
        if m:
            prev_epoch, prev_seq = int(m.group(1)), int(m.group(2), 16)
            if prev_epoch > epoch:  # clock ran backwards: stay monotonic anyway
                epoch, seq = prev_epoch, prev_seq + 1
            elif prev_epoch == epoch:
                if prev_seq < 0xFFFF:
                    seq = prev_seq + 1
                else:  # 65k entries in one second: widen honestly
                    epoch, seq = prev_epoch + 1, 0
    return f"ecm-{epoch}-{seq:04x}"


def validate(entry: dict) -> None:
    """Raise ValueError on any entry a reader would have to guess about. Strict before
    write, so the file never contains a line the validator would reject."""
    if not isinstance(entry, dict):
        raise ValueError(f"entry is {type(entry).__name__}, not an object")
    for field, typ in REQUIRED_TYPES.items():
        value = entry.get(field)
        if not isinstance(value, typ):
            raise ValueError(f"{field} must be {typ.__name__}, got {value!r}")
    if not EVENT_RE.match(entry["event"]):
        raise ValueError(f"event {entry['event']!r} is outside the owning-subsystem family")
    if entry["decision"] not in DECISIONS:
        raise ValueError(f"decision {entry['decision']!r} not in {sorted(DECISIONS)}")
    if entry["source"] not in SOURCES:
        raise ValueError(f"source {entry['source']!r} not in {sorted(SOURCES)}")
    if entry["decision"] in NEEDS_REASON:
        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"decision {entry['decision']} must carry the reason verbatim")
        hint = entry.get("operator_hint")
        if not isinstance(hint, str) or not hint.strip():
            raise ValueError(f"decision {entry['decision']} must carry a literal operator_hint")
    m = ID_RE.match(entry.get("id", ""))
    if entry.get("id") is not None and not m:
        raise ValueError(f"id {entry['id']!r} does not match ecm-<epoch>-<4hex>")
    for field in ("guest", "primary", "lease"):
        if entry.get(field) is not None and not isinstance(entry[field], dict):
            raise ValueError(f"{field} must be an object or absent")


def append(path: str, *, event: str, decision: str = "observe", source: str = "ecm",
           reason: str | None = None, operator_hint: str | None = None,
           guest: dict | None = None, primary: dict | None = None,
           lease: dict | None = None) -> dict:
    """Append one entry and return it (id and ts included). The validator runs BEFORE the
    line is written: a bad entry raises and the file keeps its old length."""
    entry: dict = {
        "id": new_id(last_id(path)),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event, "decision": decision, "source": source,
    }
    for field, value in (("reason", reason), ("operator_hint", operator_hint),
                         ("guest", guest), ("primary", primary), ("lease", lease)):
        if value is not None:
            entry[field] = value
    validate(entry)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # REVIEW F6 (ecmw24f): the log is rw-bound into both engines — concurrent appenders
    # must not interleave id-selection with the write (two readers, same previous id →
    # duplicate ids → the file becomes UNREADABLE to read()'s monotonic law). One
    # exclusive flock on a sibling lockfile spans last_id→fsync; a fresh file's first
    # writer may briefly duplicate id -0000 — the reader names it, never guesses.
    try:
        import fcntl
    except ImportError:      # non-POSIX: same single-writer assumption as before F6
        fcntl = None         # type: ignore[assignment]
    lock_fh = open(path + ".lock", "w") if fcntl else None
    try:
        if fcntl and lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        entry["id"] = new_id(last_id(path))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    finally:
        if fcntl and lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
            lock_fh.close()
    return entry


def last_id(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            last = None
            for line in fh:
                if line.strip():
                    last = line
            if last is None:
                return None
            return json.loads(last.decode("utf-8")).get("id")
    except FileNotFoundError:
        return None


def read(path: str) -> list[dict]:
    """Every entry, oldest first. A corrupt line RAISES with its line number — the reader
    never silently skips, and neither does the bench built on it."""
    out: list[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    raise ValueError(f"{path}:{number} is not JSON") from None
    except FileNotFoundError:
        return []
    _assert_monotonic(out, path)
    return out


def _assert_monotonic(entries: list[dict], path: str) -> None:
    key = None
    for e in entries:
        m = ID_RE.match(e.get("id") or "")
        if not m:
            raise ValueError(f"{path}: entry id {e.get('id')!r} is not well-formed")
        k = (int(m.group(1)), int(m.group(2), 16))
        if key is not None and k <= key:
            raise ValueError(f"{path}: ids not monotonic at {e['id']}")
        key = k
