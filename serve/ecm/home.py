"""Where the ECM persists its state (the decision log). One home, visible from both sides.

The home is the rw=true, persistent, host-visible bind the primary's container already
mounts as /out/window (verified via podman inspect, 11-27): inside the container the ECM
writes /out/window/inkling/ecm; host tools see the same bytes under the bind's source.
No new mounts, no container recreate, survives restarts both ways - and notably NOT a
/tmp file: the modefile's restart-fragility is exactly what W2e replaces.

ECM_STATE_DIR overrides for explicit deployments. No home means UNASSIGNED, and the ping
says so - unassigned is a truth, not an error. No machine-specific paths live here:
host-side tools reach the same directory by passing ECM_STATE_DIR.
"""
from __future__ import annotations

import os

CONTAINER_ROOT = "/out/window/inkling/ecm"  # the primary's serve sees its own side of the bind


def state_root() -> str | None:
    env = os.environ.get("ECM_STATE_DIR")
    if env:
        return env
    anchor = os.path.dirname(CONTAINER_ROOT)  # /out/window/inkling - exists when the bind is mounted
    if os.path.isdir(anchor) and os.access(anchor, os.W_OK):
        return CONTAINER_ROOT
    return None


def decision_log_path() -> str | None:
    root = state_root()
    return os.path.join(root, "decisions.jsonl") if root else None
