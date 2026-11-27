"""Calibrated VRAM fit-points (v1). The law these encode (ECM-PLAN §4, ECM-W2-DESIGN §7):
the demand model is CALIBRATED to measured points, never summed from components — plan is
not ceiling. The measured demand at this point (25.2 GiB) EXCEEDS the lease it fits inside
(21,504 MiB); any check that compares a naive component sum against the card size would
refuse a configuration that in fact runs.

The floor below is a calibration constant, not a derivation: do not compute it from cache
and kv cell counts until W3 measures the live p3-shape grid (ECM-W2-DESIGN §9 item 2).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FitPoint:
    name: str
    cache_cells: int
    kv_reserve_cells: int
    demand_mib: int      # measured demand footprint at this operating point
    lease_floor_mib: int  # smallest lease observed to carry it (calibrated, not derived)


CALIBRATION_V1 = FitPoint("v1-p2-era", cache_cells=17000, kv_reserve_cells=8192,
                          demand_mib=25_805, lease_floor_mib=21_504)


def demand_fits(lease_applied_mib: int, point: FitPoint = CALIBRATION_V1) -> bool:
    """Does a lease carry the calibrated demand? Compare against the calibrated floor —
    NEVER against demand_mib, which is a footprint, not a reservation."""
    return lease_applied_mib >= point.lease_floor_mib
