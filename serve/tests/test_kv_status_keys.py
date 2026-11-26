"""M4: the keep-warm rails' status keys. The engine prints `INFO kv...` facts at its settle points (the same
places its human ledger lines print); serve republishes them under /v1/status as a `kv` block, read — never
popped — from the engine's shared info dict. No facts, no block: status answers None, never a guess."""
from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.engine.process import ProcessEngine
from serve.service import kv_view

FAKE = r'''
import sys
print("INFO engine=fake context=4096 kv_rails=1 kv_rails_adaptive=1 kv_rails_negotiate=1 kv_rails_low_mib=8192", flush=True)
print("READY 4096 stop", flush=True)
for raw in sys.stdin:
    if raw.startswith("GEN"):
        print("T 65", flush=True)
        print("INFO kv0_band=pressure kv0_kept=0 kv0_moved_mib=2041", flush=True)
        print("INFO kv1_band=room kv1_kept=-1 kv1_moved_mib=0", flush=True)
        print("T 66", flush=True)
        print("DONE 2 3 1.0 1.0 length 0 0 0", flush=True)
    elif raw.startswith("QUIT"):
        break
'''


class KvView(unittest.TestCase):
    def test_no_facts_no_block(self):
        self.assertIsNone(kv_view({}))
        self.assertIsNone(kv_view({"engine": "fake"}))

    def test_the_rails_posture_alone_is_a_block_without_lanes(self):
        view = kv_view({"kv_rails_adaptive": 1, "kv_rails_negotiate": 0, "kv_rails_low_mib": 8192})
        self.assertEqual(view, {"rails": {"adaptive": 1, "negotiate": 0, "low_mib": 8192}, "lanes": None})

    def test_lanes_read_in_order_and_the_view_never_pops(self):
        info = {"kv1_band": "room", "kv1_kept": -1, "kv1_moved_mib": 0,
                "kv0_band": "pressure", "kv0_kept": 0, "kv0_moved_mib": 2041}
        before = dict(info)
        view = kv_view(info)
        assert view is not None
        self.assertEqual([lane["band"] for lane in view["lanes"]], ["pressure", "room"])
        self.assertEqual(view["lanes"][1], {"band": "room", "kept_cells": -1, "moved_mib": 0})
        self.assertEqual(info, before)   # the engine's dict is shared; the view only reads


class FactsFromTheEngine(unittest.TestCase):
    def test_the_settle_facts_survive_the_pump_under_their_lane_keys(self):
        with tempfile.TemporaryDirectory() as d:
            script, log = Path(d) / "fake_engine.py", Path(d) / "engine.log"
            script.write_text(FAKE, encoding="utf-8")
            real = subprocess.Popen
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                    subprocess, "Popen", lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engine = ProcessEngine("basalt", [], log=str(log))
                self.addCleanup(engine.close)
                self.assertEqual(engine.info["kv_rails_adaptive"], 1)              # the boot posture
                self.assertEqual(engine.info["kv_rails_low_mib"], 8192)
                out = [t for t in engine.generate([1, 2, 3], 4, {}, threading.Event()) if t is not None]
            self.assertEqual(out, [65, 66])
            self.assertEqual(engine.info["kv0_band"], "pressure")                  # a settle point's facts
            self.assertEqual(engine.info["kv0_moved_mib"], 2041)
            self.assertEqual(engine.info["kv1_kept"], -1)
            view = kv_view(engine.info)
            assert view is not None
            self.assertEqual([lane["band"] for lane in view["lanes"]], ["pressure", "room"])
            self.assertEqual(view["rails"], {"adaptive": 1, "negotiate": 1, "low_mib": 8192})


if __name__ == "__main__":
    unittest.main()
