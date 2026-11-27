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
from serve.config import kv_rails_words, rails_clock_spoken
from serve.service import kv_view

RAILS = {"--kv-keep-warm-adaptive", "--kv-admit-negotiate", "--no-kv-keep-warm-adaptive", "--no-kv-admit-negotiate"}
CLOCK = ["--kv-idle-demote-seconds", "30"]   # the substrate every rail rides (shared serve gates on idle > 0)


class RailsClock(unittest.TestCase):
    """The clock law (Q0): a rails word without its substrate is a boot error (adaptive) or a silence
    (negotiate) — so the argv's `--kv-idle-demote-seconds` decides whether the words ride at all."""

    def test_the_clock_helper(self):
        self.assertFalse(rails_clock_spoken([]))
        self.assertFalse(rails_clock_spoken(["--kv-idle-demote-seconds", "0"]))
        self.assertFalse(rails_clock_spoken(["--kv-idle-demote-seconds", "abc"]))
        self.assertFalse(rails_clock_spoken(["--kv-idle-demote-seconds", "-5"]))
        self.assertFalse(rails_clock_spoken(["--model", "x", "--kv-idle-demote-seconds"]))  # trailing, valueless
        self.assertTrue(rails_clock_spoken(CLOCK))
        self.assertTrue(rails_clock_spoken(["--kv-idle-demote-seconds", "1"]))
        self.assertTrue(rails_clock_spoken(["--kv-idle-demote-seconds", "1e3"]))
        # the engine's law is last-flag-wins, and flag_value reads the LAST occurrence
        self.assertTrue(rails_clock_spoken(["--kv-idle-demote-seconds", "0", "--kv-idle-demote-seconds", "30"]))
        self.assertFalse(rails_clock_spoken([*CLOCK, "--kv-idle-demote-seconds", "0"]))
        # the equals form is documented not to parse (the argv builder emits separate words): reads as no clock
        self.assertFalse(rails_clock_spoken(["--kv-idle-demote-seconds=30"]))

    def test_no_probe_ever_runs_without_the_clock(self):
        # the comment's claim, pinned: the clock gate comes first, so a refused-probe that WOULD refuse is
        # never even called — a probe-order regression must fail this, not pass silently
        calls = []
        def refused(flag):
            calls.append(flag)
            return False
        self.assertEqual(kv_rails_words(RAILS, [], refused=refused), ([], ["--kv-keep-warm-adaptive", "--kv-admit-negotiate"]))
        self.assertEqual(calls, [])

    def test_no_clock_means_no_rails_words_and_both_are_named(self):
        # even a model that accepts every rail gets nothing: the substrate comes first (and no probe runs)
        words, skipped = kv_rails_words(RAILS, [], refused=lambda flag: False)
        self.assertEqual(words, [])
        self.assertEqual(skipped, ["--kv-keep-warm-adaptive", "--kv-admit-negotiate"])
        # a spoken clock restores the posture
        self.assertEqual(kv_rails_words(RAILS, CLOCK, refused=lambda flag: False),
                         (["--kv-keep-warm-adaptive", "--kv-admit-negotiate"], []))


class RailsDefault(unittest.TestCase):
    """M4's posture: the rails ride by default on any engine that has the flags and has been given its
    clock, exactly as grammar words default on — and the negation form is the explicit off, honored over
    the default."""

    def test_advertised_rails_ride_with_their_clock(self):
        self.assertEqual(kv_rails_words(RAILS, CLOCK), (["--kv-keep-warm-adaptive", "--kv-admit-negotiate"], []))
        self.assertEqual(kv_rails_words(RAILS, [*CLOCK, "--kv-keep-warm-adaptive"]), (["--kv-admit-negotiate"], []))
        self.assertEqual(kv_rails_words(RAILS, [*CLOCK, "--no-kv-admit-negotiate"]), (["--kv-keep-warm-adaptive"], []))

    def test_an_engine_without_the_flags_starts_byte_identical(self):
        self.assertEqual(kv_rails_words({"--grammar", "--model"}, ["--model", "x"]), ([], []))
        self.assertEqual(kv_rails_words(None, []), ([], []))      # blind probe: add nothing, guess nothing

    def test_a_model_that_refuses_a_rail_is_skipped_and_named(self):
        # the binary can advertise the rails while this model's adapter refuses them (qwen4exp, which has no
        # keep-warm rails): the rail is dropped and named, never appended — the engine is told the truth
        refused = {"--kv-keep-warm-adaptive"}
        words, skipped = kv_rails_words(RAILS, CLOCK, refused=lambda flag: flag in refused)
        self.assertEqual(words, ["--kv-admit-negotiate"])
        self.assertEqual(skipped, ["--kv-keep-warm-adaptive"])
        # a model that wants only one of its rails still gets the other
        words, skipped = kv_rails_words(RAILS, CLOCK, refused=lambda flag: True)
        self.assertEqual(words, [])
        self.assertEqual(skipped, ["--kv-keep-warm-adaptive", "--kv-admit-negotiate"])
        # an explicit off is an intent, not a refusal: the negated rail is neither added nor named as skipped
        words, skipped = kv_rails_words(RAILS, [*CLOCK, "--no-kv-admit-negotiate"], refused=lambda flag: False)
        self.assertEqual(words, ["--kv-keep-warm-adaptive"])
        self.assertEqual(skipped, [])
        words, skipped = kv_rails_words(RAILS, [*CLOCK, "--no-kv-keep-warm-adaptive"], refused=lambda flag: True)
        self.assertEqual(words, [])
        self.assertEqual(skipped, ["--kv-admit-negotiate"])

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
