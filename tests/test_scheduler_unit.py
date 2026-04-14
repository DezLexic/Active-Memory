"""
test_scheduler_unit.py

Unit tests for Scheduler — init defaults, start/stop lifecycle, the
background _loop logic, and __repr__.

No real LLM calls.  Uses a _FakePipeline with real ChromaDB via pytest
tmp_path.  Threading tests use short timeouts (1-2 s) to prevent hangs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import threading
import time
from datetime import datetime, date
from unittest.mock import patch, MagicMock

import pytest

from active_memory.scheduler import Scheduler
from active_memory.retrieval import Retrieval


# ── Helpers ──────────────────────────────────────────────────────────────────

class _FakePipeline:
    """Minimal stand-in for Pipeline — Scheduler only touches _retrieval."""

    def __init__(self, tmp_path):
        self._retrieval = Retrieval(chroma_path=str(tmp_path / "sched_chroma"))


# ── TestInit ─────────────────────────────────────────────────────────────────

class TestInit:

    def test_scheduler_creates_librarian_from_pipeline(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe)
        assert sched._librarian is not None

    def test_default_run_hour_is_two(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe)
        assert sched._run_hour == 2

    def test_custom_run_hour(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe, run_hour=14)
        assert sched._run_hour == 14

    def test_not_running_initially(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe)
        assert sched._running is False

    def test_no_thread_initially(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe)
        assert sched._thread is None


# ── TestStartStop ────────────────────────────────────────────────────────────

class TestStartStop:

    def test_start_sets_running_true(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe)
        # Patch _loop so the background thread does nothing and exits quickly.
        with patch.object(sched, "_loop"):
            sched.start()
            try:
                assert sched._running is True
            finally:
                sched.stop()
                sched._thread.join(timeout=2)

    def test_start_creates_daemon_thread(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe)
        with patch.object(sched, "_loop"):
            sched.start()
            try:
                assert sched._thread is not None
                assert sched._thread.daemon is True
            finally:
                sched.stop()
                sched._thread.join(timeout=2)

    def test_stop_sets_running_false(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe)
        with patch.object(sched, "_loop"):
            sched.start()
            sched.stop()
            assert sched._running is False
            sched._thread.join(timeout=2)

    def test_thread_stops_after_stop(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe)

        # Use the real _loop but make time.sleep immediately set _running=False
        # so the loop exits after one iteration.
        def _quick_sleep(_seconds):
            sched._running = False

        with patch("active_memory.scheduler.time.sleep", side_effect=_quick_sleep):
            sched.start()
            sched._thread.join(timeout=2)
            assert not sched._thread.is_alive()


# ── TestLoop ─────────────────────────────────────────────────────────────────

class TestLoop:

    def test_loop_exits_when_running_is_false(self, tmp_path):
        """_loop() should return immediately when _running is already False."""
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe)
        sched._running = False

        # Run _loop in a thread with a short timeout — it must exit on its own.
        t = threading.Thread(target=sched._loop)
        t.start()
        t.join(timeout=2)
        assert not t.is_alive()

    def test_consolidation_runs_at_correct_hour(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe, run_hour=3)

        fake_now = datetime(2026, 4, 13, 3, 0, 0)  # hour == run_hour

        call_count = 0

        def _stop_after_first_sleep(_seconds):
            nonlocal call_count
            call_count += 1
            sched._running = False

        with patch("active_memory.scheduler.datetime") as mock_dt, \
             patch("active_memory.scheduler.time.sleep", side_effect=_stop_after_first_sleep), \
             patch.object(sched._librarian, "run_consolidation", return_value={}) as mock_consol:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            sched._running = True
            sched._loop()

            mock_consol.assert_called_once()

    def test_consolidation_does_not_run_twice_same_day(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe, run_hour=3)

        fake_now = datetime(2026, 4, 13, 3, 0, 0)

        iteration = 0

        def _sleep_side_effect(_seconds):
            nonlocal iteration
            iteration += 1
            if iteration >= 2:
                sched._running = False

        with patch("active_memory.scheduler.datetime") as mock_dt, \
             patch("active_memory.scheduler.time.sleep", side_effect=_sleep_side_effect), \
             patch.object(sched._librarian, "run_consolidation", return_value={}) as mock_consol:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            sched._running = True
            sched._loop()

            # Two loop iterations at the same hour/date — consolidation once only.
            mock_consol.assert_called_once()


# ── TestRepr ─────────────────────────────────────────────────────────────────

class TestRepr:

    def test_repr_contains_run_hour(self, tmp_path):
        pipe = _FakePipeline(tmp_path)
        sched = Scheduler(pipe, run_hour=5)
        r = repr(sched)
        assert "run_hour=5" in r
