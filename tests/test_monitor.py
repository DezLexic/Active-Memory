"""
tests/test_monitor.py

Tests for ProcessMonitor.  Uses very short thresholds (milliseconds) so
no real waiting is needed beyond a short sleep to let threads fire.
"""

import time

import pytest

from active_memory.monitor import ProcessMonitor


def test_no_warning_on_fast_completion(capsys):
    """Timer is cancelled before it fires — nothing printed."""
    with ProcessMonitor("fastop", warn_after=0.05, repeat_every=0.05):
        pass  # exits well before 50 ms
    time.sleep(0.15)  # give any leaked timer a chance to (not) fire
    assert capsys.readouterr().out == ""


def test_warning_fires_after_threshold(capsys):
    """First warning prints after warn_after seconds."""
    with ProcessMonitor("slowop", warn_after=0.05, repeat_every=10):
        time.sleep(0.15)
    out = capsys.readouterr().out
    assert "[monitor] slowop" in out


def test_warning_contains_elapsed_minutes(capsys):
    """_fire() prints the correct elapsed minute count."""
    monitor = ProcessMonitor("observer writing summary", warn_after=999, repeat_every=999)
    monitor._start_time = time.time() - 150  # pretend 2.5 minutes elapsed
    monitor._cancelled  = False
    monitor._fire()
    monitor._cancelled  = True
    if monitor._timer:
        monitor._timer.cancel()
    out = capsys.readouterr().out
    assert "2 minutes" in out
    assert "observer writing summary" in out


def test_repeated_warnings_fire(capsys):
    """Multiple warnings print at repeat_every intervals."""
    with ProcessMonitor("slowop", warn_after=0.05, repeat_every=0.05):
        time.sleep(0.35)
    lines = [l for l in capsys.readouterr().out.splitlines() if "[monitor]" in l]
    assert len(lines) >= 2


def test_no_fire_after_exit(capsys):
    """No warnings print after the context manager exits."""
    with ProcessMonitor("slowop", warn_after=0.05, repeat_every=0.05):
        time.sleep(0.12)
    capsys.readouterr()  # discard anything printed during the context
    time.sleep(0.25)     # wait for any leaked timers to potentially fire
    assert capsys.readouterr().out == ""
