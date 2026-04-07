"""
monitor.py

Lightweight context manager that prints a warning when a named process
runs longer than expected.  Used to diagnose hung LLM calls.

Usage:
    with ProcessMonitor("observer writing summary"):
        backend.chat(...)

Output (after 2 minutes, then every 1 minute):
    [monitor] observer writing summary for 2 minutes
    [monitor] observer writing summary for 3 minutes
"""

from __future__ import annotations

import threading
import time


class ProcessMonitor:
    """
    Context manager that warns via print() if a named operation runs long.

    Parameters
    ----------
    name            Human-readable label printed in warnings.
    warn_after      Seconds before the first warning. Default: 120 (2 min).
    repeat_every    Seconds between subsequent warnings. Default: 60 (1 min).
    """

    def __init__(
        self,
        name: str,
        warn_after: int = 120,
        repeat_every: int = 60,
    ) -> None:
        self._name         = name
        self._warn_after   = warn_after
        self._repeat_every = repeat_every
        self._start_time: float | None = None
        self._timer: threading.Timer | None = None
        self._cancelled    = False

    def __enter__(self) -> "ProcessMonitor":
        self._start_time = time.time()
        self._cancelled  = False
        self._schedule(self._warn_after)
        return self

    def __exit__(self, *args) -> None:
        self._cancelled = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule(self, delay: float) -> None:
        self._timer = threading.Timer(delay, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        if self._cancelled:
            return
        elapsed_minutes = int((time.time() - self._start_time) / 60)
        print(f"[monitor] {self._name} for {elapsed_minutes} minutes")
        self._schedule(self._repeat_every)
