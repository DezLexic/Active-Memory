"""
scheduler.py

Runs the Librarian on a background thread during a configured quiet hour.

Usage:
    pipeline  = Pipeline(...)
    scheduler = Scheduler(pipeline, run_hour=2)
    scheduler.start()   # returns immediately; maintenance runs at 2 am
    # ... later ...
    scheduler.stop()
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from .librarian import Librarian


class Scheduler:
    """
    Background cron-like thread that calls Librarian.run_consolidation()
    once per day at run_hour.

    Parameters
    ----------
    pipeline    A Pipeline instance.  The Librarian is created from
                pipeline._retrieval.
    run_hour    Hour of day (0-23) to run consolidation.  Default 2 (2 am).
    """

    def __init__(self, pipeline, run_hour: int = 2) -> None:
        self._librarian     = Librarian(pipeline._retrieval)
        self._run_hour      = run_hour
        self._last_run_date = None
        self._running       = False
        self._thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background scheduler thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the scheduler to stop after its current sleep."""
        self._running = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            now   = datetime.now()
            today = now.date()
            if now.hour == self._run_hour and self._last_run_date != today:
                self._last_run_date = today
                print(f"[scheduler] starting consolidation at {now.isoformat()}")
                results = self._librarian.run_consolidation()
                print(f"[scheduler] consolidation results: {results}")
            time.sleep(60)

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Scheduler(librarian={self._librarian!r}, "
            f"run_hour={self._run_hour})"
        )
