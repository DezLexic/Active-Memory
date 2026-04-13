"""
observer.py

Updates the Bucket's rolling summary whenever a batch of message pairs is
evicted from the recent-messages stack.

No model call is made unless pairs have actually been popped (popped_pairs
is not None and non-empty), so turns that do not trigger eviction are free.

One LLM call per batch regardless of batch size.  The prompt includes
all evicted pairs in sequence and instructs the model to extend the existing
summary to capture everything that was just lost.
"""

from __future__ import annotations

import json
import logging

from .bucket        import Bucket
from .backends.base import LLMBackend
from .monitor       import ProcessMonitor

logger = logging.getLogger(__name__)


class Observer:
    """
    Maintains the conversation summary stored inside the Bucket.

    Parameters
    ----------
    backend             Any LLMBackend-conforming object.
    max_summary_length  Soft character limit per topic node summary.
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_summary_length: int = 150,
    ) -> None:
        self._backend             = backend
        self._max_summary_length  = max_summary_length

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        bucket: Bucket,
        popped_pairs: list[dict[str, str]] | None,
    ) -> None:
        """
        Update the Bucket's topic tree to reflect a batch of evicted Q/A pairs.

        If popped_pairs is None or empty the stack was not yet full and
        nothing was lost, so this method returns immediately without making
        any calls.  When a batch is provided all pairs are folded into a
        single prompt and exactly one LLM call is made regardless of batch
        size.

        Parameters
        ----------
        bucket       The shared Bucket whose topic_tree will be updated.
        popped_pairs The list returned by bucket.push_message() when it
                     evicts a batch of old pairs.  Each element is a dict
                     with keys "question" and "response".  Pass None (or
                     the raw return value) when no eviction occurred.
        """
        if not popped_pairs:
            return

        current_tree_json = json.dumps(bucket.topic_tree, indent=2)
        pairs_text = "\n\n".join(
            f"Q: {p['question'].strip()}\nA: {p['response'].strip()}"
            for p in popped_pairs
        )

        prompt = (
            "You are maintaining a structured topic tree that summarises a conversation.\n\n"
            "CURRENT TOPIC TREE (JSON):\n"
            f"{current_tree_json}\n\n"
            "MESSAGE PAIRS BEING REMOVED FROM RECENT HISTORY:\n"
            f"{pairs_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Update the topic tree to incorporate information from the evicted messages.\n"
            "2. Each topic node must have this exact shape:\n"
            '   {"id": "<slug>", "title": "<short label>", "summary": "<prose>", '
            '"subtopics": [...], "created_at": <unix_ts>, "updated_at": <unix_ts>, '
            '"updated_at_turn": <int>}\n'
            f"3. Keep each node's summary under {self._max_summary_length} characters.\n"
            "4. Merge related information into existing topics — do not create duplicates.\n"
            "5. Use subtopics for specific sub-decisions under a broader topic.\n"
            f"6. The current turn count is {bucket._turn_count}. "
            "Set updated_at_turn to this value for any node you create or modify.\n"
            "7. Never delete existing topics — only update or add.\n"
            "8. Return ONLY the updated JSON object with a top-level \"topics\" array. "
            "No preamble, no commentary, no markdown fences."
        )

        with ProcessMonitor("observer updating topic tree"):
            raw = self._backend.chat([{"role": "user", "content": prompt}])

        # Strip markdown fences if model wraps response
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(l for l in lines if not l.startswith("```"))

        try:
            parsed = json.loads(raw)
            if "topics" not in parsed or not isinstance(parsed["topics"], list):
                raise ValueError("Response missing 'topics' list")
            bucket.topic_tree = parsed
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Observer: failed to parse topic tree JSON (%s). Tree unchanged.", exc)

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Observer(backend={self._backend!r}, "
            f"max_summary_length={self._max_summary_length})"
        )
