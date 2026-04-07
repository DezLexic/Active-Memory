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

from .bucket        import Bucket
from .backends.base import LLMBackend
from .monitor       import ProcessMonitor


class Observer:
    """
    Maintains the conversation summary stored inside the Bucket.

    Parameters
    ----------
    backend     Any LLMBackend-conforming object.
    max_words   Soft target word count passed to the model as a guide.
                Keeps the summary from growing unboundedly.
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_words: int = 600,
    ) -> None:
        self._backend   = backend
        self._max_words = max_words

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        bucket: Bucket,
        popped_pairs: list[dict[str, str]] | None,
    ) -> None:
        """
        Update the Bucket's summary to reflect a batch of evicted Q/A pairs.

        If popped_pairs is None or empty the stack was not yet full and
        nothing was lost, so this method returns immediately without making
        any calls.  When a batch is provided all pairs are folded into a
        single prompt and exactly one LLM call is made regardless of batch
        size.

        Parameters
        ----------
        bucket       The shared Bucket whose summary will be updated.
        popped_pairs The list returned by bucket.push_message() when it
                     evicts a batch of old pairs.  Each element is a dict
                     with keys "question" and "response".  Pass None (or
                     the raw return value) when no eviction occurred.
        """
        if not popped_pairs:
            return

        current_summary = bucket.summary.strip() or "(no summary yet)"

        pairs_text = "\n\n".join(
            f"Q: {p['question'].strip()}\nA: {p['response'].strip()}"
            for p in popped_pairs
        )

        prompt = (
            f"You are maintaining a rolling summary of a conversation.\n\n"
            f"CURRENT SUMMARY:\n{current_summary}\n\n"
            f"MESSAGE PAIRS BEING REMOVED FROM RECENT HISTORY:\n"
            f"{pairs_text}\n\n"
            f"Update the summary to incorporate the information in those message "
            f"pairs so nothing important is lost. Preserve all decisions, "
            f"preferences, constraints, and directions that have been stated. "
            f"Keep the summary under {self._max_words} words. "
            f"Return only the updated summary with no preamble or commentary."
        )

        with ProcessMonitor("observer writing summary"):
            updated_summary = self._backend.chat([{"role": "user", "content": prompt}])
        bucket.set_summary(updated_summary)

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Observer(backend={self._backend!r}, "
            f"max_words={self._max_words})"
        )
