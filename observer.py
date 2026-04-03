"""
observer.py

Updates the Bucket's rolling summary whenever a message pair is evicted
from the recent-messages stack.

No model call is made unless a pair has actually been popped (popped_pair
is not None), so turns that do not trigger eviction are free.

One Ollama call per eviction.  The prompt instructs the model to extend
the existing summary to capture what was just lost, preserving any
decisions, preferences, constraints, or directions mentioned in the pair.
"""

from __future__ import annotations

import ollama
from bucket import Bucket


class Observer:
    """
    Maintains the conversation summary stored inside the Bucket.

    Parameters
    ----------
    model       Ollama model name used for summarisation.
    max_words   Soft target word count passed to the model as a guide.
                Keeps the summary from growing unboundedly.
    """

    def __init__(self, model: str = "gemma3:4b", max_words: int = 600) -> None:
        self._model     = model
        self._max_words = max_words

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, bucket: Bucket, popped_pair: dict[str, str] | None) -> None:
        """
        Update the Bucket's summary to reflect a recently evicted Q/A pair.

        If popped_pair is None the stack was not yet full and nothing was
        lost, so this method returns immediately without making any calls.

        Parameters
        ----------
        bucket      The shared Bucket whose summary will be updated.
        popped_pair The dict returned by bucket.push_message() when it
                    evicts an old pair: {"question": str, "response": str}.
                    Pass None (or the raw return value) when no eviction
                    occurred.
        """
        if popped_pair is None:
            return

        current_summary = bucket.summary.strip() or "(no summary yet)"
        question        = popped_pair["question"].strip()
        response        = popped_pair["response"].strip()

        prompt = (
            f"You are maintaining a rolling summary of a conversation.\n\n"
            f"CURRENT SUMMARY:\n{current_summary}\n\n"
            f"MESSAGE PAIR BEING REMOVED FROM RECENT HISTORY:\n"
            f"Q: {question}\n"
            f"A: {response}\n\n"
            f"Update the summary to incorporate the information in that message "
            f"pair so nothing important is lost. Preserve all decisions, "
            f"preferences, constraints, and directions that have been stated. "
            f"Keep the summary under {self._max_words} words. "
            f"Return only the updated summary with no preamble or commentary."
        )

        result = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )

        updated_summary = result["message"]["content"].strip()
        bucket.set_summary(updated_summary)

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Observer(model={self._model!r}, max_words={self._max_words})"
