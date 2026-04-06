"""
active_agent.py

The Active Agent is the only model call the user waits for.  Its job is
exactly one thing: receive the assembled Bucket and respond.

It does not touch the memory store.  It does not update the summary.  It
does not evaluate what to remember.  All of that happens before or after
this call.  The Agent only reads from the Bucket.

One Ollama call per turn.

    system message  ->  bucket.to_context_string()
    user message    ->  bucket.current_prompt
"""

from __future__ import annotations

import ollama
from .bucket import Bucket

class ActiveAgent:
    """
    Conversational agent that reads the assembled Bucket and responds.

    Parameters
    ----------
    model   Ollama model name used for response generation.
    """

    def __init__(self, model: str = "gemma3:4b") -> None:
        self._model = model

    # ── Public API ─────────────────────────────────────────────────────────────

    def respond(self, bucket: Bucket) -> str:
        """
        Generate a response from the assembled Bucket.

        Assembles the full context string from the Bucket and sends it as
        the system prompt.  Sends bucket.current_prompt as the user message.
        Returns the model's response as a plain string.

        Parameters
        ----------
        bucket  The shared context window.  All slots (summary, recent
                messages, memories, current_prompt) should be populated
                before calling this method.
        """
        context = bucket.to_context_string()

        result = ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": context},
                {"role": "user",   "content": bucket.current_prompt},
            ],
        )

        return result["message"]["content"].strip()

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ActiveAgent(model={self._model!r})"
