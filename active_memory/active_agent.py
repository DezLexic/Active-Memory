"""
active_agent.py

The Active Agent is the only model call the user waits for.  Its job is
exactly one thing: receive the assembled Bucket and respond.

It does not touch the memory store.  It does not update the summary.  It
does not evaluate what to remember.  All of that happens before or after
this call.  The Agent only reads from the Bucket.

One LLM call per turn via the injected backend.

    system message  ->  bucket.to_context_string()
    user message    ->  bucket.current_prompt
"""

from __future__ import annotations

from .bucket        import Bucket
from .backends.base import LLMBackend


class ActiveAgent:
    """
    Conversational agent that reads the assembled Bucket and responds.

    Parameters
    ----------
    backend     Any LLMBackend-conforming object.  Responsible for the
                provider-specific call and returns a plain response string.
    """

    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    # ── Public API ─────────────────────────────────────────────────────────────

    def respond(self, bucket: Bucket) -> str:
        """
        Generate a response from the assembled Bucket.

        Assembles the full context string from the Bucket and sends it as
        the system message.  Sends bucket.current_prompt as the user message.
        Returns the model's response as a plain string.

        Parameters
        ----------
        bucket  The shared context window.  All slots (summary, recent
                messages, memories, current_prompt) should be populated
                before calling this method.
        """
        context  = bucket.to_context_string()
        messages = [
            {"role": "system", "content": context},
            {"role": "user",   "content": bucket.current_prompt},
        ]
        return self._backend.chat(messages)

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"ActiveAgent(backend={self._backend!r})"
