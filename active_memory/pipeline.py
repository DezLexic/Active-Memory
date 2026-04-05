"""
pipeline.py

Orchestration layer that wires all components together.

Not an agent — a plain Python class that controls the exact sequence of
operations on every exchange.  The outside world calls one method: chat().

Sequence on every chat() call
------------------------------
1. bucket.set_current_prompt(user_message)
2. retrieval.update_bucket(bucket, user_message)   -- inject relevant memories
3. active_agent.respond(bucket)                    -- the only call user waits for
4. bucket.push_message(user_message, response)     -- advance the recent stack
5a. If a pair was popped -> Observer runs in a background daemon thread
5b. If a pair was popped -> Curator runs in a background daemon thread
6. Return response string

Observer and Curator both receive only the popped pair.  The Curator
evaluates the pair on its own and stores it if it contains an explicit
decision, hard constraint, or established preference.

Both background threads are daemon threads so they never block program exit.
"""

from __future__ import annotations

import threading

from .bucket       import Bucket
from .retrieval    import Retrieval
from .observer     import Observer
from .curator      import Curator
from .active_agent import ActiveAgent


class Pipeline:
    """
    Wires Bucket, Retrieval, Observer, Curator, and ActiveAgent together.

    Parameters
    ----------
    model                  Ollama model name passed to all agents.
    chroma_path            Path to the local Chroma database directory.
    max_recent_messages    Max Q/A pairs kept in the Bucket's recent stack.
    system_instructions    Optional override for the Bucket's system prompt
                           header.  Pass None to use the Bucket default.
    observer_url           Ollama base URL for the Observer agent.
                           Defaults to http://localhost:11434.
    curator_url            Ollama base URL for the Curator agent.
                           Defaults to http://localhost:11434.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        chroma_path: str = "./chroma_db",
        max_recent_messages: int = 5,
        system_instructions: str | None = None,
        observer_url: str = "http://localhost:11434",
        curator_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model

        _CONCISE = "Be concise. Answer directly. Do not over-explain."

        if system_instructions is not None:
            combined_instructions = f"{system_instructions}\n{_CONCISE}"
        else:
            from .bucket import _DEFAULT_SYSTEM_INSTRUCTIONS
            combined_instructions = f"{_DEFAULT_SYSTEM_INSTRUCTIONS}\n{_CONCISE}"

        bucket_kwargs: dict = {
            "max_recent": max_recent_messages,
            "system_instructions": combined_instructions,
        }

        self._bucket      = Bucket(**bucket_kwargs)
        self._retrieval   = Retrieval(chroma_path=chroma_path)
        self._observer    = Observer(model=model, base_url=observer_url)
        self._curator     = Curator(model=model, retrieval=self._retrieval, base_url=curator_url)
        self._agent       = ActiveAgent(model=model)

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_message: str, skip_observer: bool = False) -> str:
        """
        Run the full pipeline for one user exchange.

        Steps
        -----
        1. Set the current prompt on the Bucket.
        2. Retrieve relevant memories and inject them into the Bucket.
        3. Get the agent's response (the only step the user waits for).
        4. Push the completed pair onto the Bucket's recent stack; capture
           any evicted pair.
        5. If a pair was evicted, launch Curator as a background daemon thread.
           Also launch Observer unless skip_observer is True.
        6. Return the response string.

        Parameters
        ----------
        user_message    The raw message from the user.
        skip_observer   When True, the Observer is not run on evicted pairs.
                        Useful during batch ingestion where summary updates
                        are not needed and the cost is not justified.
        """
        bucket = self._bucket

        # 1. Set prompt.
        bucket.set_current_prompt(user_message)

        # 2. Inject memories.
        self._retrieval.update_bucket(bucket, user_message)

        # 3. Respond (user waits for this).
        response = self._agent.respond(bucket)

        # 4. Advance the recent stack.
        popped = bucket.push_message(user_message, response)

        # 5. Background memory work — only when a pair was evicted.
        if popped is not None:
            if not skip_observer:
                threading.Thread(
                    target=self._observer.update,
                    args=(bucket, popped),
                    daemon=True,
                ).start()

            threading.Thread(
                target=self._curator.evaluate,
                args=(popped,),
                daemon=True,
            ).start()

        # 6. Return response.
        return response

    def ingest(self, question: str, response: str) -> None:
        """
        Push a pre-formed Q/A pair into the Bucket without invoking the
        Active Agent.  Intended for batch pre-seeding from a known
        conversation log.

        The Observer and Curator still run on eviction so the rolling
        summary and vector store are built correctly.

        Parameters
        ----------
        question   The user turn of the pair.
        response   The assistant turn of the pair.
        """
        popped = self._bucket.push_message(question, response)

        if popped is not None:
            # Observer runs synchronously so the rolling summary is fully built
            # before the next pair is ingested.
            self._observer.update(self._bucket, popped)

            # Curator can remain async -- its Chroma writes do not affect the
            # summary and it has ample time to finish during Phase 3 recall.
            threading.Thread(
                target=self._curator.evaluate,
                args=(popped,),
                daemon=True,
            ).start()

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def bucket(self) -> Bucket:
        """The shared Bucket — inspect after a conversation to see final state."""
        return self._bucket

    @property
    def retrieval(self) -> Retrieval:
        """The Retrieval instance — query directly to inspect stored memories."""
        return self._retrieval

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Pipeline("
            f"model={self._model!r}, "
            f"bucket={self._bucket!r}, "
            f"stored={self._retrieval._collection.count()})"
        )
