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
5a. If a batch was evicted -> Observer receives the full batch list (one LLM call)
5b. If a batch was evicted -> Curator evaluates the pair returned by
    bucket.peek_curator_target() (one LLM call for a mid-stack stable pair)
6. Return response string

Both Observer and Curator calls are sequential (no daemon threads) to avoid
Ollama queue contention on a single-instance setup.

Observer receives the entire evicted batch and makes exactly one LLM call
regardless of batch size.  Curator evaluates one stable mid-stack pair via
peek_curator_target() rather than every evicted pair.

Backend configuration
---------------------
All three agents (ActiveAgent, Observer, Curator) share the same backend by
default.  Pass observer_backend or curator_backend to route different roles
to different providers or models — for example, a heavyweight cloud model for
user-facing responses and a lightweight local model for bookkeeping tasks.
"""

from __future__ import annotations

from .bucket        import Bucket
from .retrieval     import Retrieval
from .observer      import Observer
from .curator       import Curator
from .active_agent  import ActiveAgent
from .backends.base import LLMBackend


class Pipeline:
    """
    Wires Bucket, Retrieval, Observer, Curator, and ActiveAgent together.

    Parameters
    ----------
    backend             LLM backend used by all three agents unless overrides
                        are provided.  When None, an OllamaBackend with
                        model "gemma3:4b" and default base URL is created
                        automatically — preserving the zero-config experience
                        for Ollama users.
    chroma_path         Path to the local Chroma database directory.
    max_recent_messages Max Q/A pairs kept in the Bucket's recent stack.
    batch_reduction     Number of pairs evicted at once when the stack is
                        full.  Must be <= max_recent_messages.
    system_instructions Optional override for the Bucket's system prompt
                        header.  Pass None to use the Bucket default.
    observer_backend    If provided, used exclusively by the Observer instead
                        of `backend`.  Useful when summarisation should run
                        on a different model or provider.
    curator_backend     If provided, used exclusively by the Curator instead
                        of `backend`.
    """

    def __init__(
        self,
        backend: LLMBackend | None = None,
        chroma_path: str = "./chroma_db",
        max_recent_messages: int = 20,
        batch_reduction: int = 10,
        system_instructions: str | None = None,
        observer_backend: LLMBackend | None = None,
        curator_backend:  LLMBackend | None = None,
    ) -> None:
        # Default to backend_from_env() so Pipeline() with no args reads .env.
        # Falls back to OllamaBackend("gemma3:4b") when no .env is present.
        if backend is None:
            from .config import backend_from_env
            backend = backend_from_env()

        _observer_backend = observer_backend if observer_backend is not None else backend
        _curator_backend  = curator_backend  if curator_backend  is not None else backend

        _CONCISE = "Be concise. Answer directly. Do not over-explain."

        if system_instructions is not None:
            combined_instructions = f"{system_instructions}\n{_CONCISE}"
        else:
            from .bucket import _DEFAULT_SYSTEM_INSTRUCTIONS
            combined_instructions = f"{_DEFAULT_SYSTEM_INSTRUCTIONS}\n{_CONCISE}"

        self._bucket    = Bucket(
            max_recent=max_recent_messages,
            batch_reduction=batch_reduction,
            system_instructions=combined_instructions,
        )
        self._retrieval = Retrieval(chroma_path=chroma_path)
        self._observer  = Observer(backend=_observer_backend)
        self._curator   = Curator(backend=_curator_backend, retrieval=self._retrieval)
        self._agent     = ActiveAgent(backend=backend)
        self._backend   = backend   # kept for __repr__

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
           any evicted batch.
        5. If a batch was evicted:
           - Run Observer with the full batch list (one LLM call) unless
             skip_observer is True.
           - If use_batch_mode is True, call peek_curator_target() and pass
             the result to Curator (one LLM call for the mid-stack pair).
             Otherwise fall back to evaluating each evicted pair in sequence.
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

        # 5. Sequential memory work — only when a batch was evicted.
        if popped is not None:
            if not skip_observer:
                self._observer.update(bucket, popped)

            if self._curator.use_batch_mode:
                peeked = bucket.peek_curator_target()
                if peeked is not None:
                    self._curator.evaluate(peeked)
            else:
                for pair in popped:
                    self._curator.evaluate(pair)

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
            # Observer runs synchronously so the rolling summary is fully
            # built before the next pair is ingested.
            self._observer.update(self._bucket, popped)

            # Curator evaluates the stable mid-stack pair rather than all
            # evicted pairs.
            if self._curator.use_batch_mode:
                peeked = self._bucket.peek_curator_target()
                if peeked is not None:
                    self._curator.evaluate(peeked)
            else:
                for pair in popped:
                    self._curator.evaluate(pair)

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
            f"backend={self._backend!r}, "
            f"bucket={self._bucket!r}, "
            f"stored={self._retrieval._collection.count()})"
        )
