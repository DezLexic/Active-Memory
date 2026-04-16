"""
curator.py

Evaluates an evicted message pair and stores it in Chroma if it is worth
remembering.  Runs after the Active Agent responds -- the user never waits
for it.

The Curator receives only the popped pair -- no summary, no external context.
The prompt is direct: does this exchange contain an explicit decision, hard
constraint, or established preference?  If yes, store it.  If the content is
ambiguous or seems incomplete on its own, do not store it.  Be conservative --
when in doubt, drop it.

The model is asked to return a JSON object:

    {"store": true/false, "reason": "<one sentence>"}

If store is true the pair is written to the Retrieval instance using the
current UTC timestamp.  Malformed JSON is caught and logged; on parse
failure the pair is not stored.
"""

from __future__ import annotations

import json
import logging
import time

from .retrieval     import Retrieval
from .backends.base import LLMBackend
from .monitor       import ProcessMonitor

logger = logging.getLogger(__name__)


class Curator:
    """
    Agent that decides whether an evicted Q/A pair is worth persisting to
    the vector store.

    Parameters
    ----------
    backend         Any LLMBackend-conforming object.
    retrieval       A Retrieval instance that owns the Chroma collection.
    use_batch_mode  When True, Pipeline passes the single pair returned by
                    bucket.peek_curator_target() rather than every evicted
                    pair.  Default True.
    """

    def __init__(
        self,
        backend: LLMBackend,
        retrieval: Retrieval | None = None,
        use_batch_mode: bool = True,
    ) -> None:
        self._backend      = backend
        self._retrieval    = retrieval
        self.use_batch_mode = use_batch_mode   # public — Pipeline reads it

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, popped_pair: dict[str, str]) -> None:
        """
        Evaluate a popped Q/A pair and store it in Chroma if worthwhile.

        Steps
        -----
        1. Build a prompt from the pair alone.
        2. Make one LLM call to judge whether the pair contains an explicit
           decision, hard constraint, or established preference.
        3. Parse the JSON response.
        4. If store is true, write to Chroma via retrieval.store().

        Parameters
        ----------
        popped_pair  Dict with keys "question" and "response".
        """
        question = popped_pair.get("question", "").strip()
        response = popped_pair.get("response", "").strip()
        combined = f"{question} {response}"

        prompt = (
            "You are deciding whether a conversation exchange is worth storing "
            "as a long-term memory.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"RESPONSE:\n{response}\n\n"
            "Store it only if this exchange, on its own, contains an explicit "
            "decision, a hard constraint, or an established preference.\n"
            "If the content is ambiguous or seems incomplete without additional "
            "context, do not store it. Be conservative -- when in doubt, drop it.\n\n"
            "Also decide which tier this memory belongs in.\n"
            "Warm is for recent decisions, active constraints, and preferences "
            "likely needed soon.\n"
            "Cold is for older context, background information, and decisions "
            "unlikely to be revisited soon.\n\n"
            "Respond with a JSON object containing exactly three fields:\n"
            "  store  : boolean\n"
            "  reason : one sentence\n"
            "  tier   : \"warm\" or \"cold\"\n\n"
            "Respond with only the JSON object. No preamble, no commentary."
        )

        try:
            with ProcessMonitor("curator evaluating pair"):
                raw_text = self._backend.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            logger.error("Curator: LLM call failed (%s); pair not stored.", exc)
            return

        try:
            # Strip markdown code fences if the model wraps its response.
            if raw_text.startswith("```"):
                lines    = raw_text.splitlines()
                raw_text = "\n".join(
                    line for line in lines
                    if not line.startswith("```")
                )
            parsed = json.loads(raw_text)

            # Handle the case where the model returns a list instead of an object.
            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else {}

            store  = bool(parsed.get("store", False))
            reason = str(parsed.get("reason", ""))
            tier   = str(parsed.get("tier", "warm")).lower().strip()
            if tier not in ("warm", "cold"):
                tier = "warm"
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("Curator: JSON parse error (%s); raw=%r; pair not stored.", exc, raw_text)
            return

        if store and self._retrieval is not None:
            timestamp = float(time.time())
            self._retrieval.store(combined, timestamp, tier=tier)

        # Surface the decision for callers / test scripts that want visibility.
        self._last_store  = store
        self._last_reason = reason
        self._last_tier   = tier if store else None

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Curator(backend={self._backend!r})"
