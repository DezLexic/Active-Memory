"""
curator.py

Evaluates the stable mid-stack conversation pair (the one near the eviction
boundary) and assigns it to the correct memory tier.  This runs after
Pipeline has already auto-stored the full evicted batch to cold storage.

The Curator's job is NOT to decide whether to store — Pipeline already
handles that.  It decides whether the peeked pair deserves *warm* priority
(frequent access, surfaced before cold results) vs *cold* (background
reference, surfaced only when directly relevant).

Warm = explicit decision, hard constraint, or active preference the agent
will likely need soon.  Cold = everything else.

The model returns:

    {"store": true, "reason": "<one sentence>", "tier": "warm"|"cold"}

"store" is always true in the new framing; on parse failure the pair is
stored as cold (the safe default) rather than dropped silently.
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
    Agent that assigns the mid-stack conversation pair to warm or cold tier.

    Pipeline auto-stores the full evicted batch to cold storage before
    calling Curator.  Curator's role is warm promotion: it decides whether
    the peeked mid-stack pair deserves warm priority over the cold entries
    that were already written.

    Parameters
    ----------
    backend         Any LLMBackend-conforming object.
    retrieval       A Retrieval instance that owns the Chroma collection.
    use_batch_mode  When True, Pipeline passes the single pair returned by
                    bucket.peek_curator_target() for warm-promotion evaluation.
                    When False, Pipeline passes every evicted pair (legacy
                    behaviour).  Default True.
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
        Evaluate a Q/A pair and store it to warm or cold in Chroma.

        Steps
        -----
        1. Build a triage prompt from the pair.
        2. Make one LLM call to decide warm vs cold tier.
        3. Parse the JSON response.
        4. Store to the appropriate tier via retrieval.store().
           On parse failure: defaults to cold (safe fallback, never drops).

        Parameters
        ----------
        popped_pair  Dict with keys "question" and "response".
        """
        question = popped_pair.get("question", "").strip()
        response = popped_pair.get("response", "").strip()
        combined = f"{question} {response}"

        prompt = (
            "You are assigning a conversation exchange to the right memory "
            "tier.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"RESPONSE:\n{response}\n\n"
            "This exchange WILL be stored. Your only job is to decide the "
            "tier.\n\n"
            "WARM — use when the exchange contains:\n"
            "  • An explicit decision or commitment (\"we will\", \"agreed\", "
            "\"decided\")\n"
            "  • A hard constraint or requirement (\"must\", \"never\", "
            "\"required\", \"non-negotiable\")\n"
            "  • An active preference the assistant will likely need in a "
            "future turn\n\n"
            "COLD — use for everything else:\n"
            "  • Personal facts, stories, feelings, emotions, opinions\n"
            "  • Background information, descriptions, casual context\n"
            "  • When in doubt, cold.\n\n"
            "Respond with a JSON object containing exactly three fields:\n"
            "  store  : true\n"
            "  reason : one sentence\n"
            "  tier   : \"warm\" or \"cold\"\n\n"
            "Respond with only the JSON object. No preamble, no commentary."
        )

        try:
            with ProcessMonitor("curator evaluating pair"):
                raw_text = self._backend.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            logger.error(
                "Curator: LLM call failed (%s); storing as cold.", exc
            )
            # Safe fallback: cold rather than drop.
            if self._retrieval is not None:
                self._retrieval.store(combined, float(time.time()), tier="cold")
            self._last_store  = True
            self._last_reason = f"(LLM failure — stored as cold: {exc})"
            self._last_tier   = "cold"
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

            store  = bool(parsed.get("store", True))   # default True — always store
            reason = str(parsed.get("reason", ""))
            tier   = str(parsed.get("tier", "cold")).lower().strip()
            if tier not in ("warm", "cold"):
                tier = "cold"
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning(
                "Curator: JSON parse error (%s); raw=%r; storing as cold.",
                exc, raw_text,
            )
            # Safe fallback: cold rather than drop.
            if self._retrieval is not None:
                self._retrieval.store(combined, float(time.time()), tier="cold")
            self._last_store  = True
            self._last_reason = "(JSON parse failure — stored as cold)"
            self._last_tier   = "cold"
            return

        if store and self._retrieval is not None:
            timestamp = float(time.time())
            self._retrieval.store(combined, timestamp, tier=tier)

        # Surface the decision for callers / test scripts that want visibility.
        self._last_store  = store
        self._last_reason = reason
        self._last_tier   = tier

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Curator(backend={self._backend!r})"
