"""
curator.py

Classifies an evicted conversation pair into the correct memory tier.

The Curator no longer writes to storage directly. Its only job is to decide
whether a pair should end up in warm or cold storage, with cold as the safe
fallback on any error.
"""

from __future__ import annotations

import json
import logging
from .backends.base import LLMBackend
from .monitor       import ProcessMonitor

logger = logging.getLogger(__name__)


class Curator:
    """
    Agent that classifies conversation pairs as warm or cold.

    Parameters
    ----------
    backend         Any LLMBackend-conforming object.
    use_batch_mode  Retained for backward compatibility with older pipeline
                    flows. The current pipeline evaluates each evicted pair.
    """

    def __init__(
        self,
        backend: LLMBackend,
        retrieval=None,
        use_batch_mode: bool = True,
    ) -> None:
        self._backend      = backend
        self._retrieval    = retrieval
        self.use_batch_mode = use_batch_mode   # public — Pipeline reads it

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, popped_pair: dict[str, str]) -> str:
        """
        Evaluate a Q/A pair and return the chosen storage tier.

        Steps
        -----
        1. Build a triage prompt from the pair.
        2. Make one LLM call to decide warm vs cold tier.
        3. Parse the JSON response.
        4. Return the selected tier.
           On parse failure: defaults to cold (safe fallback).

        Parameters
        ----------
        popped_pair  Dict with keys "question" and "response".
        """
        question = popped_pair.get("question", "").strip()
        response = popped_pair.get("response", "").strip()

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
            "Respond with a JSON object containing exactly two fields:\n"
            "  reason : one sentence\n"
            "  tier   : \"warm\" or \"cold\"\n\n"
            "Respond with only the JSON object. No preamble, no commentary."
        )

        try:
            with ProcessMonitor("curator evaluating pair"):
                raw_text = self._backend.chat([{"role": "user", "content": prompt}])
        except Exception as exc:
            logger.error(
                "Curator: LLM call failed (%s); defaulting to cold.", exc
            )
            self._last_store  = True
            self._last_reason = f"(LLM failure — defaulted to cold: {exc})"
            self._last_tier   = "cold"
            return "cold"

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

            reason = str(parsed.get("reason", ""))
            tier   = str(parsed.get("tier", "cold")).lower().strip()
            if tier not in ("warm", "cold"):
                tier = "cold"
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning(
                "Curator: JSON parse error (%s); raw=%r; defaulting to cold.",
                exc, raw_text,
            )
            self._last_store  = True
            self._last_reason = "(JSON parse failure — defaulted to cold)"
            self._last_tier   = "cold"
            return "cold"

        # Surface the decision for callers / test scripts that want visibility.
        self._last_store  = True
        self._last_reason = reason
        self._last_tier   = tier
        return tier

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Curator(backend={self._backend!r})"
