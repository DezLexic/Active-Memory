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
    max_summary_length  Deprecated. Retained for backward-compat; has no effect
                        since topic nodes now use typed slots (facts, decisions,
                        preferences, open_threads) rather than a single prose
                        summary. A 200-char soft limit per slot item is baked
                        into the prompt.
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
            "You are maintaining a structured topic tree that captures concrete detail "
            "from a conversation so the information survives after messages scroll out "
            "of recent history.\n\n"
            "CURRENT TOPIC TREE (JSON):\n"
            f"{current_tree_json}\n\n"
            "MESSAGE PAIRS BEING REMOVED FROM RECENT HISTORY:\n"
            f"{pairs_text}\n\n"
            "NODE SHAPE:\n"
            "Each topic node must have this exact shape:\n"
            '  {"id": "<slug>", "title": "<short label>",\n'
            '   "facts": [...], "decisions": [...], "preferences": [...],\n'
            '   "open_threads": [...],\n'
            '   "subtopics": [...],\n'
            '   "created_at": <unix_ts>, "updated_at": <unix_ts>,\n'
            '   "updated_at_turn": <int>}\n\n'
            "SLOT DEFINITIONS — fill each with concrete items drawn from the evicted "
            "messages. Each slot is a list of strings. Leave a slot as [] when nothing "
            "applies. Keep each item under 200 characters.\n\n"
            "  facts:        Concrete statements preserving names, places, objects, "
            "dates, numbers, or specific events. A fact MUST contain at least one "
            "concrete noun, proper name, place, or number. Category labels like "
            "'reflection traditions' or 'memories of her mother' are NOT facts — "
            "they are topic titles and belong in the 'title' field, not here.\n"
            "                Example: 'Jolene and her mother shared Saturday coffee on "
            "the wooden bench in the backyard under the maple tree.'\n\n"
            "  decisions:    Explicit commitments, choices, or plans the speakers made.\n"
            "                Example: 'Jolene will restore the bench this summer.'\n\n"
            "  preferences:  Stated likes, dislikes, values, or communication style.\n"
            "                Example: 'Jolene prefers quiet mornings when discussing "
            "her mom.'\n\n"
            "  open_threads: Things raised but unresolved — the Active Agent should "
            "revisit these.\n"
            "                Example: 'Whether to keep or sell the family house.'\n\n"
            "RULES:\n"
            "1. Merge new information into existing topics — do not create duplicate "
            "topics that cover the same subject.\n"
            "2. Use subtopics for specific sub-decisions or distinct threads under a "
            "broader topic. Subtopic nodes have the same shape.\n"
            f"3. The current turn count is {bucket._turn_count}. Set updated_at_turn "
            "to this value for any node you create or modify.\n"
            "4. Never delete existing topics or drop existing items from slots — only "
            "update, extend, or add.\n"
            "5. When extending an existing slot, append new items to the existing list.\n"
            "6. Return ONLY the updated JSON object with a top-level \"topics\" array. "
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
