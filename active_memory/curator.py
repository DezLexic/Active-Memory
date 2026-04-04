"""
curator.py

Evaluates an evicted message pair and stores it in Chroma if it is worth
remembering.  Runs in the background after the Active Agent responds --
the user never waits for it.

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
import time
import ollama

from .retrieval import Retrieval


class Curator:
    """
    Background agent that decides whether an evicted Q/A pair is worth
    persisting to the vector store.

    Parameters
    ----------
    model       Ollama model name used for evaluation.
    retrieval   A Retrieval instance that owns the Chroma collection.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        retrieval: Retrieval | None = None,
    ) -> None:
        self._model     = model
        self._retrieval = retrieval

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, popped_pair: dict[str, str]) -> None:
        """
        Evaluate a popped Q/A pair and store it in Chroma if worthwhile.

        Steps
        -----
        1. Build a prompt from the pair alone.
        2. Make one Ollama call to judge whether the pair contains an
           explicit decision, hard constraint, or established preference.
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
            "Respond with a JSON object containing exactly two fields:\n"
            "  store  : boolean\n"
            "  reason : one sentence\n\n"
            "Respond with only the JSON object. No preamble, no commentary."
        )

        try:
            result   = ollama.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = result["message"]["content"].strip()
        except Exception as exc:
            print(f"[Curator] Ollama call failed: {exc}")
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
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            print(f"[Curator] JSON parse error: {exc}  raw={raw_text!r}")
            return

        if store and self._retrieval is not None:
            timestamp = float(time.time())
            self._retrieval.store(combined, timestamp)

        # Surface the decision for callers / test scripts that want visibility.
        self._last_store  = store
        self._last_reason = reason

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Curator(model={self._model!r})"
