"""
curator.py

Evaluates an evicted message pair and stores it in Chroma if it is worth
remembering.  Runs in the background after the Active Agent responds --
the user never waits for it.

Every popped pair is evaluated regardless of length.  Short exchanges can
still contain important decisions -- word count is not a signal of importance.

An optional conversation summary can be passed to evaluate() as context.
When provided, the model sees the broader conversation before judging the
pair, which improves accuracy on short or ambiguous exchanges.

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

    def evaluate(
        self,
        popped_pair: dict[str, str],
        summary: str | None = None,
    ) -> None:
        """
        Evaluate a popped Q/A pair and store it in Chroma if worthwhile.

        Steps
        -----
        1. Build a prompt, optionally prepending the current conversation
           summary as context.
        2. Make one Ollama call to judge whether the pair contains a
           decision, preference, constraint, or direction.
        3. Parse the JSON response.
        4. If store is true, write to Chroma via retrieval.store().

        Parameters
        ----------
        popped_pair  Dict with keys "question" and "response".
        summary      Optional current conversation summary.  When provided
                     it is included in the prompt before the pair so the
                     model has enough context to judge short or ambiguous
                     exchanges accurately.
        """
        question = popped_pair.get("question", "").strip()
        response = popped_pair.get("response", "").strip()
        combined = f"{question} {response}"

        # Build the context block only when a summary is supplied.
        if summary and summary.strip():
            context_block = (
                "Here is the current conversation summary for context:\n"
                f"{summary.strip()}\n\n"
                "Now evaluate this exchange:\n"
            )
        else:
            context_block = ""

        prompt = (
            "You are evaluating a conversation message pair to decide whether "
            "it is worth storing as a long-term memory.\n"
            "Short exchanges can still contain important decisions -- word count "
            "is not a signal of importance.\n\n"
            f"{context_block}"
            f"QUESTION:\n{question}\n\n"
            f"RESPONSE:\n{response}\n\n"
            "Evaluate whether this pair contains any of the following:\n"
            "- a decision made\n"
            "- a preference established\n"
            "- a constraint agreed on\n"
            "- a direction chosen\n\n"
            "Respond with a JSON object containing exactly two fields:\n"
            "  store  : boolean (true if the pair is worth storing, false otherwise)\n"
            "  reason : one sentence explaining your decision\n\n"
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
