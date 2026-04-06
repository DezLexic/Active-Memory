"""
bucket.py

The Bucket is the shared context window for Active Memory.  It is the single
object the Active Agent reads from when composing a response.  Every other
component in the pipeline writes into it; the Agent only reads from it.

No model calls.  No external dependencies.  Standard library only.
"""

from __future__ import annotations

_DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant.\n"
    "Retrieved memories shown below are approximate reconstructions drawn from "
    "earlier conversation. They may be incomplete or imprecise. Use them as "
    "supporting context but do not treat them as verbatim ground truth.\n"
    "When the conversation summary and retrieved memories conflict, prefer the "
    "summary as it is more recent."
)


class Bucket:
    """
    Shared context window passed to the Active Agent on every turn.

    Slots
    -----
    summary             Rolling summary of the conversation so far (set by Observer).
    recent_messages     Fixed-size stack of the most recent Q/A pairs.
    memories            Up to 3 retrieved memory strings (set by retrieval layer).
    system_instructions Static instructions prepended to every context string.
    current_prompt      The user message currently being answered.
    """

    def __init__(
        self,
        max_recent: int = 20,
        batch_reduction: int = 10,
        system_instructions: str = _DEFAULT_SYSTEM_INSTRUCTIONS,
    ) -> None:
        self._max_recent: int          = max_recent
        self._batch_reduction: int     = batch_reduction
        self._max_memories: int        = 3
        self.system_instructions: str  = system_instructions
        self.summary: str              = ""
        self.recent_messages: list[dict[str, str]] = []
        self.memories: list[str]       = []
        self.current_prompt: str       = ""

    # ── Mutators ──────────────────────────────────────────────────────────────

    def push_message(
        self, question: str, response: str
    ) -> list[dict[str, str]] | None:
        """
        Append a completed Q/A pair to recent_messages.

        When the stack reaches max capacity, batch_reduction pairs are evicted
        from the front all at once and returned as a list.  The new pair is
        then appended.  Returns None when the stack was not yet full and no
        eviction was necessary.
        """
        evicted: list[dict[str, str]] | None = None
        if len(self.recent_messages) >= self._max_recent:
            evicted = self.recent_messages[: self._batch_reduction]
            self.recent_messages = self.recent_messages[self._batch_reduction :]
        self.recent_messages.append({"question": question, "response": response})
        return evicted

    def peek_curator_target(self) -> dict[str, str] | None:
        """
        Return the pair at index (max_recent - batch_reduction) in the recent
        stack without removing it.

        This is the pair old enough to have surrounding context but not yet
        part of an eviction batch — a stable, mid-stack candidate for Curator
        evaluation.  Returns None if the stack has not reached that depth yet.
        """
        target_idx = self._max_recent - self._batch_reduction
        if len(self.recent_messages) > target_idx:
            return self.recent_messages[target_idx]
        return None

    def set_summary(self, summary: str) -> None:
        """Replace the conversation summary (called by the Observer)."""
        self.summary = summary

    def set_memories(self, memories: list[str]) -> None:
        """
        Replace the memories slot with a fresh retrieval result.
        Silently truncates to the maximum of 3 if the list is longer.
        """
        self.memories = memories[: self._max_memories]

    def set_current_prompt(self, prompt: str) -> None:
        """Set the user message that is about to be answered."""
        self.current_prompt = prompt

    # ── Context assembly ──────────────────────────────────────────────────────

    def to_context_string(self) -> str:
        """
        Assemble all slots into a single formatted string ready to be passed
        as context to the Active Agent.
        """
        divider = "-" * 60
        sections: list[str] = []

        # ── SYSTEM INSTRUCTIONS ───────────────────────────────────────────────
        sections.append(
            f"{'=' * 60}\n"
            f"SYSTEM INSTRUCTIONS\n"
            f"{divider}\n"
            f"{self.system_instructions.strip()}"
        )

        # ── CONVERSATION SUMMARY ──────────────────────────────────────────────
        summary_text = self.summary.strip() if self.summary.strip() else "(no summary yet)"
        sections.append(
            f"{'=' * 60}\n"
            f"CONVERSATION SUMMARY\n"
            f"{divider}\n"
            f"{summary_text}"
        )

        # ── RECENT MESSAGES ───────────────────────────────────────────────────
        if self.recent_messages:
            pairs: list[str] = []
            for i, pair in enumerate(self.recent_messages, 1):
                pairs.append(
                    f"[{i}]\n"
                    f"  Q: {pair['question'].strip()}\n"
                    f"  A: {pair['response'].strip()}"
                )
            recent_text = "\n\n".join(pairs)
        else:
            recent_text = "(no messages yet)"

        sections.append(
            f"{'=' * 60}\n"
            f"RECENT MESSAGES  ({len(self.recent_messages)} / {self._max_recent})\n"
            f"{divider}\n"
            f"{recent_text}"
        )

        # ── RELEVANT MEMORIES ─────────────────────────────────────────────────
        if self.memories:
            mem_lines: list[str] = []
            for i, mem in enumerate(self.memories, 1):
                mem_lines.append(f"[{i}] {mem.strip()}")
            memories_text = "\n\n".join(mem_lines)
        else:
            memories_text = "(none retrieved)"

        sections.append(
            f"{'=' * 60}\n"
            f"RELEVANT MEMORIES  ({len(self.memories)} / {self._max_memories})\n"
            f"{divider}\n"
            f"{memories_text}"
        )

        # ── CURRENT PROMPT ────────────────────────────────────────────────────
        prompt_text = self.current_prompt.strip() if self.current_prompt.strip() else "(not set)"
        sections.append(
            f"{'=' * 60}\n"
            f"CURRENT PROMPT\n"
            f"{divider}\n"
            f"{prompt_text}"
        )

        return "\n\n".join(sections)

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Bucket("
            f"recent={len(self.recent_messages)}/{self._max_recent}, "
            f"batch_reduction={self._batch_reduction}, "
            f"memories={len(self.memories)}/{self._max_memories}, "
            f"summary={'set' if self.summary else 'empty'}, "
            f"prompt={'set' if self.current_prompt else 'empty'})"
        )
