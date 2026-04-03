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
        max_recent: int = 5,
        system_instructions: str = _DEFAULT_SYSTEM_INSTRUCTIONS,
    ) -> None:
        self._max_recent: int          = max_recent
        self._max_memories: int        = 3
        self.system_instructions: str  = system_instructions
        self.summary: str              = ""
        self.recent_messages: list[dict[str, str]] = []
        self.memories: list[str]       = []
        self.current_prompt: str       = ""

    # ── Mutators ──────────────────────────────────────────────────────────────

    def push_message(self, question: str, response: str) -> dict[str, str] | None:
        """
        Append a completed Q/A pair to recent_messages.

        If the stack is already at max capacity the oldest pair is evicted
        before the new one is appended.  The evicted pair is returned so the
        caller can forward it to the Observer and Curator for memory
        consideration.  Returns None when no eviction was necessary.
        """
        evicted: dict[str, str] | None = None
        if len(self.recent_messages) >= self._max_recent:
            evicted = self.recent_messages.pop(0)
        self.recent_messages.append({"question": question, "response": response})
        return evicted

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
            f"memories={len(self.memories)}/{self._max_memories}, "
            f"summary={'set' if self.summary else 'empty'}, "
            f"prompt={'set' if self.current_prompt else 'empty'})"
        )
