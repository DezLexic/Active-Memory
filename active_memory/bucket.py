"""
bucket.py

The Bucket is the shared context window for Active Memory.  It is the single
object the Active Agent reads from when composing a response.  Every other
component in the pipeline writes into it; the Agent only reads from it.

No model calls.  No external dependencies.  Standard library only.
"""

from __future__ import annotations

import time

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
    memories            Retrieved memory dicts with metadata (set by retrieval layer).
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
        self.system_instructions: str  = system_instructions
        self.topic_tree: dict          = {"topics": []}
        self._turn_count: int          = 0
        self.recent_messages: list[dict[str, str]] = []
        self.memories: list[dict]      = []
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
        self._turn_count += 1
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

    def get_summary_text(self) -> str:
        """Flatten the topic tree into indented readable prose with staleness."""
        if not self.topic_tree.get("topics"):
            return ""
        lines: list[str] = []
        self._flatten_topics(self.topic_tree["topics"], lines, depth=0)
        return "\n\n".join(lines)

    # Slot name -> rendered label, in the order they appear in flattened output.
    _SLOT_LABELS = (
        ("facts",        "Facts"),
        ("decisions",    "Decisions"),
        ("preferences",  "Preferences"),
        ("open_threads", "Open threads"),
        ("quotes",       "Quotes"),
    )

    def _flatten_topics(self, topics: list[dict], lines: list[str], depth: int) -> None:
        indent = "  " * depth
        label = "Subtopic" if depth > 0 else "Topic"
        for t in topics:
            turns_ago = self._turn_count - t.get("updated_at_turn", 0)
            ago_text = f"{turns_ago} turn{'s' if turns_ago != 1 else ''} ago"
            header = f"{indent}[{label}: {t['title']} — updated {ago_text}]"

            body_lines: list[str] = []
            for slot, slot_label in self._SLOT_LABELS:
                items = t.get(slot) or []
                if not items:
                    continue
                body_lines.append(f"{indent}  {slot_label}:")
                for item in items:
                    body_lines.append(f"{indent}    • {item}")

            if body_lines:
                lines.append(header + "\n" + "\n".join(body_lines))
            else:
                lines.append(header)

            if t.get("subtopics"):
                self._flatten_topics(t["subtopics"], lines, depth + 1)

    @property
    def summary(self) -> str:
        """Deprecated — returns flattened topic tree text for backward compat."""
        return self.get_summary_text()

    @summary.setter
    def summary(self, value: str) -> None:
        """Deprecated setter — wraps string into a single-topic node.

        The incoming prose is stored as a single entry in the ``facts`` slot
        so that legacy callers still get their text rendered under the new
        typed-slot schema.
        """
        if value and value.strip():
            now = int(time.time())
            self.topic_tree = {
                "topics": [{
                    "id": "legacy_summary",
                    "title": "Conversation summary",
                    "facts":        [value.strip()],
                    "decisions":    [],
                    "preferences":  [],
                    "open_threads": [],
                    "quotes":       [],
                    "subtopics":    [],
                    "created_at":   now,
                    "updated_at":   now,
                    "updated_at_turn": self._turn_count,
                }]
            }

    def set_summary(self, summary: str) -> None:
        """Deprecated. Wraps the string into a single-topic tree node."""
        self.summary = summary

    def set_memories(self, memories: list[dict]) -> None:
        """
        Replace the memories slot with a fresh retrieval result.

        Each dict should have: content, similarity, tier, retrieval_count.
        """
        self.memories = memories

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
        summary_text = self.get_summary_text().strip() or "(no summary yet)"
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

        # ── RETRIEVED MEMORIES ────────────────────────────────────────────────
        if self.memories:
            mem_lines: list[str] = []
            for mem in self.memories:
                sim = mem.get("similarity", 0.0)
                tier = mem.get("tier", "unknown")
                content = mem.get("content", "").strip()
                mem_lines.append(f"[relevance: {sim:.2f} | {tier}] {content}")
            memories_text = "\n\n".join(mem_lines)
        else:
            memories_text = "(none retrieved)"

        sections.append(
            f"{'=' * 60}\n"
            f"RETRIEVED MEMORIES  ({len(self.memories)} results)\n"
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
            f"memories={len(self.memories)}, "
            f"topics={len(self.topic_tree.get('topics', []))}, "
            f"prompt={'set' if self.current_prompt else 'empty'})"
        )
