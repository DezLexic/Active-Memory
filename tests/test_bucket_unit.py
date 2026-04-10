"""
test_bucket_unit.py

Unit tests for the Bucket class covering edge cases, empty-slot fallbacks,
and boundary configurations not exercised by the existing test_bucket.py
integration script.

No external dependencies — standard library only.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from active_memory.bucket import Bucket, _DEFAULT_SYSTEM_INSTRUCTIONS


# ── push_message() ────────────────────────────────────────────────────────────

class TestPushMessage:

    def test_returns_none_when_not_full(self):
        b = Bucket(max_recent=5, batch_reduction=2)
        assert b.push_message("q", "a") is None

    def test_returns_list_on_eviction(self):
        b = Bucket(max_recent=3, batch_reduction=2)
        b.push_message("q0", "a0")
        b.push_message("q1", "a1")
        b.push_message("q2", "a2")
        evicted = b.push_message("q3", "a3")  # triggers eviction
        assert isinstance(evicted, list)

    def test_eviction_list_length_equals_batch_reduction(self):
        b = Bucket(max_recent=3, batch_reduction=2)
        for i in range(3):
            b.push_message(f"q{i}", f"a{i}")
        evicted = b.push_message("q3", "a3")
        assert len(evicted) == 2

    def test_evicted_items_are_oldest_pairs(self):
        b = Bucket(max_recent=3, batch_reduction=2)
        for i in range(3):
            b.push_message(f"q{i}", f"a{i}")
        evicted = b.push_message("q3", "a3")
        assert evicted[0]["question"] == "q0"
        assert evicted[1]["question"] == "q1"

    def test_stack_shrinks_after_eviction(self):
        b = Bucket(max_recent=3, batch_reduction=2)
        for i in range(3):
            b.push_message(f"q{i}", f"a{i}")
        b.push_message("q3", "a3")
        # stack was [0,1,2], evict [0,1], push 3 → [2,3]
        assert len(b.recent_messages) == 2

    def test_pair_structure_is_dict_with_question_and_response(self):
        b = Bucket(max_recent=5, batch_reduction=2)
        b.push_message("What is X?", "X is Y.")
        pair = b.recent_messages[0]
        assert pair["question"] == "What is X?"
        assert pair["response"] == "X is Y."

    def test_batch_reduction_one_evicts_one_pair(self):
        b = Bucket(max_recent=2, batch_reduction=1)
        b.push_message("q0", "a0")
        b.push_message("q1", "a1")
        evicted = b.push_message("q2", "a2")
        assert len(evicted) == 1
        assert evicted[0]["question"] == "q0"

    def test_max_recent_one_evicts_on_second_push(self):
        b = Bucket(max_recent=1, batch_reduction=1)
        assert b.push_message("q0", "a0") is None
        evicted = b.push_message("q1", "a1")
        assert evicted is not None
        assert len(evicted) == 1
        assert evicted[0]["question"] == "q0"

    def test_second_eviction_evicts_next_oldest(self):
        b = Bucket(max_recent=3, batch_reduction=2)
        for i in range(3):
            b.push_message(f"q{i}", f"a{i}")
        b.push_message("q3", "a3")        # first eviction  → stack [2,3]
        b.push_message("q4", "a4")        # stack [2,3,4]   → no eviction yet
        evicted2 = b.push_message("q5", "a5")  # second eviction → evict [2,3]
        assert evicted2[0]["question"] == "q2"
        assert evicted2[1]["question"] == "q3"


# ── peek_curator_target() ─────────────────────────────────────────────────────

class TestPeekCuratorTarget:

    def test_returns_none_when_stack_shallow(self):
        b = Bucket(max_recent=5, batch_reduction=2)
        b.push_message("q", "a")
        # target_idx = 5-2 = 3; stack has 1 pair → None
        assert b.peek_curator_target() is None

    def test_returns_none_on_empty_stack(self):
        b = Bucket(max_recent=5, batch_reduction=2)
        assert b.peek_curator_target() is None

    def test_returns_dict_at_target_index(self):
        b = Bucket(max_recent=3, batch_reduction=2)
        # target_idx = 3-2 = 1
        for i in range(4):
            b.push_message(f"q{i}", f"a{i}")
        # After eviction: stack=[q2,q3]; index 1 = q3
        result = b.peek_curator_target()
        assert result is not None
        assert "question" in result
        assert "response" in result

    def test_does_not_remove_item_from_stack(self):
        b = Bucket(max_recent=3, batch_reduction=2)
        for i in range(4):
            b.push_message(f"q{i}", f"a{i}")
        size_before = len(b.recent_messages)
        b.peek_curator_target()
        assert len(b.recent_messages) == size_before


# ── to_context_string() ───────────────────────────────────────────────────────

class TestToContextString:

    def test_empty_summary_shows_fallback(self):
        b = Bucket()
        ctx = b.to_context_string()
        assert "(no summary yet)" in ctx

    def test_empty_recent_messages_shows_fallback(self):
        b = Bucket()
        ctx = b.to_context_string()
        assert "(no messages yet)" in ctx

    def test_empty_memories_shows_fallback(self):
        b = Bucket()
        ctx = b.to_context_string()
        assert "(none retrieved)" in ctx

    def test_empty_prompt_shows_fallback(self):
        b = Bucket()
        ctx = b.to_context_string()
        assert "(not set)" in ctx

    def test_summary_appears_when_set(self):
        b = Bucket()
        b.set_summary("We decided on Elixir.")
        ctx = b.to_context_string()
        assert "We decided on Elixir." in ctx

    def test_memories_appear_when_set(self):
        b = Bucket()
        b.set_memories(["Memory A.", "Memory B."])
        ctx = b.to_context_string()
        assert "Memory A." in ctx
        assert "Memory B." in ctx

    def test_recent_messages_appear_when_set(self):
        b = Bucket()
        b.push_message("What DB?", "PostgreSQL.")
        ctx = b.to_context_string()
        assert "What DB?" in ctx
        assert "PostgreSQL." in ctx

    def test_current_prompt_appears_when_set(self):
        b = Bucket()
        b.set_current_prompt("User asks this.")
        ctx = b.to_context_string()
        assert "User asks this." in ctx

    def test_context_string_is_str(self):
        b = Bucket()
        assert isinstance(b.to_context_string(), str)

    def test_context_string_has_all_sections(self):
        b = Bucket()
        ctx = b.to_context_string()
        assert "SYSTEM INSTRUCTIONS" in ctx
        assert "CONVERSATION SUMMARY" in ctx
        assert "RECENT MESSAGES" in ctx
        assert "RELEVANT MEMORIES" in ctx
        assert "CURRENT PROMPT" in ctx


# ── set_memories() ────────────────────────────────────────────────────────────

class TestSetMemories:

    def test_truncates_to_max_three(self):
        b = Bucket()
        b.set_memories(["m1", "m2", "m3", "m4", "m5"])
        assert len(b.memories) == 3

    def test_fewer_than_max_are_kept_as_is(self):
        b = Bucket()
        b.set_memories(["m1", "m2"])
        assert len(b.memories) == 2

    def test_empty_list_accepted(self):
        b = Bucket()
        b.set_memories([])
        assert b.memories == []


# ── repr ──────────────────────────────────────────────────────────────────────

class TestRepr:

    def test_repr_contains_max_recent(self):
        b = Bucket(max_recent=7)
        assert "7" in repr(b)

    def test_repr_shows_empty_summary(self):
        b = Bucket()
        assert "empty" in repr(b)

    def test_repr_shows_set_summary(self):
        b = Bucket()
        b.set_summary("Something.")
        assert "set" in repr(b)
