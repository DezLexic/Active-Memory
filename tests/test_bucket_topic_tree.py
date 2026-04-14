"""
test_bucket_topic_tree.py

Unit tests for the Bucket's topic tree, _flatten_topics helper,
summary property (getter and setter), and _turn_count tracking.

No external dependencies -- standard library only.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from active_memory.bucket import Bucket


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_topic(title, summary="", updated_at_turn=0, subtopics=None):
    """Return a minimal topic dict matching the schema used by Observer."""
    return {
        "id": title.lower().replace(" ", "_"),
        "title": title,
        "summary": summary,
        "subtopics": subtopics or [],
        "created_at": 0,
        "updated_at": 0,
        "updated_at_turn": updated_at_turn,
    }


# ── TestTopicTreeInit ───────────────────────────────────────────────────────

class TestTopicTreeInit:

    def test_initial_topic_tree_is_empty(self):
        b = Bucket()
        assert b.topic_tree == {"topics": []}

    def test_initial_turn_count_is_zero(self):
        b = Bucket()
        assert b._turn_count == 0


# ── TestTurnCount ───────────────────────────────────────────────────────────

class TestTurnCount:

    def test_push_message_increments_turn_count(self):
        b = Bucket()
        b.push_message("q", "a")
        assert b._turn_count == 1

    def test_multiple_pushes_increment_correctly(self):
        b = Bucket()
        for i in range(5):
            b.push_message(f"q{i}", f"a{i}")
        assert b._turn_count == 5

    def test_turn_count_increments_on_eviction_too(self):
        b = Bucket(max_recent=3, batch_reduction=2)
        for i in range(3):
            b.push_message(f"q{i}", f"a{i}")
        # 4th push triggers eviction; count should still reflect all pushes
        evicted = b.push_message("q3", "a3")
        assert evicted is not None
        assert b._turn_count == 4


# ── TestGetSummaryText ──────────────────────────────────────────────────────

class TestGetSummaryText:

    def test_empty_tree_returns_empty_string(self):
        b = Bucket()
        assert b.get_summary_text() == ""

    def test_single_topic_returns_formatted_text(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [_make_topic("Deployment plan", "Use Docker on AWS.")]
        }
        text = b.get_summary_text()
        assert "[Topic: Deployment plan" in text
        assert "Use Docker on AWS." in text

    def test_staleness_annotation_shows_turns_ago(self):
        b = Bucket()
        b._turn_count = 10
        b.topic_tree = {
            "topics": [_make_topic("Old topic", "stale", updated_at_turn=7)]
        }
        text = b.get_summary_text()
        assert "3 turns ago" in text

    def test_singular_turn_ago(self):
        b = Bucket()
        b._turn_count = 1
        b.topic_tree = {
            "topics": [_make_topic("Recent", "just now", updated_at_turn=0)]
        }
        text = b.get_summary_text()
        assert "1 turn ago" in text
        assert "1 turns ago" not in text

    def test_zero_turns_ago(self):
        b = Bucket()
        b._turn_count = 5
        b.topic_tree = {
            "topics": [_make_topic("Current", "now", updated_at_turn=5)]
        }
        text = b.get_summary_text()
        assert "0 turns ago" in text


# ── TestNestedSubtopics ────────────────────────────────────────────────────

class TestNestedSubtopics:

    def test_subtopics_use_subtopic_label(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("Parent", "parent summary", subtopics=[
                    _make_topic("Child", "child summary"),
                ]),
            ]
        }
        text = b.get_summary_text()
        assert "[Subtopic: Child" in text

    def test_subtopics_are_indented(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("Parent", "parent summary", subtopics=[
                    _make_topic("Child", "child summary"),
                ]),
            ]
        }
        text = b.get_summary_text()
        # Depth-1 subtopic lines should start with exactly 2 spaces
        for line in text.splitlines():
            if "Subtopic: Child" in line:
                assert line.startswith("  [Subtopic:")
            if line.strip() == "child summary":
                assert line.startswith("  ")

    def test_deeply_nested_subtopics(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("L0", "level 0", subtopics=[
                    _make_topic("L1", "level 1", subtopics=[
                        _make_topic("L2", "level 2"),
                    ]),
                ]),
            ]
        }
        text = b.get_summary_text()
        # Depth-2 should have 4-space indent
        for line in text.splitlines():
            if "L2" in line and "Subtopic" in line:
                assert line.startswith("    [Subtopic:")
            if line.strip() == "level 2":
                assert line.startswith("    ")

    def test_multiple_subtopics_under_one_topic(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("Parent", "parent summary", subtopics=[
                    _make_topic("ChildA", "A notes"),
                    _make_topic("ChildB", "B notes"),
                ]),
            ]
        }
        text = b.get_summary_text()
        assert "ChildA" in text
        assert "ChildB" in text


# ── TestSummaryPropertyGetter ──────────────────────────────────────────────

class TestSummaryPropertyGetter:

    def test_summary_getter_returns_get_summary_text(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [_make_topic("Foo", "bar")]
        }
        assert b.summary == b.get_summary_text()

    def test_summary_getter_empty_when_no_topics(self):
        b = Bucket()
        assert b.summary == ""


# ── TestSummaryPropertySetter ──────────────────────────────────────────────

class TestSummaryPropertySetter:

    def test_setting_summary_creates_legacy_topic(self):
        b = Bucket()
        b.summary = "hello"
        topics = b.topic_tree["topics"]
        assert len(topics) == 1
        assert topics[0]["title"] == "Conversation summary"
        assert topics[0]["summary"] == "hello"
        assert topics[0]["id"] == "legacy_summary"

    def test_empty_string_does_not_create_topic(self):
        b = Bucket()
        b.summary = ""
        assert b.topic_tree == {"topics": []}

    def test_whitespace_only_does_not_create_topic(self):
        b = Bucket()
        b.summary = "   "
        assert b.topic_tree == {"topics": []}

    def test_none_does_not_create_topic(self):
        b = Bucket()
        b.summary = None
        assert b.topic_tree == {"topics": []}

    def test_legacy_topic_has_correct_turn_count(self):
        b = Bucket()
        for i in range(3):
            b.push_message(f"q{i}", f"a{i}")
        b.summary = "After three turns."
        assert b.topic_tree["topics"][0]["updated_at_turn"] == 3


# ── TestContextStringWithTopicTree ─────────────────────────────────────────

class TestContextStringWithTopicTree:

    def test_context_string_shows_topic_summary(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("Architecture", "Chose microservices over monolith."),
                _make_topic("Database", "PostgreSQL for relational data."),
            ]
        }
        ctx = b.to_context_string()
        assert "Architecture" in ctx
        assert "Chose microservices over monolith." in ctx
        assert "Database" in ctx
        assert "PostgreSQL for relational data." in ctx

    def test_context_string_shows_no_summary_fallback_when_empty(self):
        b = Bucket()
        ctx = b.to_context_string()
        assert "(no summary yet)" in ctx
