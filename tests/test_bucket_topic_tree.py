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

def _make_topic(
    title,
    facts=None,
    decisions=None,
    preferences=None,
    open_threads=None,
    updated_at_turn=0,
    subtopics=None,
):
    """Return a minimal topic dict matching the typed-slot schema."""
    return {
        "id": title.lower().replace(" ", "_"),
        "title": title,
        "facts":        list(facts or []),
        "decisions":    list(decisions or []),
        "preferences":  list(preferences or []),
        "open_threads": list(open_threads or []),
        "subtopics":    subtopics or [],
        "created_at":   0,
        "updated_at":   0,
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
            "topics": [_make_topic("Deployment plan", facts=["Use Docker on AWS."])]
        }
        text = b.get_summary_text()
        assert "[Topic: Deployment plan" in text
        assert "Facts:" in text
        assert "• Use Docker on AWS." in text

    def test_staleness_annotation_shows_turns_ago(self):
        b = Bucket()
        b._turn_count = 10
        b.topic_tree = {
            "topics": [_make_topic("Old topic", facts=["stale"], updated_at_turn=7)]
        }
        text = b.get_summary_text()
        assert "3 turns ago" in text

    def test_singular_turn_ago(self):
        b = Bucket()
        b._turn_count = 1
        b.topic_tree = {
            "topics": [_make_topic("Recent", facts=["just now"], updated_at_turn=0)]
        }
        text = b.get_summary_text()
        assert "1 turn ago" in text
        assert "1 turns ago" not in text

    def test_zero_turns_ago(self):
        b = Bucket()
        b._turn_count = 5
        b.topic_tree = {
            "topics": [_make_topic("Current", facts=["now"], updated_at_turn=5)]
        }
        text = b.get_summary_text()
        assert "0 turns ago" in text

    def test_all_slot_labels_rendered(self):
        """Every populated slot should appear with its label and bullets."""
        b = Bucket()
        b.topic_tree = {
            "topics": [_make_topic(
                "Full node",
                facts=["fact one"],
                decisions=["decision one"],
                preferences=["pref one"],
                open_threads=["thread one"],
            )]
        }
        text = b.get_summary_text()
        assert "Facts:" in text
        assert "Decisions:" in text
        assert "Preferences:" in text
        assert "Open threads:" in text
        assert "• fact one" in text
        assert "• decision one" in text
        assert "• pref one" in text
        assert "• thread one" in text

    def test_empty_slots_are_omitted(self):
        """Unpopulated slots should not render a label."""
        b = Bucket()
        b.topic_tree = {
            "topics": [_make_topic("Sparse", facts=["only fact"])]
        }
        text = b.get_summary_text()
        assert "Facts:" in text
        assert "Decisions:" not in text
        assert "Preferences:" not in text
        assert "Open threads:" not in text
        assert "Quotes:" not in text

    def test_multiple_items_in_one_slot(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [_make_topic("Many facts", facts=["one", "two", "three"])]
        }
        text = b.get_summary_text()
        assert "• one" in text
        assert "• two" in text
        assert "• three" in text

    def test_node_with_no_slot_items_renders_header_only(self):
        """A topic with every slot empty still renders its header line."""
        b = Bucket()
        b.topic_tree = {"topics": [_make_topic("Title only")]}
        text = b.get_summary_text()
        assert "[Topic: Title only" in text
        # No bullets anywhere since all slots empty.
        assert "•" not in text


# ── TestNestedSubtopics ────────────────────────────────────────────────────

class TestNestedSubtopics:

    def test_subtopics_use_subtopic_label(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("Parent", facts=["parent fact"], subtopics=[
                    _make_topic("Child", facts=["child fact"]),
                ]),
            ]
        }
        text = b.get_summary_text()
        assert "[Subtopic: Child" in text

    def test_subtopics_are_indented(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("Parent", facts=["parent fact"], subtopics=[
                    _make_topic("Child", facts=["child fact"]),
                ]),
            ]
        }
        text = b.get_summary_text()
        lines = text.splitlines()
        # Depth-1 subtopic header line should start with exactly 2 spaces.
        assert any(l.startswith("  [Subtopic: Child") for l in lines)
        # Depth-1 bullet under child: 2 (indent) + 4 (body offset) = 6 spaces.
        assert any(l == "      • child fact" for l in lines)
        # Depth-0 bullet under parent: 0 (indent) + 4 (body offset) = 4 spaces.
        assert any(l == "    • parent fact" for l in lines)

    def test_deeply_nested_subtopics(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("L0", facts=["level 0"], subtopics=[
                    _make_topic("L1", facts=["level 1"], subtopics=[
                        _make_topic("L2", facts=["level 2"]),
                    ]),
                ]),
            ]
        }
        text = b.get_summary_text()
        # Depth-2 should have 4-space indent on the header
        for line in text.splitlines():
            if "L2" in line and "Subtopic" in line:
                assert line.startswith("    [Subtopic:")
            # Depth-2 bullet: 4 (indent) + 4 (body offset) = 8 spaces before •
            if line.strip() == "• level 2":
                assert line.startswith("        • level 2")

    def test_multiple_subtopics_under_one_topic(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("Parent", facts=["parent"], subtopics=[
                    _make_topic("ChildA", facts=["A notes"]),
                    _make_topic("ChildB", facts=["B notes"]),
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
            "topics": [_make_topic("Foo", facts=["bar"])]
        }
        assert b.summary == b.get_summary_text()

    def test_summary_getter_empty_when_no_topics(self):
        b = Bucket()
        assert b.summary == ""


# ── TestSummaryPropertySetter ──────────────────────────────────────────────

class TestSummaryPropertySetter:

    def test_setting_summary_creates_legacy_topic(self):
        """The deprecated setter should land the prose in the facts slot."""
        b = Bucket()
        b.summary = "hello"
        topics = b.topic_tree["topics"]
        assert len(topics) == 1
        assert topics[0]["title"] == "Conversation summary"
        assert topics[0]["facts"] == ["hello"]
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

    def test_legacy_summary_renders_through_facts_slot(self):
        """End-to-end: legacy setter output should flow through the new renderer."""
        b = Bucket()
        b.summary = "Chose Postgres."
        text = b.get_summary_text()
        assert "[Topic: Conversation summary" in text
        assert "Facts:" in text
        assert "• Chose Postgres." in text


# ── TestContextStringWithTopicTree ─────────────────────────────────────────

class TestContextStringWithTopicTree:

    def test_context_string_shows_topic_summary(self):
        b = Bucket()
        b.topic_tree = {
            "topics": [
                _make_topic("Architecture", facts=["Chose microservices over monolith."]),
                _make_topic("Database",     facts=["PostgreSQL for relational data."]),
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
