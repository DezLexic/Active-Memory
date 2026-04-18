"""
test_observer_unit.py

Unit tests for Observer.update() --- JSON parsing, markdown fence stripping,
early-return guard, prompt construction, and error-path resilience.

No real LLM calls.  Uses FakeBackend (from conftest.py) and plain Bucket
instances with no external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import logging
import pytest

from active_memory.observer import Observer
from active_memory.bucket   import Bucket
from tests.conftest         import FakeBackend


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_tree(*titles: str) -> dict:
    """Build a minimal valid topic tree with the given topic titles."""
    return {
        "topics": [
            {
                "id": t.lower().replace(" ", "_"),
                "title": t,
                "facts":        [f"Concrete fact for {t}"],
                "decisions":    [],
                "preferences":  [],
                "open_threads": [],
                "subtopics":    [],
                "created_at":   1000000,
                "updated_at":   1000000,
                "updated_at_turn": 0,
            }
            for t in titles
        ]
    }


def _pairs(*texts: tuple[str, str]) -> list[dict[str, str]]:
    """Shorthand for building popped_pairs lists."""
    return [{"question": q, "response": r} for q, r in texts]


_SAMPLE_PAIRS = _pairs(
    ("What language?", "Python with FastAPI."),
    ("Which database?", "PostgreSQL, no ORM."),
)

_ORIGINAL_TREE = _make_tree("Setup")


# ── TestEarlyReturn ──────────────────────────────────────────────────────────

class TestEarlyReturn:

    def test_none_popped_pairs_returns_immediately(self):
        backend = FakeBackend()
        observer = Observer(backend=backend)
        bucket = Bucket()
        observer.update(bucket, None)
        assert len(backend.calls) == 0

    def test_empty_list_returns_immediately(self):
        backend = FakeBackend()
        observer = Observer(backend=backend)
        bucket = Bucket()
        observer.update(bucket, [])
        assert len(backend.calls) == 0

    def test_bucket_tree_unchanged_on_none(self):
        backend = FakeBackend()
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = _make_tree("Original")
        original_copy = json.loads(json.dumps(bucket.topic_tree))
        observer.update(bucket, None)
        assert bucket.topic_tree == original_copy

    def test_bucket_tree_unchanged_on_empty_list(self):
        backend = FakeBackend()
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = _make_tree("Original")
        original_copy = json.loads(json.dumps(bucket.topic_tree))
        observer.update(bucket, [])
        assert bucket.topic_tree == original_copy


# ── TestValidJsonResponse ────────────────────────────────────────────────────

class TestValidJsonResponse:

    def test_valid_json_updates_topic_tree(self):
        new_tree = _make_tree("Python", "Database")
        backend = FakeBackend(responses=[json.dumps(new_tree)])
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = _make_tree("Setup")

        observer.update(bucket, _SAMPLE_PAIRS)
        assert bucket.topic_tree == new_tree

    def test_single_topic_node_stored(self):
        single = _make_tree("Authentication")
        backend = FakeBackend(responses=[json.dumps(single)])
        observer = Observer(backend=backend)
        bucket = Bucket()

        observer.update(bucket, _SAMPLE_PAIRS)
        assert len(bucket.topic_tree["topics"]) == 1
        assert bucket.topic_tree["topics"][0]["title"] == "Authentication"
        assert bucket.topic_tree["topics"][0]["id"] == "authentication"
        # New typed-slot schema: every node should carry all four slot keys.
        for slot in ("facts", "decisions", "preferences", "open_threads"):
            assert slot in bucket.topic_tree["topics"][0]

    def test_multiple_topics_stored(self):
        multi = _make_tree("Language", "Database")
        backend = FakeBackend(responses=[json.dumps(multi)])
        observer = Observer(backend=backend)
        bucket = Bucket()

        observer.update(bucket, _SAMPLE_PAIRS)
        assert len(bucket.topic_tree["topics"]) == 2

    def test_prompt_includes_current_tree(self):
        existing = _make_tree("Existing Topic")
        new_tree = _make_tree("Existing Topic", "New Topic")
        backend = FakeBackend(responses=[json.dumps(new_tree)])
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = existing

        observer.update(bucket, _SAMPLE_PAIRS)

        # The prompt sent to the backend should contain the serialized existing tree
        assert len(backend.calls) == 1
        prompt = backend.calls[0][0]["content"]
        assert "Existing Topic" in prompt
        assert json.dumps(existing, indent=2) in prompt

    def test_prompt_includes_evicted_pairs(self):
        new_tree = _make_tree("Anything")
        backend = FakeBackend(responses=[json.dumps(new_tree)])
        observer = Observer(backend=backend)
        bucket = Bucket()

        pairs = _pairs(
            ("What language?", "Python with FastAPI."),
            ("Which database?", "PostgreSQL, no ORM."),
        )
        observer.update(bucket, pairs)

        prompt = backend.calls[0][0]["content"]
        assert "Q: What language?" in prompt
        assert "A: Python with FastAPI." in prompt
        assert "Q: Which database?" in prompt
        assert "A: PostgreSQL, no ORM." in prompt

    def test_exactly_one_backend_call(self):
        new_tree = _make_tree("Topic")
        backend = FakeBackend(responses=[json.dumps(new_tree)])
        observer = Observer(backend=backend)
        bucket = Bucket()

        observer.update(bucket, _SAMPLE_PAIRS)
        assert len(backend.calls) == 1


# ── TestJsonParseFailure ─────────────────────────────────────────────────────

class TestJsonParseFailure:

    def test_invalid_json_leaves_tree_unchanged(self):
        backend = FakeBackend(responses=["not json at all"])
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = _make_tree("Original")
        original_copy = json.loads(json.dumps(bucket.topic_tree))

        observer.update(bucket, _SAMPLE_PAIRS)
        assert bucket.topic_tree == original_copy

    def test_missing_topics_key_leaves_tree_unchanged(self):
        backend = FakeBackend(responses=['{"data": []}'])
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = _make_tree("Original")
        original_copy = json.loads(json.dumps(bucket.topic_tree))

        observer.update(bucket, _SAMPLE_PAIRS)
        assert bucket.topic_tree == original_copy

    def test_topics_not_a_list_leaves_tree_unchanged(self):
        backend = FakeBackend(responses=['{"topics": "string"}'])
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = _make_tree("Original")
        original_copy = json.loads(json.dumps(bucket.topic_tree))

        observer.update(bucket, _SAMPLE_PAIRS)
        assert bucket.topic_tree == original_copy

    def test_truncated_json_leaves_tree_unchanged(self):
        backend = FakeBackend(responses=['{"topics": [{"id": "a"'])
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = _make_tree("Original")
        original_copy = json.loads(json.dumps(bucket.topic_tree))

        observer.update(bucket, _SAMPLE_PAIRS)
        assert bucket.topic_tree == original_copy

    def test_warning_logged_on_parse_failure(self, caplog):
        backend = FakeBackend(responses=["not json at all"])
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = _make_tree("Original")

        with caplog.at_level(logging.WARNING, logger="active_memory.observer"):
            observer.update(bucket, _SAMPLE_PAIRS)

        assert len(caplog.records) >= 1
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("failed to parse topic tree JSON" in m for m in warning_messages)


# ── TestMarkdownFenceStripping ───────────────────────────────────────────────

class TestMarkdownFenceStripping:

    def test_json_fences_stripped(self):
        tree = _make_tree("Fenced")
        fenced = "```json\n" + json.dumps(tree) + "\n```"
        backend = FakeBackend(responses=[fenced])
        observer = Observer(backend=backend)
        bucket = Bucket()

        observer.update(bucket, _SAMPLE_PAIRS)
        assert bucket.topic_tree == tree

    def test_plain_fences_stripped(self):
        tree = _make_tree("Fenced")
        fenced = "```\n" + json.dumps(tree) + "\n```"
        backend = FakeBackend(responses=[fenced])
        observer = Observer(backend=backend)
        bucket = Bucket()

        observer.update(bucket, _SAMPLE_PAIRS)
        assert bucket.topic_tree == tree

    def test_no_fences_works_normally(self):
        tree = _make_tree("NoFence")
        backend = FakeBackend(responses=[json.dumps(tree)])
        observer = Observer(backend=backend)
        bucket = Bucket()

        observer.update(bucket, _SAMPLE_PAIRS)
        assert bucket.topic_tree == tree

    def test_nested_fence_markers_handled(self):
        """
        Edge case: a topic title contains backtick fence markers.
        The stripping logic removes lines that START with ```, so interior
        occurrences inside JSON values should survive.  This test verifies
        the call does not crash; the tree may or may not parse correctly
        depending on how the fence-stripping interacts with the content.
        """
        tree = {
            "topics": [
                {
                    "id": "code_example",
                    "title": "```code```",
                    "facts":        ["A code example topic"],
                    "decisions":    [],
                    "preferences":  [],
                    "open_threads": [],
                        "subtopics":    [],
                    "created_at":   1000000,
                    "updated_at":   1000000,
                    "updated_at_turn": 0,
                }
            ]
        }
        fenced = "```json\n" + json.dumps(tree) + "\n```"
        backend = FakeBackend(responses=[fenced])
        observer = Observer(backend=backend)
        bucket = Bucket()
        bucket.topic_tree = _make_tree("Original")

        # Must not raise --- that is the primary assertion
        observer.update(bucket, _SAMPLE_PAIRS)


# ── TestRepr ─────────────────────────────────────────────────────────────────

class TestRepr:

    def test_repr_contains_backend(self):
        backend = FakeBackend()
        observer = Observer(backend=backend)
        r = repr(observer)
        assert "FakeBackend" in r

    def test_repr_contains_max_summary_length(self):
        backend = FakeBackend()
        observer = Observer(backend=backend, max_summary_length=150)
        r = repr(observer)
        assert "150" in r
