"""
test_integration.py

Integration tests that wire multiple components through the Pipeline with
FakeBackend — no live LLM calls, real ChromaDB via pytest tmp_path.

These verify that Observer, Curator, Retrieval, and Bucket work together
correctly when orchestrated by the Pipeline.

Eviction setup used across all tests
-------------------------------------
    max_recent_messages=3, batch_reduction=2

    push 1: stack=[1]          no eviction
    push 2: stack=[1,2]        no eviction
    push 3: stack=[1,2,3]      no eviction   (len was 2 before push)
    push 4: len==3 >= 3 → evict [1,2], stack=[3,4]  ← first eviction

When a single FakeBackend serves all roles, response ordering depends on
which method drives the pipeline:

  pipeline.update()  — no agent call → [observer, curator, curator] per eviction
  pipeline.ingest()  — no agent call → [observer, curator, curator] per eviction
  pipeline.chat()    — agent first   → [agent, observer, curator, curator] per eviction turn
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from active_memory import Pipeline
from tests.conftest import FakeBackend


# ── Helpers ──────────────────────────────────────────────────────────────────

_OBSERVER_TREE = json.dumps({
    "topics": [{
        "id": "integration_topic",
        "title": "Integration Test Topic",
        "facts":        ["A topic produced during integration testing."],
        "decisions":    [],
        "preferences":  [],
        "open_threads": [],
        "subtopics":    [],
        "created_at":   0,
        "updated_at":   0,
        "updated_at_turn": 0,
    }]
})

_CURATOR_STORE = json.dumps({
    "reason": "important information",
    "tier": "warm",
})

_CURATOR_DROP = json.dumps({
    "reason": "not important",
    "tier": "cold",
})


def _pipeline(tmp_path, responses=None, max_recent=3, batch_reduction=2):
    """
    Create a Pipeline with a single FakeBackend serving all roles.

    When only `backend` is passed (no observer_backend / curator_backend),
    Pipeline routes all three agents (ActiveAgent, Observer, Curator) through
    the same backend.  Response order matters: calls are consumed in the
    order the pipeline makes them.
    """
    backend = FakeBackend(responses or [])
    pipe = Pipeline(
        backend=backend,
        chroma_path=str(tmp_path / "int_chroma"),
        max_recent_messages=max_recent,
        batch_reduction=batch_reduction,
    )
    return pipe, backend


# ── TestObserverTopicTreeFlow ────────────────────────────────────────────────

class TestObserverTopicTreeFlow:
    """Verify Pipeline → Observer → Bucket topic tree round-trip."""

    def test_eviction_updates_topic_tree(self, tmp_path):
        """Push enough messages to trigger eviction; verify topic_tree is updated."""
        # update() makes no agent call.  On eviction: Observer call, then Curator call.
        # Pushes 1-3 consume no responses.  Push 4 triggers eviction:
        #   call 1 → Observer (needs valid topic tree JSON)
        #   call 2 → Curator  (needs valid store/drop JSON)
        responses = [_OBSERVER_TREE, _CURATOR_DROP]
        pipe, backend = _pipeline(tmp_path, responses=responses)

        for i in range(4):
            pipe.update(f"q{i}", f"a{i}")

        tree = pipe.bucket.topic_tree
        assert "topics" in tree
        assert len(tree["topics"]) >= 1
        assert tree["topics"][0]["title"] == "Integration Test Topic"

    def test_topic_tree_appears_in_context_string(self, tmp_path):
        """After eviction updates the tree, build_context() includes the topic title."""
        responses = [_OBSERVER_TREE, _CURATOR_DROP]
        pipe, backend = _pipeline(tmp_path, responses=responses)

        for i in range(4):
            pipe.update(f"q{i}", f"a{i}")

        msgs = pipe.build_context("follow-up question")
        system_content = msgs[0]["content"]
        assert "Integration Test Topic" in system_content

    def test_observer_parse_failure_preserves_old_tree(self, tmp_path):
        """Invalid Observer JSON must not overwrite the existing topic tree."""
        # First set up a known tree via a successful eviction.
        good_responses = [_OBSERVER_TREE, _CURATOR_DROP]
        pipe, _ = _pipeline(tmp_path, responses=good_responses)

        for i in range(4):
            pipe.update(f"q{i}", f"a{i}")

        original_tree = pipe.bucket.topic_tree.copy()
        assert original_tree["topics"][0]["title"] == "Integration Test Topic"

        # Now trigger a second eviction with broken Observer JSON.
        # Pushes 5-6 fill the stack back to 3 (stack after first eviction = [q2,a2, q3,a3]).
        # Actually after eviction: stack=[q2,q3,q4(just pushed)]=3 items? Let me trace:
        #   After push 4: evict [pair0, pair1], stack=[pair2, pair3] then append pair3...
        #   Wait, re-read: push_message evicts first, then appends.
        #   push 4: len=3 >= 3 → evict first 2 → stack=[pair2], append pair3 → stack=[pair2, pair3]
        #   push 5: len=2 < 3 → no eviction, append → stack=[pair2, pair3, pair4]
        #   push 6: len=3 >= 3 → evict [pair2, pair3] → stack=[pair4], append pair5 → second eviction
        #
        # The FakeBackend from the first _pipeline call is exhausted.
        # We need to inject new responses for the second eviction.
        # Easier: create a fresh pipeline with enough responses for two evictions.

        bad_observer = "NOT VALID JSON {{{{"
        responses_for_two_evictions = [
            _OBSERVER_TREE, _CURATOR_DROP,   # first eviction — sets tree
            bad_observer,   _CURATOR_DROP,   # second eviction — Observer fails
        ]
        pipe2, _ = _pipeline(tmp_path / "sub", responses=responses_for_two_evictions)

        # First eviction: sets the tree successfully
        for i in range(4):
            pipe2.update(f"q{i}", f"a{i}")
        assert pipe2.bucket.topic_tree["topics"][0]["title"] == "Integration Test Topic"

        # Second eviction: Observer response is garbage → tree should stay the same
        pipe2.update("q4", "a4")   # stack goes to 3: [pair2, pair3, pair4]
        pipe2.update("q5", "a5")   # len=3 → eviction, Observer gets bad JSON

        assert pipe2.bucket.topic_tree["topics"][0]["title"] == "Integration Test Topic"


# ── TestCuratorRetrievalRoundTrip ────────────────────────────────────────────

class TestCuratorRetrievalRoundTrip:
    """Verify Pipeline → Curator → Retrieval stores and retrieves memories."""

    def test_curator_warm_classification_lands_in_chroma(self, tmp_path):
        """Warm classification should persist memories in the warm tier."""
        # update() on eviction: Observer, then Curator once per evicted pair.
        # batch_reduction=2 → 2 pairs evicted → 2 Curator calls needed.
        responses = [_OBSERVER_TREE, _CURATOR_STORE, _CURATOR_STORE]
        pipe, _ = _pipeline(tmp_path, responses=responses)

        for i in range(4):
            pipe.update(f"q{i}", f"a{i}")

        all_meta = pipe.retrieval.get_all_metadata()
        assert len(all_meta) == 2
        assert all(m["tier"] == "warm" for m in all_meta)

    def test_stored_memory_surfaces_in_warm_collection(self, tmp_path):
        """Warm classification should write each evicted pair once."""
        # batch_reduction=2 → 2 pairs evicted → 2 Curator calls needed.
        responses = [_OBSERVER_TREE, _CURATOR_STORE, _CURATOR_STORE]
        pipe, _ = _pipeline(tmp_path, responses=responses)

        assert pipe.retrieval._warm.count() == 0

        for i in range(4):
            pipe.update(f"q{i}", f"a{i}")

        assert pipe.retrieval._warm.count() == 2


# ── TestIngestFlow ───────────────────────────────────────────────────────────

class TestIngestFlow:
    """Verify Pipeline.ingest() triggers Observer and Curator correctly."""

    def test_ingest_fills_stack_and_triggers_observer(self, tmp_path):
        """Ingest enough pairs to trigger eviction; Observer updates topic tree."""
        # ingest() on eviction: Observer, then Curator (same as update).
        responses = [_OBSERVER_TREE, _CURATOR_DROP]
        pipe, backend = _pipeline(tmp_path, responses=responses)

        for i in range(4):
            pipe.ingest(f"q{i}", f"a{i}")

        # Observer was called → topic tree updated
        tree = pipe.bucket.topic_tree
        assert "topics" in tree
        assert len(tree["topics"]) >= 1
        assert tree["topics"][0]["title"] == "Integration Test Topic"

    def test_ingest_many_pairs_builds_memories(self, tmp_path):
        """Ingest 10 pairs with warm classification; memories accumulate."""
        # With max_recent=3, batch_reduction=2:
        #   push 1-3: no eviction
        #   push 4: eviction 1 → Observer + Curator
        #   push 5: stack=[pair2, pair3, pair4] → no eviction
        #   push 6: eviction 2 → Observer + Curator
        #   push 7: no eviction
        #   push 8: eviction 3 → Observer + Curator
        #   push 9: no eviction
        #   push 10: eviction 4 → Observer + Curator
        #
        # 4 evictions × 2 responses each = 8 responses needed.
        num_evictions = 4
        responses = [_OBSERVER_TREE, _CURATOR_STORE] * num_evictions
        pipe, _ = _pipeline(tmp_path, responses=responses)

        for i in range(10):
            pipe.ingest(f"question_{i}", f"answer_{i}")

        all_meta = pipe.retrieval.get_all_metadata()
        assert len(all_meta) >= 1
        # Each eviction writes both evicted pairs once to the selected tier.
        assert pipe.retrieval._warm.count() > 0
