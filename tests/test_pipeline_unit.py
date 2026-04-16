"""
test_pipeline_unit.py

Unit tests for Pipeline.build_context(), Pipeline.update(), Pipeline.chat(),
and Pipeline.ingest() — no real LLM calls, real ChromaDB via pytest tmp_path.

Every test uses FakeBackend (from conftest.py) for all three roles so that
Observer and Curator never attempt a network connection.

Eviction setup used across multiple tests
-----------------------------------------
    max_recent_messages=3, batch_reduction=2

    push 1: stack=[1]          no eviction
    push 2: stack=[1,2]        no eviction
    push 3: stack=[1,2,3]      no eviction   (len==max before this push was 2)
    push 4: len==3 >= 3 → evict [1,2], stack=[3,4]  ← first eviction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from active_memory import Pipeline
from active_memory.bucket import Bucket
from active_memory.retrieval import Retrieval

# Import shared FakeBackend from conftest.py
from tests.conftest import FakeBackend

# ── Helpers ───────────────────────────────────────────────────────────────────

_CURATOR_OK = '{"store": false, "reason": "unit test", "tier": "warm"}'
_OBS_REPLY  = (
    '{"topics": [{'
    '"id": "obs_update", "title": "Observer update", '
    '"facts": ["Updated."], "decisions": [], "preferences": [], '
    '"open_threads": [], "quotes": [], '
    '"subtopics": [], "created_at": 0, "updated_at": 0, "updated_at_turn": 0'
    '}]}'
)


def _pipeline(tmp_path, *, max_recent=10, batch_reduction=5,
              agent_responses=None, obs_responses=None, cur_responses=None,
              system_instructions=None):
    """Convenience factory producing an isolated Pipeline for each test."""
    return Pipeline(
        backend=FakeBackend(responses=agent_responses or ["Agent reply."]),
        observer_backend=FakeBackend(responses=obs_responses or [_OBS_REPLY] * 20),
        curator_backend=FakeBackend(responses=cur_responses or [_CURATOR_OK] * 20),
        chroma_path=str(tmp_path),
        max_recent_messages=max_recent,
        batch_reduction=batch_reduction,
        system_instructions=system_instructions,
    )


def _fill_to_eviction(pipeline, *, count=4, max_recent=3):
    """Push `count` pairs; with max_recent=3 the 4th triggers the first eviction."""
    for i in range(count):
        pipeline.update(f"question_{i}", f"answer_{i}")


# ── build_context() ───────────────────────────────────────────────────────────

class TestBuildContext:

    def test_returns_two_message_list(self, tmp_path):
        p = _pipeline(tmp_path)
        msgs = p.build_context("hello")
        assert len(msgs) == 2

    def test_first_message_is_system(self, tmp_path):
        p = _pipeline(tmp_path)
        msgs = p.build_context("hello")
        assert msgs[0]["role"] == "system"

    def test_second_message_is_user(self, tmp_path):
        p = _pipeline(tmp_path)
        msgs = p.build_context("hello")
        assert msgs[1]["role"] == "user"

    def test_user_content_matches_input(self, tmp_path):
        p = _pipeline(tmp_path)
        msgs = p.build_context("What is the plan?")
        assert msgs[1]["content"] == "What is the plan?"

    def test_sets_bucket_current_prompt(self, tmp_path):
        p = _pipeline(tmp_path)
        p.build_context("my question")
        assert p.bucket.current_prompt == "my question"

    def test_system_message_contains_custom_instructions(self, tmp_path):
        p = _pipeline(tmp_path, system_instructions="CUSTOM RULE")
        msgs = p.build_context("hi")
        assert "CUSTOM RULE" in msgs[0]["content"]

    def test_system_message_contains_no_summary_fallback_when_empty(self, tmp_path):
        p = _pipeline(tmp_path)
        msgs = p.build_context("hi")
        assert "(no summary yet)" in msgs[0]["content"]

    def test_system_message_contains_memories_fallback_when_empty(self, tmp_path):
        p = _pipeline(tmp_path)
        msgs = p.build_context("hi")
        assert "(none retrieved)" in msgs[0]["content"]

    def test_system_content_is_string(self, tmp_path):
        p = _pipeline(tmp_path)
        msgs = p.build_context("hi")
        assert isinstance(msgs[0]["content"], str)
        assert len(msgs[0]["content"]) > 0

    def test_called_twice_updates_prompt_each_time(self, tmp_path):
        p = _pipeline(tmp_path)
        p.build_context("first")
        assert p.bucket.current_prompt == "first"
        p.build_context("second")
        assert p.bucket.current_prompt == "second"


# ── update() ─────────────────────────────────────────────────────────────────

class TestUpdate:

    def test_pushes_pair_to_recent_messages(self, tmp_path):
        p = _pipeline(tmp_path)
        p.update("q", "a")
        assert len(p.bucket.recent_messages) == 1
        pair = p.bucket.recent_messages[0]
        assert pair["question"] == "q"
        assert pair["response"] == "a"

    def test_no_eviction_does_not_call_observer(self, tmp_path):
        obs = FakeBackend()
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=obs,
            curator_backend=FakeBackend(responses=[_CURATOR_OK] * 5),
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        p.update("q", "a")          # stack = 1/3 — no eviction
        assert len(obs.calls) == 0

    def test_no_eviction_does_not_call_curator(self, tmp_path):
        cur = FakeBackend()
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=FakeBackend(responses=[_OBS_REPLY] * 5),
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        p.update("q", "a")          # stack = 1/3 — no eviction
        assert len(cur.calls) == 0

    def test_eviction_calls_observer_once(self, tmp_path):
        obs = FakeBackend(responses=[_OBS_REPLY] * 10)
        cur = FakeBackend(responses=[_CURATOR_OK] * 10)
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=obs,
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        for i in range(3):
            p.update(f"q{i}", f"a{i}")
        assert len(obs.calls) == 0   # not yet evicted

        p.update("q3", "a3")         # 4th push triggers eviction
        assert len(obs.calls) == 1

    def test_skip_observer_true_suppresses_observer(self, tmp_path):
        obs = FakeBackend()
        cur = FakeBackend(responses=[_CURATOR_OK] * 10)
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=obs,
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        for i in range(4):
            p.update(f"q{i}", f"a{i}", skip_observer=True)
        assert len(obs.calls) == 0

    def test_skip_observer_false_still_calls_observer(self, tmp_path):
        obs = FakeBackend(responses=[_OBS_REPLY] * 5)
        cur = FakeBackend(responses=[_CURATOR_OK] * 10)
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=obs,
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        for i in range(4):
            p.update(f"q{i}", f"a{i}", skip_observer=False)
        assert len(obs.calls) == 1

    def test_eviction_calls_curator_in_batch_mode(self, tmp_path):
        cur = FakeBackend(responses=[_CURATOR_OK] * 10)
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=FakeBackend(responses=[_OBS_REPLY] * 5),
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        for i in range(4):
            p.update(f"q{i}", f"a{i}")
        # In batch mode, Curator is called once for the peek_curator_target pair
        assert len(cur.calls) == 1

    def test_multiple_evictions_accumulate_observer_calls(self, tmp_path):
        obs = FakeBackend(responses=[_OBS_REPLY] * 10)
        cur = FakeBackend(responses=[_CURATOR_OK] * 20)
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=obs,
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        # 4 → first eviction, 6 → second eviction
        for i in range(6):
            p.update(f"q{i}", f"a{i}")
        assert len(obs.calls) == 2


# ── chat() ───────────────────────────────────────────────────────────────────

class TestChat:

    def test_returns_agent_response(self, tmp_path):
        agent = FakeBackend(responses=["Agent says hello."])
        p = Pipeline(
            backend=agent,
            observer_backend=FakeBackend(responses=[_OBS_REPLY] * 5),
            curator_backend=FakeBackend(responses=[_CURATOR_OK] * 5),
            chroma_path=str(tmp_path),
        )
        result = p.chat("hi")
        assert result == "Agent says hello."

    def test_chat_calls_agent_backend_once_per_turn(self, tmp_path):
        agent = FakeBackend(responses=["r1", "r2", "r3"])
        p = Pipeline(
            backend=agent,
            observer_backend=FakeBackend(responses=[_OBS_REPLY] * 5),
            curator_backend=FakeBackend(responses=[_CURATOR_OK] * 10),
            chroma_path=str(tmp_path),
        )
        p.chat("turn 1")
        p.chat("turn 2")
        assert len(agent.calls) == 2

    def test_chat_advances_bucket_recent_messages(self, tmp_path):
        p = _pipeline(tmp_path, agent_responses=["reply"] * 5)
        p.chat("msg1")
        p.chat("msg2")
        assert len(p.bucket.recent_messages) == 2

    def test_chat_skip_observer_propagates_to_update(self, tmp_path):
        obs = FakeBackend()
        cur = FakeBackend(responses=[_CURATOR_OK] * 20)
        agent = FakeBackend(responses=["ok"] * 20)
        p = Pipeline(
            backend=agent,
            observer_backend=obs,
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        for _ in range(4):
            p.chat("q", skip_observer=True)
        assert len(obs.calls) == 0   # skip_observer prevented observer

    def test_chat_without_skip_observer_calls_observer_on_eviction(self, tmp_path):
        obs = FakeBackend(responses=[_OBS_REPLY] * 5)
        cur = FakeBackend(responses=[_CURATOR_OK] * 20)
        agent = FakeBackend(responses=["ok"] * 20)
        p = Pipeline(
            backend=agent,
            observer_backend=obs,
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        for _ in range(4):
            p.chat("q")
        assert len(obs.calls) == 1


# ── ingest() ─────────────────────────────────────────────────────────────────

class TestIngest:

    def test_pushes_pair_without_calling_agent(self, tmp_path):
        agent = FakeBackend()
        p = Pipeline(
            backend=agent,
            observer_backend=FakeBackend(responses=[_OBS_REPLY] * 5),
            curator_backend=FakeBackend(responses=[_CURATOR_OK] * 5),
            chroma_path=str(tmp_path),
        )
        p.ingest("my question", "my answer")
        assert len(agent.calls) == 0

    def test_pair_appears_in_recent_messages(self, tmp_path):
        p = _pipeline(tmp_path)
        p.ingest("q", "a")
        assert len(p.bucket.recent_messages) == 1
        assert p.bucket.recent_messages[0]["question"] == "q"
        assert p.bucket.recent_messages[0]["response"] == "a"

    def test_ingest_many_pairs(self, tmp_path):
        obs = FakeBackend(responses=[_OBS_REPLY] * 20)
        cur = FakeBackend(responses=[_CURATOR_OK] * 20)
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=obs,
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        for i in range(10):
            p.ingest(f"q{i}", f"a{i}")
        # Eviction fires every time the 4th pair is pushed after a fresh stack
        assert len(obs.calls) > 0

    def test_ingest_runs_observer_on_eviction(self, tmp_path):
        obs = FakeBackend(responses=[_OBS_REPLY] * 10)
        cur = FakeBackend(responses=[_CURATOR_OK] * 10)
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=obs,
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        for i in range(4):
            p.ingest(f"q{i}", f"a{i}")
        assert len(obs.calls) == 1

    def test_ingest_runs_curator_on_eviction(self, tmp_path):
        cur = FakeBackend(responses=[_CURATOR_OK] * 10)
        p = Pipeline(
            backend=FakeBackend(),
            observer_backend=FakeBackend(responses=[_OBS_REPLY] * 5),
            curator_backend=cur,
            chroma_path=str(tmp_path),
            max_recent_messages=3,
            batch_reduction=2,
        )
        for i in range(4):
            p.ingest(f"q{i}", f"a{i}")
        assert len(cur.calls) == 1


# ── build_context / update manual loop ───────────────────────────────────────

class TestManualLoop:

    def test_build_context_then_update_full_cycle(self, tmp_path):
        """The developer drop-in pattern should work end-to-end."""
        p = _pipeline(tmp_path)
        msgs = p.build_context("hello")
        # Developer makes their own call here
        p.update("hello", "world from custom client")
        assert len(p.bucket.recent_messages) == 1
        pair = p.bucket.recent_messages[0]
        assert pair["question"] == "hello"
        assert pair["response"] == "world from custom client"

    def test_build_context_messages_match_what_agent_would_see(self, tmp_path):
        """build_context() returns the same messages ActiveAgent.respond() builds.

        Note: build_context() calls retrieval.update_bucket() which overwrites
        the memories slot from ChromaDB (empty in this test → no memories).
        Only the summary — which is not touched by retrieval — is checked here.
        """
        p = _pipeline(tmp_path)
        p.bucket.set_summary("Running summary.")
        msgs = p.build_context("new question")
        system_text = msgs[0]["content"]
        assert "Running summary." in system_text
        assert msgs[1]["content"] == "new question"


# ── Properties ────────────────────────────────────────────────────────────────

class TestProperties:

    def test_bucket_property_returns_bucket_instance(self, tmp_path):
        p = _pipeline(tmp_path)
        assert isinstance(p.bucket, Bucket)

    def test_retrieval_property_returns_retrieval_instance(self, tmp_path):
        p = _pipeline(tmp_path)
        assert isinstance(p.retrieval, Retrieval)

    def test_repr_contains_backend(self, tmp_path):
        p = _pipeline(tmp_path)
        r = repr(p)
        assert "Pipeline(" in r
        assert "FakeBackend" in r
