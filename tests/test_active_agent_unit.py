"""
test_active_agent_unit.py

Unit tests for ActiveAgent.respond() and __repr__.

Uses FakeBackend from conftest.py — zero network calls.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from active_memory.active_agent import ActiveAgent
from active_memory.bucket import Bucket
from tests.conftest import FakeBackend


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bucket(prompt="Hello", summary="", memories=None, recent=None):
    """Build a Bucket pre-loaded with the given slots."""
    b = Bucket(max_recent=20, batch_reduction=10)
    b.set_current_prompt(prompt)
    if summary:
        b.set_summary(summary)
    if memories:
        b.set_memories(memories)
    if recent:
        for pair in recent:
            b.push_message(pair["question"], pair["response"])
    return b


# ── TestRespond ──────────────────────────────────────────────────────────────

class TestRespond:

    def test_returns_backend_response(self):
        backend = FakeBackend(responses=["Hello!"])
        agent = ActiveAgent(backend=backend)
        bucket = _bucket(prompt="Hi")
        result = agent.respond(bucket)
        assert result == "Hello!"

    def test_sends_system_and_user_messages(self):
        backend = FakeBackend(responses=["ok"])
        agent = ActiveAgent(backend=backend)
        bucket = _bucket(prompt="test prompt")
        agent.respond(bucket)
        assert len(backend.calls) == 1
        messages = backend.calls[0]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_is_context_string(self):
        backend = FakeBackend(responses=["ok"])
        agent = ActiveAgent(backend=backend)
        bucket = _bucket(
            prompt="anything",
            summary="We discussed Python decorators",
            memories=[{
                "content": "User prefers dark mode",
                "similarity": 0.85,
                "tier": "warm",
                "retrieval_count": 1,
            }],
        )
        agent.respond(bucket)
        system_content = backend.calls[0][0]["content"]
        assert "Python decorators" in system_content
        assert "User prefers dark mode" in system_content

    def test_user_message_is_current_prompt(self):
        backend = FakeBackend(responses=["ok"])
        agent = ActiveAgent(backend=backend)
        bucket = _bucket(prompt="What is the meaning of life?")
        agent.respond(bucket)
        user_content = backend.calls[0][1]["content"]
        assert user_content == "What is the meaning of life?"

    def test_exactly_one_backend_call(self):
        backend = FakeBackend(responses=["reply"])
        agent = ActiveAgent(backend=backend)
        bucket = _bucket(prompt="question")
        agent.respond(bucket)
        assert len(backend.calls) == 1

    def test_empty_prompt_still_works(self):
        backend = FakeBackend(responses=["still works"])
        agent = ActiveAgent(backend=backend)
        bucket = _bucket(prompt="")
        result = agent.respond(bucket)
        assert result == "still works"


# ── TestRepr ─────────────────────────────────────────────────────────────────

class TestRepr:

    def test_repr_contains_backend(self):
        backend = FakeBackend()
        agent = ActiveAgent(backend=backend)
        r = repr(agent)
        assert "FakeBackend" in r
