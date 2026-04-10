"""
tests/conftest.py

Shared fixtures and helpers for the Active Memory unit test suite.
"""

import pytest


class FakeBackend:
    """
    Fake LLM backend for unit tests — zero network calls.

    Accepts a list of pre-canned responses that are returned in order.
    After the list is exhausted every call returns a valid Curator JSON
    string so tests that don't care about the response don't crash.

    Attributes
    ----------
    calls   List of message lists received — inspect in assertions.
    """

    def __init__(self, responses=None):
        self.calls: list = []
        self._responses: list[str] = list(responses or [])
        self._idx: int = 0

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if self._idx < len(self._responses):
            result = self._responses[self._idx]
            self._idx += 1
            return result
        # Default: well-formed Curator JSON that tells it not to store anything.
        return '{"store": false, "reason": "unit test default", "tier": "warm"}'

    def __repr__(self) -> str:
        return f"FakeBackend(calls={len(self.calls)})"


@pytest.fixture
def fake_backend():
    """Return a fresh FakeBackend with no pre-canned responses."""
    return FakeBackend()
