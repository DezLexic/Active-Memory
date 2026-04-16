"""
test_curator_unit.py

Unit tests for Curator.evaluate() — JSON parsing edge cases, store/drop
decisions, and error-path resilience.

No real LLM calls.  Uses FakeBackend (from conftest.py) and real ChromaDB
via pytest tmp_path.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from active_memory.curator import Curator
from active_memory.retrieval import Retrieval

from tests.conftest import FakeBackend


_CLEAR_DECISION = {
    "question": "Which database are we using?",
    "response":  "PostgreSQL. That is locked in — no revisiting.",
}

_AMBIGUOUS = {
    "question": "What do you think about databases?",
    "response":  "Databases are important.",
}


def _curator(tmp_path, response: str, *, use_batch_mode: bool = True):
    retrieval = Retrieval(chroma_path=str(tmp_path))
    backend   = FakeBackend(responses=[response] * 10)
    return Curator(backend=backend, retrieval=retrieval, use_batch_mode=use_batch_mode), retrieval


# ── store=true ────────────────────────────────────────────────────────────────

class TestStoreTrue:

    def test_stores_memory_when_store_is_true(self, tmp_path):
        curator, retrieval = _curator(
            tmp_path,
            '{"store": true, "reason": "explicit decision", "tier": "warm"}'
        )
        curator.evaluate(_CLEAR_DECISION)
        # Use get_all_metadata() — retrieve() applies similarity thresholds
        # which may filter results depending on embedding distance.
        all_meta = retrieval.get_all_metadata()
        assert len(all_meta) == 1

    def test_stored_memory_contains_question_and_response(self, tmp_path):
        curator, retrieval = _curator(
            tmp_path,
            '{"store": true, "reason": "decision", "tier": "warm"}'
        )
        curator.evaluate(_CLEAR_DECISION)
        all_meta = retrieval.get_all_metadata()
        assert len(all_meta) == 1
        assert "PostgreSQL" in all_meta[0]["content"]

    def test_sets_last_store_true(self, tmp_path):
        curator, _ = _curator(
            tmp_path,
            '{"store": true, "reason": "decision", "tier": "warm"}'
        )
        curator.evaluate(_CLEAR_DECISION)
        assert curator._last_store is True

    def test_sets_last_reason(self, tmp_path):
        curator, _ = _curator(
            tmp_path,
            '{"store": true, "reason": "explicit decision", "tier": "warm"}'
        )
        curator.evaluate(_CLEAR_DECISION)
        assert curator._last_reason == "explicit decision"

    def test_warm_tier_stored_in_warm_collection(self, tmp_path):
        curator, retrieval = _curator(
            tmp_path,
            '{"store": true, "reason": "decision", "tier": "warm"}'
        )
        curator.evaluate(_CLEAR_DECISION)
        meta = retrieval.get_all_metadata()
        warm = [m for m in meta if m["tier"] == "warm"]
        assert len(warm) == 1

    def test_cold_tier_stored_in_cold_collection(self, tmp_path):
        curator, retrieval = _curator(
            tmp_path,
            '{"store": true, "reason": "background context", "tier": "cold"}'
        )
        curator.evaluate(_CLEAR_DECISION)
        meta = retrieval.get_all_metadata()
        cold = [m for m in meta if m["tier"] == "cold"]
        assert len(cold) == 1


# ── store=false ───────────────────────────────────────────────────────────────

class TestStoreFalse:

    def test_nothing_stored_when_store_is_false(self, tmp_path):
        curator, retrieval = _curator(
            tmp_path,
            '{"store": false, "reason": "too vague", "tier": "warm"}'
        )
        curator.evaluate(_AMBIGUOUS)
        memories = retrieval.retrieve("databases")
        assert len(memories) == 0

    def test_sets_last_store_false(self, tmp_path):
        curator, _ = _curator(
            tmp_path,
            '{"store": false, "reason": "too vague", "tier": "warm"}'
        )
        curator.evaluate(_AMBIGUOUS)
        assert curator._last_store is False


# ── JSON parsing edge cases ───────────────────────────────────────────────────

class TestJsonParsing:

    def test_strips_markdown_fences_before_parsing(self, tmp_path):
        fenced = '```json\n{"store": true, "reason": "decision", "tier": "warm"}\n```'
        curator, retrieval = _curator(tmp_path, fenced)
        curator.evaluate(_CLEAR_DECISION)   # should not raise
        assert len(retrieval.get_all_metadata()) == 1

    def test_strips_plain_code_fences(self, tmp_path):
        fenced = '```\n{"store": true, "reason": "decision", "tier": "warm"}\n```'
        curator, retrieval = _curator(tmp_path, fenced)
        curator.evaluate(_CLEAR_DECISION)
        assert len(retrieval.get_all_metadata()) == 1

    def test_handles_array_response_takes_first_element(self, tmp_path):
        array_resp = '[{"store": true, "reason": "decision", "tier": "warm"}]'
        curator, retrieval = _curator(tmp_path, array_resp)
        curator.evaluate(_CLEAR_DECISION)
        assert len(retrieval.get_all_metadata()) == 1

    def test_malformed_json_stores_as_cold_not_drop(self, tmp_path):
        """Parse failure must store as cold — never silently drop the pair."""
        curator, retrieval = _curator(tmp_path, "not valid json at all")
        curator.evaluate(_CLEAR_DECISION)   # must not raise
        meta = retrieval.get_all_metadata()
        assert len(meta) == 1
        assert meta[0]["tier"] == "cold"
        assert curator._last_store is True
        assert curator._last_tier == "cold"

    def test_invalid_tier_defaults_to_cold(self, tmp_path):
        """An unrecognised tier string should fall back to cold, not warm."""
        curator, retrieval = _curator(
            tmp_path,
            '{"store": true, "reason": "decision", "tier": "unknown"}'
        )
        curator.evaluate(_CLEAR_DECISION)
        meta = retrieval.get_all_metadata()
        assert meta[0]["tier"] == "cold"

    def test_missing_tier_field_defaults_to_cold(self, tmp_path):
        """Missing tier key should default to cold."""
        curator, retrieval = _curator(
            tmp_path,
            '{"store": true, "reason": "decision"}'
        )
        curator.evaluate(_CLEAR_DECISION)
        meta = retrieval.get_all_metadata()
        assert meta[0]["tier"] == "cold"

    def test_empty_array_response_stores_as_cold(self, tmp_path):
        """Empty JSON array → parsed={} → store=True (default), tier=cold."""
        curator, retrieval = _curator(tmp_path, "[]")
        curator.evaluate(_CLEAR_DECISION)
        meta = retrieval.get_all_metadata()
        assert len(meta) == 1
        assert meta[0]["tier"] == "cold"


# ── retrieval=None ────────────────────────────────────────────────────────────

class TestNoRetrieval:

    def test_evaluate_without_retrieval_does_not_crash(self, tmp_path):
        backend = FakeBackend(
            responses=['{"store": true, "reason": "decision", "tier": "warm"}']
        )
        curator = Curator(backend=backend, retrieval=None)
        curator.evaluate(_CLEAR_DECISION)  # must not raise even with store=true


# ── LLM call failure ─────────────────────────────────────────────────────────

class TestLLMFailure:

    def test_exception_in_backend_stores_as_cold(self, tmp_path):
        """LLM call failure must store as cold — never silently drop the pair."""
        class ErrorBackend:
            def chat(self, messages):
                raise RuntimeError("Connection refused")
            def __repr__(self):
                return "ErrorBackend()"

        retrieval = Retrieval(chroma_path=str(tmp_path))
        curator   = Curator(backend=ErrorBackend(), retrieval=retrieval)
        curator.evaluate(_CLEAR_DECISION)  # must not raise
        meta = retrieval.get_all_metadata()
        assert len(meta) == 1
        assert meta[0]["tier"] == "cold"
        assert curator._last_store is True
        assert curator._last_tier == "cold"


# ── use_batch_mode flag ───────────────────────────────────────────────────────

class TestBatchModeFlag:

    def test_use_batch_mode_default_is_true(self, tmp_path):
        curator, _ = _curator(tmp_path, '{"store": false, "reason": "x", "tier": "warm"}')
        assert curator.use_batch_mode is True

    def test_use_batch_mode_can_be_set_false(self, tmp_path):
        retrieval = Retrieval(chroma_path=str(tmp_path))
        backend   = FakeBackend(responses=['{"store": false, "reason": "x", "tier": "warm"}'] * 5)
        curator   = Curator(backend=backend, retrieval=retrieval, use_batch_mode=False)
        assert curator.use_batch_mode is False
