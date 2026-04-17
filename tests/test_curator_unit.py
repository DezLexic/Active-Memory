"""
test_curator_unit.py

Unit tests for Curator.evaluate() as a tier classifier.

No real LLM calls. Uses FakeBackend (from conftest.py) and real ChromaDB
via pytest tmp_path.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from active_memory.curator import Curator
from active_memory.retrieval import Retrieval

from tests.conftest import FakeBackend


_CLEAR_DECISION = {
    "question": "Which database are we using?",
    "response": "PostgreSQL. That is locked in - no revisiting.",
}


def _curator(tmp_path, response: str, *, use_batch_mode: bool = True):
    retrieval = Retrieval(chroma_path=str(tmp_path))
    backend = FakeBackend(responses=[response] * 10)
    return Curator(backend=backend, retrieval=retrieval, use_batch_mode=use_batch_mode), retrieval


class TestTierClassification:

    def test_returns_warm_tier(self, tmp_path):
        curator, retrieval = _curator(
            tmp_path,
            '{"reason": "explicit decision", "tier": "warm"}',
        )
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "warm"
        assert retrieval.get_all_metadata() == []

    def test_returns_cold_tier(self, tmp_path):
        curator, retrieval = _curator(
            tmp_path,
            '{"reason": "background context", "tier": "cold"}',
        )
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "cold"
        assert retrieval.get_all_metadata() == []

    def test_sets_last_reason(self, tmp_path):
        curator, _ = _curator(
            tmp_path,
            '{"reason": "explicit decision", "tier": "warm"}',
        )
        curator.evaluate(_CLEAR_DECISION)
        assert curator._last_reason == "explicit decision"

    def test_sets_last_tier(self, tmp_path):
        curator, _ = _curator(
            tmp_path,
            '{"reason": "explicit decision", "tier": "warm"}',
        )
        curator.evaluate(_CLEAR_DECISION)
        assert curator._last_tier == "warm"

    def test_keeps_last_store_true_for_callers(self, tmp_path):
        curator, _ = _curator(
            tmp_path,
            '{"reason": "explicit decision", "tier": "warm"}',
        )
        curator.evaluate(_CLEAR_DECISION)
        assert curator._last_store is True


class TestJsonParsing:

    def test_strips_markdown_fences_before_parsing(self, tmp_path):
        fenced = '```json\n{"reason": "decision", "tier": "warm"}\n```'
        curator, retrieval = _curator(tmp_path, fenced)
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "warm"
        assert retrieval.get_all_metadata() == []

    def test_strips_plain_code_fences(self, tmp_path):
        fenced = '```\n{"reason": "decision", "tier": "warm"}\n```'
        curator, retrieval = _curator(tmp_path, fenced)
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "warm"
        assert retrieval.get_all_metadata() == []

    def test_handles_array_response_takes_first_element(self, tmp_path):
        array_resp = '[{"reason": "decision", "tier": "warm"}]'
        curator, retrieval = _curator(tmp_path, array_resp)
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "warm"
        assert retrieval.get_all_metadata() == []

    def test_malformed_json_defaults_to_cold_without_storing(self, tmp_path):
        curator, retrieval = _curator(tmp_path, "not valid json at all")
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "cold"
        assert retrieval.get_all_metadata() == []
        assert curator._last_store is True
        assert curator._last_tier == "cold"

    def test_invalid_tier_defaults_to_cold(self, tmp_path):
        curator, retrieval = _curator(
            tmp_path,
            '{"reason": "decision", "tier": "unknown"}',
        )
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "cold"
        assert retrieval.get_all_metadata() == []

    def test_missing_tier_field_defaults_to_cold(self, tmp_path):
        curator, retrieval = _curator(
            tmp_path,
            '{"reason": "decision"}',
        )
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "cold"
        assert retrieval.get_all_metadata() == []

    def test_empty_array_response_defaults_to_cold(self, tmp_path):
        curator, retrieval = _curator(tmp_path, "[]")
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "cold"
        assert retrieval.get_all_metadata() == []


class TestNoRetrieval:

    def test_evaluate_without_retrieval_does_not_crash(self, tmp_path):
        backend = FakeBackend(responses=['{"reason": "decision", "tier": "warm"}'])
        curator = Curator(backend=backend, retrieval=None)
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "warm"


class TestLLMFailure:

    def test_exception_in_backend_defaults_to_cold_without_storing(self, tmp_path):
        class ErrorBackend:
            def chat(self, messages):
                raise RuntimeError("Connection refused")

            def __repr__(self):
                return "ErrorBackend()"

        retrieval = Retrieval(chroma_path=str(tmp_path))
        curator = Curator(backend=ErrorBackend(), retrieval=retrieval)
        result = curator.evaluate(_CLEAR_DECISION)
        assert result == "cold"
        assert retrieval.get_all_metadata() == []
        assert curator._last_store is True
        assert curator._last_tier == "cold"


class TestBatchModeFlag:

    def test_use_batch_mode_default_is_true(self, tmp_path):
        curator, _ = _curator(tmp_path, '{"reason": "x", "tier": "cold"}')
        assert curator.use_batch_mode is True

    def test_use_batch_mode_can_be_set_false(self, tmp_path):
        retrieval = Retrieval(chroma_path=str(tmp_path))
        backend = FakeBackend(responses=['{"reason": "x", "tier": "cold"}'] * 5)
        curator = Curator(backend=backend, retrieval=retrieval, use_batch_mode=False)
        assert curator.use_batch_mode is False
