"""
test_retrieval_unit.py

Unit tests for the Retrieval class covering error paths, edge cases,
and boundary conditions not exercised by the existing test_retrieval.py
integration script.

Covers: move errors, empty collections, store defaults, retrieve format,
increment-counts race edge case, _CountProxy, and __repr__.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import shutil
import uuid
from datetime import datetime, timezone

import pytest
from active_memory.retrieval import Retrieval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_retrieval(tmp_path, threshold=0.0):
    """Return a Retrieval backed by a disposable Chroma directory."""
    return Retrieval(
        chroma_path=str(tmp_path / "chroma_db"),
        similarity_threshold=threshold,
    )


# ── TestMoveErrors ────────────────────────────────────────────────────────────

class TestMoveErrors:

    def test_move_to_warm_nonexistent_raises_value_error(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        r.store("something in warm", timestamp=1_700_000_000, tier="warm")
        with pytest.raises(ValueError, match="cold"):
            r.move_to_warm("nonexistent-id")

    def test_move_to_cold_nonexistent_raises_value_error(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        r.store("something in cold", timestamp=1_700_000_000, tier="cold")
        with pytest.raises(ValueError, match="warm"):
            r.move_to_cold("nonexistent-id")

    def test_move_to_warm_raises_with_id_in_message(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        bad_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with pytest.raises(ValueError, match=bad_id):
            r.move_to_warm(bad_id)


# ── TestEmptyCollections ──────────────────────────────────────────────────────

class TestEmptyCollections:

    def test_retrieve_on_empty_store_returns_empty_list(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        assert r.retrieve("anything") == []

    def test_retrieve_scored_on_empty_returns_empty_list(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        assert r._retrieve_scored("anything") == []

    def test_query_collection_on_empty_returns_empty_list(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        assert r._query_collection(r._warm, "anything") == []


# ── TestStoreDefaults ─────────────────────────────────────────────────────────

class TestStoreDefaults:

    def test_store_with_none_timestamp_uses_current_time(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        before = datetime.now(timezone.utc)
        mid = r.store("test memory", timestamp=None)
        after = datetime.now(timezone.utc)

        # Retrieve the raw metadata to inspect the timestamp.
        data = r._warm.get(ids=[mid], include=["metadatas"])
        ts_str = data["metadatas"][0]["timestamp"]
        stored_dt = datetime.fromisoformat(ts_str)

        assert before <= stored_dt <= after

    def test_store_with_explicit_timestamp(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        mid = r.store("test memory", timestamp=1_700_000_000)
        data = r._warm.get(ids=[mid], include=["metadatas"])
        ts_str = data["metadatas"][0]["timestamp"]
        expected_dt = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)
        assert ts_str == expected_dt.isoformat()

    def test_store_returns_uuid_string(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        mid = r.store("test memory")
        # uuid.UUID() raises ValueError for non-UUID strings.
        parsed = uuid.UUID(mid)
        assert str(parsed) == mid

    def test_store_defaults_to_warm_tier(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        mid = r.store("test memory")
        assert r._warm.count() == 1
        assert r._cold.count() == 0
        data = r._warm.get(ids=[mid], include=["metadatas"])
        assert data["metadatas"][0]["tier"] == "warm"


# ── TestRetrieveFormat ────────────────────────────────────────────────────────

class TestRetrieveFormat:

    def test_retrieve_returns_list_of_dicts(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        r.store("database decisions and SQL choices", timestamp=1_700_000_000)
        results = r.retrieve("database decisions")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert isinstance(results[0], dict)

    def test_retrieve_dict_has_required_keys(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        r.store("database decisions and SQL choices", timestamp=1_700_000_000)
        results = r.retrieve("database decisions")
        required = {"content", "similarity", "tier", "retrieval_count"}
        for item in results:
            assert required.issubset(item.keys())

    def test_retrieve_does_not_include_id_or_timestamp(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        r.store("database decisions and SQL choices", timestamp=1_700_000_000)
        results = r.retrieve("database decisions")
        for item in results:
            assert "id" not in item
            assert "timestamp" not in item

    def test_retrieve_sorted_by_similarity_descending(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        r.store("cats are wonderful pets", timestamp=1_700_000_000)
        r.store("dogs are loyal companions", timestamp=1_700_000_001)
        r.store("the weather is sunny today", timestamp=1_700_000_002)
        results = r.retrieve("cats and dogs as pets")
        if len(results) >= 2:
            sims = [item["similarity"] for item in results]
            assert sims == sorted(sims, reverse=True)


# ── TestIncrementCountsEdge ───────────────────────────────────────────────────

class TestIncrementCountsEdge:

    def test_increment_counts_handles_deleted_memory(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        mid = r.store("ephemeral memory about databases", timestamp=1_700_000_000)

        # Get a scored list referencing this memory.
        scored = r._retrieve_scored("databases")
        assert len(scored) >= 1

        # Delete the memory from the collection before incrementing.
        r._warm.delete(ids=[mid])

        # Should NOT crash -- the continue branch handles missing ids.
        r._increment_counts(scored)


# ── TestCountProxy ────────────────────────────────────────────────────────────

class TestCountProxy:

    def test_collection_count_returns_total(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        r.store("warm memory A", timestamp=1_700_000_000, tier="warm")
        r.store("warm memory B", timestamp=1_700_000_001, tier="warm")
        r.store("cold memory C", timestamp=1_700_000_002, tier="cold")
        assert r._collection.count() == 3
        assert r._collection.count() == r._warm.count() + r._cold.count()

    def test_collection_count_empty(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        assert r._collection.count() == 0


# ── TestRepr ──────────────────────────────────────────────────────────────────

class TestRepr:

    def test_repr_contains_threshold(self, tmp_path):
        r = _fresh_retrieval(tmp_path, threshold=0.42)
        assert "threshold=0.42" in repr(r)

    def test_repr_does_not_contain_max(self, tmp_path):
        r = _fresh_retrieval(tmp_path)
        assert "max=" not in repr(r)
