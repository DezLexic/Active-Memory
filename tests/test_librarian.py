"""
tests/test_librarian.py

Tests for the Librarian class.  Each test gets an isolated Chroma DB via
the tmp_path fixture.  No LLM calls — Librarian is pure metadata logic.
"""

from __future__ import annotations

import time

import pytest

from active_memory.retrieval import Retrieval
from active_memory.librarian import Librarian


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _old_ts(days: int) -> float:
    """Unix epoch float for `days` days ago."""
    return time.time() - (days * 24 * 3600)


def _make_retrieval(tmp_path) -> Retrieval:
    return Retrieval(chroma_path=str(tmp_path), similarity_threshold=0.0)


def _set_retrieval_count(retrieval: Retrieval, memory_id: str, count: int, tier: str) -> None:
    """Directly update retrieval_count metadata on a stored memory."""
    collection = retrieval._warm if tier == "warm" else retrieval._cold
    data = collection.get(ids=[memory_id], include=["metadatas"])
    meta = dict(data["metadatas"][0])
    meta["retrieval_count"] = count
    collection.update(ids=[memory_id], metadatas=[meta])


# ---------------------------------------------------------------------------
# promote_frequent
# ---------------------------------------------------------------------------

def test_promote_frequent_moves_cold_above_threshold(tmp_path):
    """Cold memories with count >= threshold are moved to warm."""
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    # Store 3 cold memories with high retrieval count.
    for i in range(3):
        mid = retrieval.store(f"decision {i}", tier="cold")
        _set_retrieval_count(retrieval, mid, 5, "cold")

    # Store 2 cold memories below threshold — should stay cold.
    for i in range(2):
        mid = retrieval.store(f"background {i}", tier="cold")
        _set_retrieval_count(retrieval, mid, 1, "cold")

    promoted = librarian.promote_frequent(retrieval_threshold=3)

    assert promoted == 3
    assert retrieval._warm.count() == 3
    assert retrieval._cold.count() == 2


def test_promote_frequent_ignores_warm_memories(tmp_path):
    """promote_frequent never touches warm memories regardless of count."""
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    mid = retrieval.store("already warm", tier="warm")
    _set_retrieval_count(retrieval, mid, 10, "warm")

    promoted = librarian.promote_frequent(retrieval_threshold=3)

    assert promoted == 0
    assert retrieval._warm.count() == 1
    assert retrieval._cold.count() == 0


def test_promote_frequent_returns_zero_when_nothing_qualifies(tmp_path):
    """Cold memory below threshold is not promoted."""
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    mid = retrieval.store("low use", tier="cold")
    _set_retrieval_count(retrieval, mid, 1, "cold")

    assert librarian.promote_frequent(retrieval_threshold=3) == 0


# ---------------------------------------------------------------------------
# demote_stale
# ---------------------------------------------------------------------------

def test_demote_stale_moves_old_unused_warm_to_cold(tmp_path):
    """Warm memories older than threshold with count < min are demoted."""
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    # 3 stale warm memories: 40 days old, never retrieved.
    for i in range(3):
        mid = retrieval.store(f"stale {i}", timestamp=_old_ts(40), tier="warm")
        _set_retrieval_count(retrieval, mid, 0, "warm")

    # 2 warm memories with recent timestamp — should stay warm.
    for i in range(2):
        retrieval.store(f"recent {i}", tier="warm")

    demoted = librarian.demote_stale(days_threshold=30, min_retrieval_count=1)

    assert demoted == 3
    assert retrieval._warm.count() == 2
    assert retrieval._cold.count() == 3


def test_demote_stale_keeps_warm_memories_with_retrievals(tmp_path):
    """Old warm memories with retrieval_count >= min are not demoted."""
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    mid = retrieval.store("old but used", timestamp=_old_ts(60), tier="warm")
    _set_retrieval_count(retrieval, mid, 2, "warm")

    demoted = librarian.demote_stale(days_threshold=30, min_retrieval_count=1)

    assert demoted == 0
    assert retrieval._warm.count() == 1


def test_demote_stale_ignores_cold_memories(tmp_path):
    """Cold memories are never touched by demote_stale."""
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    mid = retrieval.store("cold old", timestamp=_old_ts(60), tier="cold")
    _set_retrieval_count(retrieval, mid, 0, "cold")

    assert librarian.demote_stale(days_threshold=30, min_retrieval_count=1) == 0


# ---------------------------------------------------------------------------
# prune_old
# ---------------------------------------------------------------------------

def test_prune_old_deletes_ancient_unretrieved_memories(tmp_path):
    """Memories older than threshold with count=0 are permanently deleted."""
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    # 4 old cold memories never retrieved — should be pruned.
    for i in range(4):
        mid = retrieval.store(f"ancient cold {i}", timestamp=_old_ts(200), tier="cold")
        _set_retrieval_count(retrieval, mid, 0, "cold")

    # 2 old cold memories that WERE retrieved — should survive.
    for i in range(2):
        mid = retrieval.store(f"old used {i}", timestamp=_old_ts(200), tier="cold")
        _set_retrieval_count(retrieval, mid, 1, "cold")

    pruned = librarian.prune_old(days_threshold=180)

    assert pruned == 4
    assert retrieval._cold.count() == 2


def test_prune_old_works_on_warm_collection_too(tmp_path):
    """Stale unretrieved warm memories are also pruned."""
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    mid = retrieval.store("ancient warm", timestamp=_old_ts(200), tier="warm")
    _set_retrieval_count(retrieval, mid, 0, "warm")

    pruned = librarian.prune_old(days_threshold=180)

    assert pruned == 1
    assert retrieval._warm.count() == 0


def test_prune_old_keeps_recent_unretrieved_memories(tmp_path):
    """Recent memories with count=0 are not pruned."""
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    retrieval.store("new cold", tier="cold")  # count=0 by default, very recent

    assert librarian.prune_old(days_threshold=180) == 0


# ---------------------------------------------------------------------------
# run_consolidation — full integration with 20 memories
# ---------------------------------------------------------------------------

def test_run_consolidation_correct_counts(tmp_path, capsys):
    """
    Full integration: 20 memories with known characteristics produce
    predictable promotion, demotion, and pruning counts.

    Memory layout:
      A — 5x cold,  count=5, recent      -> promoted to warm
      B — 3x cold,  count=0, recent      -> stays cold
      C — 4x warm,  count=0, 40 days old -> demoted to cold
      D — 4x warm,  count=3, recent      -> stays warm
      E — 4x cold,  count=0, 200 days    -> pruned

    After run_consolidation(promote=3, stale_days=30, stale_min=1, prune=180):
      promoted=5, demoted=4, pruned=4
      warm_remaining=9 (D=4 + A=5), cold_remaining=7 (B=3 + C=4)
    """
    retrieval = _make_retrieval(tmp_path)
    librarian = Librarian(retrieval)

    # Category A: cold, count=5, recent
    for i in range(5):
        mid = retrieval.store(f"cat-A decision {i}", tier="cold")
        _set_retrieval_count(retrieval, mid, 5, "cold")

    # Category B: cold, count=0, recent
    for i in range(3):
        retrieval.store(f"cat-B background {i}", tier="cold")

    # Category C: warm, count=0, 40 days old
    for i in range(4):
        mid = retrieval.store(f"cat-C stale {i}", timestamp=_old_ts(40), tier="warm")
        _set_retrieval_count(retrieval, mid, 0, "warm")

    # Category D: warm, count=3, recent
    for i in range(4):
        mid = retrieval.store(f"cat-D active {i}", tier="warm")
        _set_retrieval_count(retrieval, mid, 3, "warm")

    # Category E: cold, count=0, 200 days old
    for i in range(4):
        mid = retrieval.store(f"cat-E ancient {i}", timestamp=_old_ts(200), tier="cold")
        _set_retrieval_count(retrieval, mid, 0, "cold")

    assert retrieval._warm.count() + retrieval._cold.count() == 20

    result = librarian.run_consolidation(
        promote_threshold=3,
        stale_days=30,
        stale_min_retrievals=1,
        prune_days=180,
    )

    assert result["promoted"] == 5
    assert result["demoted"]  == 4
    assert result["pruned"]   == 4
    assert result["warm_remaining"] == 9   # D(4) + A(5)
    assert result["cold_remaining"] == 7   # B(3) + C(4)

    out = capsys.readouterr().out
    assert "[librarian]" in out
    assert "promoted=5" in out
    assert "demoted=4"  in out
    assert "pruned=4"   in out
