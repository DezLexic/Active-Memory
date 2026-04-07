"""
librarian.py

The sleep cycle of Active Memory — runs during downtime to maintain the
health of the memory store by promoting, demoting, and pruning memories
based on usage patterns.  No model calls.

Usage:
    librarian = Librarian(retrieval)
    librarian.run_consolidation()
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .retrieval import Retrieval


class Librarian:
    """
    Maintains memory store health.  All decisions are based on metadata
    (retrieval_count, timestamp, tier) — no embeddings or model calls.

    Parameters
    ----------
    retrieval   A Retrieval instance that owns the warm and cold collections.
    """

    def __init__(self, retrieval: Retrieval) -> None:
        self._retrieval = retrieval

    # ── Public API ────────────────────────────────────────────────────────────

    def promote_frequent(self, retrieval_threshold: int = 3) -> int:
        """
        Move cold memories that have been retrieved often back to warm.

        A cold memory that has been retrieved retrieval_threshold or more
        times is proving its value — promote it so it gets queried first.

        Parameters
        ----------
        retrieval_threshold  Minimum retrieval_count to trigger promotion.
                             Default 3.

        Returns count of memories promoted.
        """
        memories = self._retrieval.get_all_metadata()
        promoted = 0
        for m in memories:
            if m["tier"] == "cold" and m["retrieval_count"] >= retrieval_threshold:
                self._retrieval.move_to_warm(m["id"])
                promoted += 1
        return promoted

    def demote_stale(
        self,
        days_threshold: int = 30,
        min_retrieval_count: int = 1,
    ) -> int:
        """
        Move warm memories that have not been used recently to cold.

        A warm memory is stale if it is older than days_threshold AND has
        been retrieved fewer than min_retrieval_count times.  These are
        taking up warm space without paying their way.

        Parameters
        ----------
        days_threshold       Age in days before a warm memory is considered
                             stale.  Default 30.
        min_retrieval_count  Minimum retrieval count to avoid demotion.
                             Default 1 (never retrieved → demote).

        Returns count of memories demoted.
        """
        memories = self._retrieval.get_all_metadata()
        cutoff   = datetime.now(timezone.utc) - timedelta(days=days_threshold)
        demoted  = 0
        for m in memories:
            if m["tier"] != "warm":
                continue
            ts = datetime.fromisoformat(m["timestamp"])
            if ts < cutoff and m["retrieval_count"] < min_retrieval_count:
                self._retrieval.move_to_cold(m["id"])
                demoted += 1
        return demoted

    def prune_old(self, days_threshold: int = 180) -> int:
        """
        Permanently delete memories that are very old and have never been used.

        A memory that has sat for days_threshold days without a single
        retrieval is unlikely to ever be useful.  Delete it from whichever
        collection owns it.

        Parameters
        ----------
        days_threshold  Age in days beyond which a never-retrieved memory is
                        pruned.  Default 180.

        Returns count of memories pruned.
        """
        memories = self._retrieval.get_all_metadata()
        cutoff   = datetime.now(timezone.utc) - timedelta(days=days_threshold)
        pruned   = 0
        for m in memories:
            ts = datetime.fromisoformat(m["timestamp"])
            if ts < cutoff and m["retrieval_count"] == 0:
                collection = (
                    self._retrieval._warm
                    if m["tier"] == "warm"
                    else self._retrieval._cold
                )
                collection.delete(ids=[m["id"]])
                pruned += 1
        return pruned

    def run_consolidation(
        self,
        promote_threshold:    int = 3,
        stale_days:           int = 30,
        stale_min_retrievals: int = 1,
        prune_days:           int = 180,
    ) -> dict:
        """
        Run all three maintenance passes in sequence and print a summary.

        Order: promote first (so freshly promoted memories are not
        immediately re-evaluated by demote_stale), then demote, then prune.

        Returns
        -------
        dict with keys: promoted, demoted, pruned, warm_remaining,
        cold_remaining.
        """
        promoted = self.promote_frequent(promote_threshold)
        demoted  = self.demote_stale(stale_days, stale_min_retrievals)
        pruned   = self.prune_old(prune_days)

        warm_remaining = self._retrieval._warm.count()
        cold_remaining = self._retrieval._cold.count()

        print(
            f"[librarian] consolidation complete — "
            f"promoted={promoted}  demoted={demoted}  pruned={pruned}  "
            f"warm={warm_remaining}  cold={cold_remaining}"
        )

        return {
            "promoted":       promoted,
            "demoted":        demoted,
            "pruned":         pruned,
            "warm_remaining": warm_remaining,
            "cold_remaining": cold_remaining,
        }

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Librarian(retrieval={self._retrieval!r})"
