"""
retrieval.py

Two-tier memory storage and retrieval using ChromaDB.

Warm memories  — recent decisions, active constraints, and preferences
                  likely to be needed soon.
Cold memories  — older context, background information, and decisions
                  unlikely to be revisited soon.

Retrieval queries both warm and cold, filters by similarity threshold,
and returns all qualifying results sorted by similarity descending
(most relevant first).

Every memory returned by retrieve() has its retrieval_count metadata
incremented by 1 in the owning collection.

Distance vs similarity
----------------------
Chroma returns cosine distances.  similarity = 1 - distance.
A similarity of 1.0 is a perfect match; 0.0 is orthogonal.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb

from .bucket import Bucket

_WARM_COLLECTION = "warm_memories"
_COLD_COLLECTION = "cold_memories"


def _ts_to_iso(timestamp: float) -> str:
    """Convert a Unix epoch float to a UTC ISO-8601 string."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

class _CountProxy:
    """
    Returned by the _collection property so that existing code calling
    retrieval._collection.count() continues to work after the migration
    to two collections.
    """

    def __init__(self, retrieval: "Retrieval") -> None:
        self._r = retrieval

    def count(self) -> int:
        return self._r._warm.count() + self._r._cold.count()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class Retrieval:
    """
    Stores and retrieves memories across two Chroma collections.

    Parameters
    ----------
    chroma_path          Path to the persistent Chroma database directory.
    similarity_threshold Minimum cosine similarity (0-1) to accept a result.
                         Default 0.3 — calibrated for Chroma's built-in
                         MiniLM embedder on paraphrased queries. Raise for
                         stricter matching; lower to see more loose hits.
    max_results          Deprecated. Kept for backward compatibility but no
                         longer caps results.  All memories above
                         similarity_threshold are returned.
    """

    def __init__(
        self,
        chroma_path: str,
        similarity_threshold: float = 0.3,
        max_results: int = 3,
    ) -> None:
        self._threshold  = similarity_threshold
        self._max        = max_results  # Deprecated: kept for backward compat; no longer caps results.
        self._client     = chromadb.PersistentClient(path=chroma_path)
        self._warm       = self._client.get_or_create_collection(
            name=_WARM_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._cold       = self._client.get_or_create_collection(
            name=_COLD_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        # Backward-compat: code that calls _collection.count() still works.
        self._collection = _CountProxy(self)

    # ── Public API ────────────────────────────────────────────────────────────

    def store(
        self,
        content: str,
        timestamp: Optional[float] = None,
        tier: str = "warm",
    ) -> str:
        """
        Persist a memory string to the warm or cold collection.

        Parameters
        ----------
        content    Memory text to embed and store.
        timestamp  Unix epoch float.  Defaults to current UTC time when None.
        tier       "warm" or "cold".  Default "warm".

        Returns the generated UUID for the stored memory.
        """
        memory_id  = str(uuid.uuid4())
        ts_iso     = _ts_to_iso(timestamp) if timestamp is not None else _now_iso()
        collection = self._warm if tier == "warm" else self._cold
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[{
                "timestamp":       ts_iso,
                "retrieval_count": 0,
                "tier":            tier,
            }],
        )
        return memory_id

    def retrieve(self, query: str) -> list[dict]:
        """
        Search both collections for memories relevant to query.

        Steps
        -----
        1. Query warm and cold collections.
        2. Filter both by similarity threshold.
        3. Sort combined results by similarity descending (most relevant first).
        4. Increment retrieval_count for every memory returned.
        5. Return scored dicts with keys: content, similarity, tier, retrieval_count.
        """
        scored = self._retrieve_scored(query)
        self._increment_counts(scored)
        return [
            {
                "content":         item["content"],
                "similarity":      item["similarity"],
                "tier":            item["tier"],
                "retrieval_count": item["retrieval_count"],
            }
            for item in scored
        ]

    def update_bucket(self, bucket: Bucket, query: str) -> None:
        """Retrieve relevant memories and inject them into the Bucket."""
        bucket.set_memories(self.retrieve(query))

    def get_all_metadata(self) -> list[dict]:
        """
        Return all memories from both collections with their metadata.

        Each entry contains:
            id, content, tier, timestamp, retrieval_count

        Used by the Librarian for consolidation decisions.
        """
        results: list[dict] = []
        for collection, default_tier in ((self._warm, "warm"), (self._cold, "cold")):
            if collection.count() == 0:
                continue
            data = collection.get(include=["documents", "metadatas"])
            for doc_id, doc, meta in zip(
                data["ids"], data["documents"], data["metadatas"]
            ):
                results.append({
                    "id":              doc_id,
                    "content":         doc,
                    "tier":            meta.get("tier", default_tier),
                    "timestamp":       meta.get("timestamp", ""),
                    "retrieval_count": int(meta.get("retrieval_count", 0)),
                })
        return results

    def move_to_warm(self, memory_id: str) -> None:
        """
        Move a memory from the cold collection to warm.

        Retrieves the entry by ID, writes it to warm with tier updated to
        "warm", then deletes it from cold.

        Raises ValueError if memory_id is not found in cold.
        """
        self._move(memory_id, from_collection=self._cold, to_tier="warm")

    def move_to_cold(self, memory_id: str) -> None:
        """
        Move a memory from the warm collection to cold.

        Raises ValueError if memory_id is not found in warm.
        """
        self._move(memory_id, from_collection=self._warm, to_tier="cold")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _move(
        self,
        memory_id: str,
        from_collection,
        to_tier: str,
    ) -> None:
        """Shared implementation for move_to_warm / move_to_cold."""
        data = from_collection.get(
            ids=[memory_id],
            include=["documents", "metadatas"],
        )
        if not data["ids"]:
            from_tier = "cold" if to_tier == "warm" else "warm"
            raise ValueError(
                f"Memory {memory_id!r} not found in {from_tier} collection."
            )
        content      = data["documents"][0]
        meta         = dict(data["metadatas"][0])
        meta["tier"] = to_tier

        to_collection = self._warm if to_tier == "warm" else self._cold
        to_collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[meta],
        )
        from_collection.delete(ids=[memory_id])

    def _query_collection(
        self,
        collection,
        query: str,
        candidate_hint: int = 3,
    ) -> list[dict]:
        """
        Query a single Chroma collection, apply the similarity threshold, and
        return all scored records above threshold sorted by similarity descending.

        candidate_hint controls how many raw results Chroma fetches before
        filtering — it does NOT cap the final output.

        Each record: {id, content, similarity, timestamp, tier, retrieval_count}
        """
        count = collection.count()
        if count == 0:
            return []

        n_candidates = min(count, max(candidate_hint * 10, 20))

        raw = collection.query(
            query_texts=[query],
            n_results=n_candidates,
            include=["documents", "distances", "metadatas"],
        )

        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        distances = raw["distances"][0]
        metadatas = raw["metadatas"][0]

        scored: list[dict] = []
        for doc_id, doc, dist, meta in zip(ids, documents, distances, metadatas):
            similarity = 1.0 - dist
            if similarity >= self._threshold:
                scored.append({
                    "id":              doc_id,
                    "content":         doc,
                    "similarity":      similarity,
                    "timestamp":       meta.get("timestamp", ""),
                    "tier":            meta.get("tier", "warm"),
                    "retrieval_count": int(meta.get("retrieval_count", 0)),
                })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored

    def _retrieve_scored(self, query: str) -> list[dict]:
        """
        Query warm and cold, merge, filter by threshold, and sort.

        Does NOT increment retrieval_count — that is the caller's
        responsibility (retrieve() does it; _retrieve_scored() is safe for
        inspection / debugging calls).

        Returns all results above similarity_threshold, sorted by similarity
        descending.  No hard cap on result count.

        Each dict: {id, content, similarity, timestamp, tier, retrieval_count}
        """
        warm_scored = self._query_collection(self._warm, query)
        cold_scored = self._query_collection(self._cold, query)

        combined = warm_scored + cold_scored
        if not combined:
            return []

        # Sort by similarity descending — most relevant first.
        combined.sort(key=lambda x: x["similarity"], reverse=True)
        return combined

    def _increment_counts(self, items: list[dict]) -> None:
        """Increment retrieval_count metadata for each returned memory."""
        for item in items:
            collection = self._warm if item["tier"] == "warm" else self._cold
            new_count  = item["retrieval_count"] + 1
            # Re-fetch to get the latest metadata before updating.
            data = collection.get(
                ids=[item["id"]],
                include=["metadatas"],
            )
            if not data["ids"]:
                continue
            meta = dict(data["metadatas"][0])
            meta["retrieval_count"] = new_count
            collection.update(
                ids=[item["id"]],
                metadatas=[meta],
            )

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the ChromaDB client connection.

        On Windows the underlying SQLite file stays locked until the client
        is explicitly closed. Call this before deleting the chroma_path
        directory to avoid PermissionError [WinError 32].

        clear_system_cache() is also called so that the next PersistentClient
        pointed at the same path creates a fresh system instead of reusing the
        now-stale cached one (which would raise "Could not connect to tenant
        default_tenant" after the directory has been deleted).
        """
        try:
            self._client.close()
        except Exception:
            pass
        try:
            type(self._client).clear_system_cache()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"Retrieval("
            f"threshold={self._threshold}, "
            f"warm={self._warm.count()}, "
            f"cold={self._cold.count()})"
        )
