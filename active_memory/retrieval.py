"""
retrieval.py

Two-tier memory storage and retrieval using ChromaDB.

Warm memories  — recent decisions, active constraints, and preferences
                  likely to be needed soon.
Cold memories  — older context, background information, and decisions
                  unlikely to be revisited soon.

Retrieval queries warm first.  If warm results don't fill max_results, cold
is queried to fill remaining slots.  Combined results are sorted by timestamp
descending (most recent first) and capped at max_results.

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
                         Default 0.7.
    max_results          Maximum memories injected into the Bucket. Default 3.
    """

    def __init__(
        self,
        chroma_path: str,
        similarity_threshold: float = 0.7,
        max_results: int = 3,
    ) -> None:
        self._threshold  = similarity_threshold
        self._max        = max_results
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

    def retrieve(self, query: str) -> list[str]:
        """
        Search both collections for memories relevant to query.

        Steps
        -----
        1. Query warm collection first.
        2. If warm results < max_results, query cold to fill remaining slots.
        3. Filter both by similarity threshold.
        4. Sort combined results by timestamp descending (most recent first).
        5. Increment retrieval_count for every memory returned.
        6. Return content strings capped at max_results.
        """
        scored = self._retrieve_scored(query)
        self._increment_counts(scored)
        return [item["content"] for item in scored]

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
        limit: int,
    ) -> list[dict]:
        """
        Query a single Chroma collection, apply the similarity threshold, and
        return up to `limit` scored records sorted by similarity descending.

        Each record: {id, content, similarity, timestamp, tier, retrieval_count}
        """
        count = collection.count()
        if count == 0 or limit <= 0:
            return []

        n_candidates = min(count, max(limit * 10, 20))

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
        return scored[:limit]

    def _retrieve_scored(self, query: str) -> list[dict]:
        """
        Query warm then cold, merge, filter, and sort.

        Does NOT increment retrieval_count — that is the caller's
        responsibility (retrieve() does it; _retrieve_scored() is safe for
        inspection / debugging calls).

        Returns dicts with keys: id, content, similarity, timestamp, tier,
        retrieval_count.
        """
        warm_scored = self._query_collection(self._warm, query, self._max)
        remaining   = self._max - len(warm_scored)
        cold_scored = (
            self._query_collection(self._cold, query, remaining)
            if remaining > 0
            else []
        )

        combined = warm_scored + cold_scored
        if not combined:
            return []

        # Sort by timestamp descending — most recent first.
        # ISO-8601 strings sort lexicographically in chronological order.
        combined.sort(key=lambda x: x["timestamp"], reverse=True)
        return combined[: self._max]

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

    def __repr__(self) -> str:
        return (
            f"Retrieval("
            f"threshold={self._threshold}, "
            f"max={self._max}, "
            f"warm={self._warm.count()}, "
            f"cold={self._cold.count()})"
        )
