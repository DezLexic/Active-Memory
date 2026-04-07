"""
retrieval.py

Manages the memory slot in the Bucket using ChromaDB.

No model calls.  ChromaDB's default embedding function handles vectorisation
internally.  The only external dependency is chromadb.

Distance vs similarity
----------------------
Chroma returns distances, not similarities.  This class uses cosine distance
(configured on the collection) where:

    distance = 1 - cosine_similarity

so the inverse conversion is exact:

    similarity = 1 - distance

A similarity of 1.0 is a perfect match; 0.0 is orthogonal.
"""

from __future__ import annotations

import uuid
import chromadb
from .bucket import Bucket

_COLLECTION_NAME = "active_memory"


class Retrieval:
    """
    Stores and retrieves memories from a local Chroma vector store.

    Parameters
    ----------
    chroma_path          Path to the persistent Chroma database directory.
    similarity_threshold Minimum cosine similarity (0-1) to accept a result.
                         Results below this value are discarded.  Default 0.7.
    max_results          Maximum number of memories injected into the Bucket.
                         Default 3.
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
        # Cosine distance so that similarity = 1 - distance is accurate.
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def store(self, content: str, timestamp: float) -> str:
        """
        Persist a memory string with its timestamp.

        Called by the Curator when it decides a trimmed message pair is worth
        keeping.  Returns the generated UUID for the stored memory.

        Parameters
        ----------
        content     The memory text to embed and store.
        timestamp   Unix epoch float.  Used for recency ranking during
                    retrieval when similarity scores are close.
        """
        memory_id = str(uuid.uuid4())
        self._collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[{"timestamp": float(timestamp)}],
        )
        return memory_id

    def retrieve(self, query: str) -> list[str]:
        """
        Search for memories relevant to `query`.

        Steps:
        1. Query Chroma for all candidates (up to a broad cap).
        2. Convert each distance to similarity = 1 - distance.
        3. Discard results whose similarity is below the threshold.
        4. Sort survivors by similarity descending; use timestamp descending
           as the tiebreaker so more recent memories surface first when
           similarity scores are close.
        5. Return the content strings of the top max_results entries.

        Returns an empty list if the store is empty or no result passes
        the threshold.
        """
        return [item["content"] for item in self._retrieve_scored(query)]

    def update_bucket(self, bucket: Bucket, query: str) -> None:
        """
        Retrieve relevant memories and inject them into the Bucket's memory
        slot.  This is the method the pipeline calls on every turn.
        """
        memories = self.retrieve(query)
        bucket.set_memories(memories)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _retrieve_scored(self, query: str) -> list[dict]:
        """
        Same pipeline as retrieve() but returns full detail dicts:

            {"content": str, "similarity": float, "timestamp": float}

        Used internally by retrieve() and exposed for testing / debugging.
        """
        count = self._collection.count()
        if count == 0:
            return []

        # Query enough candidates to filter meaningfully, but cap to avoid
        # pulling the entire store on a very large collection.
        n_candidates = min(count, max(self._max * 10, 20))

        raw = self._collection.query(
            query_texts=[query],
            n_results=n_candidates,
            include=["documents", "distances", "metadatas"],
        )

        documents  = raw["documents"][0]
        distances  = raw["distances"][0]
        metadatas  = raw["metadatas"][0]

        # Build scored records and apply threshold.
        scored: list[dict] = []
        for doc, dist, meta in zip(documents, distances, metadatas):
            similarity = 1.0 - dist
            if similarity >= self._threshold:
                scored.append({
                    "content":    doc,
                    "similarity": similarity,
                    "timestamp":  float(meta.get("timestamp", 0.0)),
                })

        # Primary sort: similarity descending.
        # Secondary sort: timestamp descending (recency tiebreaker).
        scored.sort(key=lambda x: (x["similarity"], x["timestamp"]), reverse=True)

        return scored[: self._max]

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Retrieval("
            f"threshold={self._threshold}, "
            f"max={self._max}, "
            f"stored={self._collection.count()})"
        )
