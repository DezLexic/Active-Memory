"""
test_retrieval.py

Stores six memories with timestamps spread across a fake timeline, then runs
a semantically focused query and prints full scored results so threshold
filtering and recency weighting are both visible.

Memory topics
-------------
  1. Elixir/Phoenix framework choice       (oldest)
  2. CockroachDB database decision         (older)
  3. Redis Cluster caching decision        (middle)
  4. Magic-link authentication decision    (recent)
  5. Database indexing strategy            (more recent)   <- should score high
  6. Weekly meal prep / cooking notes      (newest)        <- should be filtered out

Query: "what database and indexing decisions did we make?"

Expected behaviour
------------------
- Memories 2 and 5 should score highest (directly about databases/indexing).
- Memory 3 may score moderately (infrastructure, not directly database).
- Memory 1 may score low (language, not database).
- Memory 4 may score low (auth, not database).
- Memory 6 should fall below the threshold and be filtered out entirely.
- When two memories have similar similarity scores, the more recent timestamp
  (memory 5 > memory 2) should rank first.
"""

import shutil
import time
from retrieval import Retrieval
from bucket    import Bucket

# ── Use a temporary store so each test run starts clean ──────────────────────

STORE_PATH = "./chroma_store_test_retrieval"
shutil.rmtree(STORE_PATH, ignore_errors=True)   # wipe any previous run

retrieval = Retrieval(
    chroma_path=STORE_PATH,
    similarity_threshold=0.3,   # calibrated to Chroma's default embedding model;
    max_results=3,              # the cooking entry scores ~-0.02 and is still cut
)

# ── Fake timeline: Unix timestamps spaced one day apart ──────────────────────
# Day 0 = oldest, Day 5 = newest (today-ish)

DAY = 86_400   # seconds
BASE = 1_700_000_000   # arbitrary fixed base so output is deterministic

T = {
    "oldest":      BASE + 0 * DAY,
    "older":       BASE + 1 * DAY,
    "middle":      BASE + 2 * DAY,
    "recent":      BASE + 3 * DAY,
    "more_recent": BASE + 4 * DAY,
    "newest":      BASE + 5 * DAY,
}

memories = [
    (
        "Decision: use Elixir with the Phoenix framework. "
        "Chosen for BEAM concurrency, OTP fault tolerance, and LiveView "
        "for real-time UI without a JavaScript SPA.",
        T["oldest"],
    ),
    (
        "Decision: CockroachDB as the primary database. "
        "Distributed by default, PostgreSQL-wire-compatible. "
        "Raw SQL only — no Ecto ORM, no query builders.",
        T["older"],
    ),
    (
        "Decision: Redis Cluster for all caching. "
        "No in-process caching — divergent pod caches cause stale-read bugs. "
        "Redis provides a shared cache layer across all Kubernetes replicas.",
        T["middle"],
    ),
    (
        "Decision: magic-link authentication only. "
        "No passwords, no OAuth. Single-use tokens stored in Redis with a 15-minute TTL. "
        "Removes credential storage liability entirely.",
        T["recent"],
    ),
    (
        "Database indexing strategy: add a composite index on (user_id, created_at) "
        "for the products table. Use (created_at, id) as the cursor for pagination. "
        "All indexes defined in raw SQL migration files.",
        T["more_recent"],
    ),
    (
        "Weekly meal prep notes: roasted a full tray of vegetables, "
        "made a large batch of lentil soup, and portioned out lunches for the week. "
        "Added more chilli flakes this time.",
        T["newest"],
    ),
]

# ── Store all six memories ────────────────────────────────────────────────────

print("=" * 66)
print("  Storing memories")
print("=" * 66)
for content, ts in memories:
    memory_id = retrieval.store(content, ts)
    preview   = content[:55].rstrip() + "..."
    print(f"  [{memory_id[:8]}...]  ts={ts}  \"{preview}\"")

print(f"\n  Total stored: {retrieval._collection.count()}")
print(f"  {retrieval!r}\n")

# ── Run the query ─────────────────────────────────────────────────────────────

QUERY = "what database and indexing decisions did we make?"
print("=" * 66)
print(f"  Query: \"{QUERY}\"")
print(f"  Threshold: {retrieval._threshold}  (Chroma default embeddings produce scores ~0.0-0.40 for related content)")
print(f"  max_results: {retrieval._max}")
print("=" * 66)

# Use _retrieve_scored so we can print similarity and timestamp detail.
scored = retrieval._retrieve_scored(QUERY)

# Also get the raw candidates before filtering so we can show what was cut.
raw = retrieval._collection.query(
    query_texts=[QUERY],
    n_results=retrieval._collection.count(),
    include=["documents", "distances", "metadatas"],
)

print("\n  -- All candidates (before threshold filter) --\n")
all_candidates = []
for doc, dist, meta in zip(
    raw["documents"][0], raw["distances"][0], raw["metadatas"][0]
):
    sim = 1.0 - dist
    all_candidates.append((sim, float(meta.get("timestamp", 0)), doc))

# Sort same way retrieve does so the display order matches.
all_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

for sim, ts, doc in all_candidates:
    passed = "PASS" if sim >= retrieval._threshold else "FAIL"
    print(f"  [{passed}]  sim={sim:.4f}  ts={ts:.0f}  \"{doc[:55].rstrip()}...\"")

print(f"\n  -- Results returned by retrieve() (top {retrieval._max} after filter) --\n")
if scored:
    for i, item in enumerate(scored, 1):
        print(f"  [{i}]  sim={item['similarity']:.4f}  ts={item['timestamp']:.0f}")
        # Word-wrap the content at 60 chars
        words  = item["content"].split()
        line   = "       "
        for word in words:
            if len(line) + len(word) + 1 > 66:
                print(line)
                line = "       " + word
            else:
                line += " " + word
        if line.strip():
            print(line)
        print()
else:
    print("  (no results passed the threshold)")

# ── Also exercise update_bucket ───────────────────────────────────────────────

print("=" * 66)
print("  update_bucket() — inject results into a live Bucket")
print("=" * 66)

bucket = Bucket()
bucket.set_current_prompt(QUERY)
retrieval.update_bucket(bucket, QUERY)

print(f"\n  Bucket memories slot now holds {len(bucket.memories)} item(s):")
for i, mem in enumerate(bucket.memories, 1):
    print(f"\n  [{i}] {mem[:80].rstrip()}...")

print("\n\n  -- Bucket context string (memories section only) --\n")
ctx = bucket.to_context_string()
# Print just the RELEVANT MEMORIES section
start = ctx.find("RELEVANT MEMORIES")
end   = ctx.find("\n\n=", start)
print(ctx[start: end if end != -1 else len(ctx)])
