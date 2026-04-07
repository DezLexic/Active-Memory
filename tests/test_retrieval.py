"""
test_retrieval.py

Exercises the two-tier Retrieval class: warm_memories and cold_memories.

Test plan
---------
  1. Store 5 warm memories  (active decisions and constraints)
  2. Store 3 cold memories  (background context, older info)
  3. Verify collection counts (warm=5, cold=3)
  4. Run a query — confirm warm results surface first, cold fills remaining
  5. Run the same query again — confirm retrieval_count incremented for
     each returned memory
  6. move_to_cold — move one warm memory to cold; verify counts shift
  7. move_to_warm — move one cold memory to warm; verify counts shift
  8. get_all_metadata — print all memories from both tiers with metadata
  9. Final retrieval showing tier labels and current retrieval_counts
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import shutil
import time
from active_memory.retrieval import Retrieval
from active_memory.bucket    import Bucket

# ── Use a fresh store for every run ──────────────────────────────────────────

STORE_PATH = "./chroma_store_test_retrieval"
shutil.rmtree(STORE_PATH, ignore_errors=True)

retrieval = Retrieval(
    chroma_path=STORE_PATH,
    similarity_threshold=0.3,
    max_results=4,
)

# ── Fake timeline (Unix timestamps spread over ~10 days) ─────────────────────

DAY  = 86_400
BASE = 1_700_000_000

T = {
    "day0": BASE + 0 * DAY,
    "day1": BASE + 1 * DAY,
    "day2": BASE + 2 * DAY,
    "day3": BASE + 3 * DAY,
    "day4": BASE + 4 * DAY,
    "day5": BASE + 5 * DAY,
    "day6": BASE + 6 * DAY,
    "day7": BASE + 7 * DAY,
}

# ── 1. Store warm memories (active decisions) ─────────────────────────────────

print("=" * 68)
print("  1. Storing warm memories (active decisions / constraints)")
print("=" * 68)

warm_ids: list[str] = []

warm_memories = [
    (
        "Decision: use Elixir with the Phoenix framework. "
        "Chosen for BEAM concurrency, OTP fault tolerance, and LiveView "
        "for real-time UI without a JavaScript SPA.",
        T["day5"],
    ),
    (
        "Decision: CockroachDB as the primary database. "
        "Distributed by default, PostgreSQL-wire-compatible. "
        "Raw SQL only -- no Ecto ORM, no query builders anywhere in the codebase.",
        T["day6"],
    ),
    (
        "Decision: magic-link authentication only. "
        "No passwords, no OAuth. Single-use tokens stored in Redis with "
        "a 15-minute TTL. Removes credential storage liability entirely.",
        T["day6"],
    ),
    (
        "Constraint: minimum three GKE replicas at all times. "
        "Horizontal pod autoscaler is configured but the floor is three.",
        T["day7"],
    ),
    (
        "Constraint: Redis Cluster for all caching. "
        "No in-process caching -- divergent pod caches cause stale-read bugs.",
        T["day7"],
    ),
]

for content, ts in warm_memories:
    mid = retrieval.store(content, ts, tier="warm")
    warm_ids.append(mid)
    print(f"  [warm]  [{mid[:8]}...]  \"{content[:55].rstrip()}...\"")

print(f"\n  warm collection: {retrieval._warm.count()}  cold collection: {retrieval._cold.count()}")

# ── 2. Store cold memories (background / older context) ───────────────────────

print()
print("=" * 68)
print("  2. Storing cold memories (background context)")
print("=" * 68)

cold_ids: list[str] = []

cold_memories = [
    (
        "Project background: we are building a SaaS product for team "
        "collaboration. Expected initial user base is small-to-medium teams "
        "in the 10-200 seat range.",
        T["day0"],
    ),
    (
        "Early tech exploration: evaluated Elixir, Go, and Rust as primary "
        "language candidates. Elixir was preferred for its concurrency model "
        "and the existing team familiarity with functional programming.",
        T["day1"],
    ),
    (
        "Architecture discussion: considered monolith vs microservices. "
        "Agreed to start with a monolith and extract services only when "
        "a clear bottleneck justifies the split.",
        T["day2"],
    ),
]

for content, ts in cold_memories:
    mid = retrieval.store(content, ts, tier="cold")
    cold_ids.append(mid)
    print(f"  [cold]  [{mid[:8]}...]  \"{content[:55].rstrip()}...\"")

print(f"\n  warm collection: {retrieval._warm.count()}  cold collection: {retrieval._cold.count()}")
print(f"  Total via _collection.count(): {retrieval._collection.count()}")
assert retrieval._warm.count() == 5, "Expected 5 warm memories"
assert retrieval._cold.count() == 3, "Expected 3 cold memories"
print("  [PASS] counts correct (warm=5, cold=3)")

# ── 3. First retrieval — warm first, cold fills remaining ─────────────────────

QUERY = "what database and SQL decisions did we make?"

print()
print("=" * 68)
print(f"  3. First retrieval  (max_results={retrieval._max})")
print(f"  Query: \"{QUERY}\"")
print("=" * 68)

results_pass1 = retrieval._retrieve_scored(QUERY)

print(f"\n  {len(results_pass1)} result(s) before count increment:\n")
for i, item in enumerate(results_pass1, 1):
    print(
        f"  [{i}]  tier={item['tier']:<4}  "
        f"sim={item['similarity']:.4f}  "
        f"count={item['retrieval_count']}  "
        f"\"{item['content'][:50].rstrip()}...\""
    )

# Actually call retrieve() to trigger count increments.
_ = retrieval.retrieve(QUERY)
print(f"\n  retrieve() called — retrieval_counts should now be 1 for returned items.")

# ── 4. Second retrieval — counts should have incremented ─────────────────────

print()
print("=" * 68)
print(f"  4. Second retrieval — verify retrieval_count incremented")
print("=" * 68)

results_pass2 = retrieval._retrieve_scored(QUERY)

print(f"\n  {len(results_pass2)} result(s) — counts after first retrieve():\n")
all_incremented = True
for i, item in enumerate(results_pass2, 1):
    count = item["retrieval_count"]
    ok = count >= 1
    if not ok:
        all_incremented = False
    tag = "OK" if ok else "FAIL -- expected >= 1"
    print(
        f"  [{i}]  tier={item['tier']:<4}  "
        f"sim={item['similarity']:.4f}  "
        f"count={count}  [{tag}]  "
        f"\"{item['content'][:45].rstrip()}...\""
    )

assert all_incremented, "Some retrieval_counts did not increment"
print("\n  [PASS] all returned memories have retrieval_count >= 1")

# ── 5. move_to_cold — move a warm memory to cold ─────────────────────────────

target_warm_id = warm_ids[0]  # Elixir/Phoenix decision

print()
print("=" * 68)
print("  5. move_to_cold")
print(f"  Moving warm memory [{target_warm_id[:8]}...] to cold")
print("=" * 68)

warm_before = retrieval._warm.count()
cold_before = retrieval._cold.count()

retrieval.move_to_cold(target_warm_id)

warm_after = retrieval._warm.count()
cold_after = retrieval._cold.count()

print(f"\n  Before: warm={warm_before}  cold={cold_before}")
print(f"  After:  warm={warm_after}  cold={cold_after}")
assert warm_after == warm_before - 1, "Warm count should decrease by 1"
assert cold_after == cold_before + 1, "Cold count should increase by 1"

# Verify it's actually in cold now.
data = retrieval._cold.get(ids=[target_warm_id], include=["metadatas"])
assert data["ids"], "Memory should be present in cold collection"
assert data["metadatas"][0]["tier"] == "cold", "Tier metadata should be 'cold'"
print("  [PASS] memory moved to cold with tier='cold'")

# Verify it's gone from warm.
data_warm = retrieval._warm.get(ids=[target_warm_id])
assert not data_warm["ids"], "Memory should be absent from warm collection"
print("  [PASS] memory absent from warm collection")

# ── 6. move_to_warm — move a cold memory to warm ─────────────────────────────

target_cold_id = cold_ids[0]  # Project background

print()
print("=" * 68)
print("  6. move_to_warm")
print(f"  Moving cold memory [{target_cold_id[:8]}...] to warm")
print("=" * 68)

warm_before = retrieval._warm.count()
cold_before = retrieval._cold.count()

retrieval.move_to_warm(target_cold_id)

warm_after = retrieval._warm.count()
cold_after = retrieval._cold.count()

print(f"\n  Before: warm={warm_before}  cold={cold_before}")
print(f"  After:  warm={warm_after}  cold={cold_after}")
assert warm_after == warm_before + 1, "Warm count should increase by 1"
assert cold_after == cold_before - 1, "Cold count should decrease by 1"

# Verify tier metadata updated.
data = retrieval._warm.get(ids=[target_cold_id], include=["metadatas"])
assert data["ids"], "Memory should be present in warm collection"
assert data["metadatas"][0]["tier"] == "warm", "Tier metadata should be 'warm'"
print("  [PASS] memory moved to warm with tier='warm'")

# Verify absent from cold.
data_cold = retrieval._cold.get(ids=[target_cold_id])
assert not data_cold["ids"], "Memory should be absent from cold collection"
print("  [PASS] memory absent from cold collection")

# ── 7. get_all_metadata ───────────────────────────────────────────────────────

print()
print("=" * 68)
print("  7. get_all_metadata() — all memories from both collections")
print("=" * 68)

all_meta = retrieval.get_all_metadata()

print(f"\n  Total memories: {len(all_meta)}")
print(f"  Expected: {retrieval._warm.count() + retrieval._cold.count()}\n")
assert len(all_meta) == retrieval._warm.count() + retrieval._cold.count()

# Group by tier for display.
by_tier: dict[str, list] = {"warm": [], "cold": []}
for entry in all_meta:
    by_tier[entry["tier"]].append(entry)

for tier in ("warm", "cold"):
    print(f"  -- {tier.upper()} ({len(by_tier[tier])} memories) --")
    for entry in by_tier[tier]:
        print(
            f"    [{entry['id'][:8]}...]  "
            f"count={entry['retrieval_count']}  "
            f"ts={entry['timestamp'][:19]}  "
            f"\"{entry['content'][:45].rstrip()}...\""
        )
    print()

print("  [PASS] get_all_metadata returned correct total")

# ── 8. Final retrieval — show tier labels in results ─────────────────────────

FINAL_QUERY = "technology stack decisions and constraints"

print("=" * 68)
print(f"  8. Final retrieval")
print(f"  Query: \"{FINAL_QUERY}\"")
print("=" * 68)

final_scored = retrieval._retrieve_scored(FINAL_QUERY)

print(f"\n  {len(final_scored)} result(s) (max={retrieval._max}):\n")
if final_scored:
    for i, item in enumerate(final_scored, 1):
        print(f"  [{i}]  tier={item['tier']:<4}  "
              f"sim={item['similarity']:.4f}  "
              f"count={item['retrieval_count']}")
        words = item["content"].split()
        line  = "       "
        for word in words:
            if len(line) + len(word) + 1 > 68:
                print(line)
                line = "       " + word
            else:
                line += " " + word
        if line.strip():
            print(line)
        print()
else:
    print("  (no results passed the threshold)")

print("=" * 68)
print(f"  Done.  {retrieval!r}")
print("=" * 68)
