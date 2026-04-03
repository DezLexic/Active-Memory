"""
test_observer.py

Pushes 7 message pairs through a max-5 Bucket, which evicts pairs 1 and 2
in turns 6 and 7.  After each push the evicted pair (if any) is forwarded
to Observer.update().  The evolving summary is printed after each update
so the preservation of decisions and constraints is clearly visible.

Message topics
--------------
  1. Language choice        -> Decision: Elixir / Phoenix
  2. Database choice        -> Decision: CockroachDB, raw SQL only
  3. Authentication         -> Decision: magic-link, no passwords
  4. Deployment target      -> Decision: GKE, 3-replica minimum
  5. Caching layer          -> Decision: Redis Cluster, no in-process cache
  6. Router structure       (fills stack, evicts pair 1 - Elixir/Phoenix)
  7. Pagination strategy    (fills stack, evicts pair 2 - CockroachDB)

After turn 6 the summary should capture the Elixir/Phoenix decision.
After turn 7 the summary should extend to also capture CockroachDB + raw SQL.
"""

import os
os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

from bucket   import Bucket
from observer import Observer

# ── Setup ─────────────────────────────────────────────────────────────────────

bucket   = Bucket(max_recent=5)
observer = Observer(model="gemma3:4b", max_words=300)

print("=" * 60)
print(f"  {observer!r}")
print(f"  {bucket!r}")
print("=" * 60)

# ── Message pairs ─────────────────────────────────────────────────────────────

messages = [
    (
        "What language and framework are we building with?",
        "Elixir with the Phoenix framework. We chose it for BEAM concurrency, "
        "OTP fault tolerance, and Phoenix LiveView for real-time UI without a "
        "JavaScript SPA.",
    ),
    (
        "Which database are we using and what is the query policy?",
        "CockroachDB as the primary database. It is distributed by default and "
        "PostgreSQL-wire-compatible. Raw SQL only -- no Ecto ORM, no query "
        "builders anywhere in the codebase.",
    ),
    (
        "How do users authenticate?",
        "Magic-link authentication only. No passwords, no OAuth. Single-use "
        "tokens stored in Redis with a 15-minute TTL. This removes credential "
        "storage liability entirely.",
    ),
    (
        "Where are we deploying and what is the replica policy?",
        "Google Kubernetes Engine. Minimum three replicas at all times. "
        "Horizontal pod autoscaler is configured but the floor is three.",
    ),
    (
        "What handles caching and are there any restrictions?",
        "Redis Cluster for all caching. No in-process caching is allowed -- "
        "divergent pod caches cause stale-read bugs. Redis provides one shared "
        "cache layer across all Kubernetes replicas.",
    ),
    (
        "How should we structure the Phoenix router?",
        "Three pipeline scopes: browser for server-rendered pages, api for "
        "JSON endpoints, and live for LiveView routes. Each scope has its own "
        "plug stack.",
    ),
    (
        "How do we paginate large product listings?",
        "Cursor-based pagination using a composite cursor of (created_at, id). "
        "All indexes are defined in raw SQL migration files -- no schema "
        "macros.",
    ),
]

# ── Push pairs and update observer on eviction ────────────────────────────────

for i, (question, response) in enumerate(messages, 1):
    evicted = bucket.push_message(question, response)

    if evicted is not None:
        print(f"\n  -- Turn {i}: eviction triggered --")
        print(f"  Evicted Q: {evicted['question']}")
        print(f"  Evicted A: {evicted['response'][:80].rstrip()}...")
        print(f"\n  Calling observer.update() ...")

        observer.update(bucket, evicted)

        print(f"\n  Updated summary ({len(bucket.summary.split())} words):")
        print(f"  {'-' * 56}")
        # Print summary with light indentation
        for line in bucket.summary.splitlines():
            print(f"  {line}")
        print(f"  {'-' * 56}")
    else:
        print(f"\n  Turn {i}: pushed  (stack {len(bucket.recent_messages)}/5 -- no eviction)")
        print(f"  Q: {question}")

# ── Final bucket state ────────────────────────────────────────────────────────

print("\n\n")
print("=" * 60)
print("  FINAL BUCKET STATE")
print("=" * 60)
print(f"\n  {bucket!r}\n")
print(bucket.to_context_string())
