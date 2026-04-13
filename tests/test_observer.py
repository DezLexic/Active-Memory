"""
test_observer.py

Pushes 7 message pairs through a Bucket configured so that a batch of 2
pairs is evicted each time the stack fills.  After each eviction the batch
is forwarded to Observer.update() in a single call.  The evolving summary
is printed after each update so the preservation of decisions and constraints
is clearly visible.
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

from active_memory.bucket    import Bucket
from active_memory.observer  import Observer
from active_memory.backends  import OllamaBackend

# max_recent=5, batch_reduction=2 — evicts 2 pairs at once when full
bucket   = Bucket(max_recent=5, batch_reduction=2)
observer = Observer(backend=OllamaBackend(model="gemma3:4b"), max_summary_length=150)

print("=" * 60)
print(f"  {observer!r}")
print(f"  {bucket!r}")
print("=" * 60)

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

for i, (question, response) in enumerate(messages, 1):
    evicted = bucket.push_message(question, response)

    if evicted is not None:
        print(f"\n  -- Turn {i}: eviction triggered ({len(evicted)} pair(s)) --")
        for j, pair in enumerate(evicted, 1):
            print(f"  [{j}] Evicted Q: {pair['question']}")
            print(f"  [{j}] Evicted A: {pair['response'][:80].rstrip()}...")
        print(f"\n  Calling observer.update() with {len(evicted)} pair(s)...")

        observer.update(bucket, evicted)

        print(f"\n  Topic tree ({len(bucket.topic_tree.get('topics', []))} topics):")
        print(json.dumps(bucket.topic_tree, indent=2))
        print(f"\n  Flattened summary:")
        print(bucket.get_summary_text())
    else:
        print(f"\n  Turn {i}: pushed  (stack {len(bucket.recent_messages)}/{bucket._max_recent} -- no eviction)")
        print(f"  Q: {question}")

print("\n\n")
print("=" * 60)
print("  FINAL BUCKET STATE")
print("=" * 60)
print(f"\n  {bucket!r}\n")
print(bucket.to_context_string())
