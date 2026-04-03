"""
test_bucket.py

Exercises the Bucket class by pushing 7 message pairs through a max-5 stack,
which triggers two evictions.  Prints each evicted pair as it is returned,
then prints the final to_context_string() output so the full formatted context
is visible.
"""

from bucket import Bucket

# ── Set up the bucket ─────────────────────────────────────────────────────────

bucket = Bucket(max_recent=5)

# Give it a summary and a couple of memories so the context string is
# fully populated when we print it at the end.
bucket.set_summary(
    "The team is designing a SaaS product using Elixir and Phoenix LiveView. "
    "They have agreed on CockroachDB with raw SQL, magic-link authentication, "
    "GKE Kubernetes with a three-replica minimum, and Redis Cluster for caching."
)

bucket.set_memories([
    "Decision: raw SQL only — Ecto is explicitly prohibited across the codebase.",
    "Decision: magic links for auth — no passwords, no OAuth providers.",
    "Decision: Redis Cluster for shared cache — no in-process caching allowed.",
])

bucket.set_current_prompt("What testing strategy did we agree on for the LiveView layer?")

# ── Push 7 pairs — pairs 1-5 fill the stack, pairs 6 and 7 each cause a pop ──

messages = [
    ("What language are we using?",                 "Elixir with Phoenix framework."),
    ("Which database?",                              "CockroachDB with raw SQL only."),
    ("How do users log in?",                         "Magic links — no passwords, no OAuth."),
    ("Where are we deploying?",                      "GKE Kubernetes, minimum three replicas."),
    ("What handles caching?",                        "Redis Cluster; no in-process caching."),
    ("How do we structure the Phoenix router?",      "Three pipeline scopes: browser, api, and live."),
    ("How do we paginate large product listings?",   "Cursor-based pagination using (created_at, id)."),
]

print("=" * 60)
print("  Pushing 7 message pairs (max stack size = 5)")
print("=" * 60)

for i, (question, response) in enumerate(messages, 1):
    evicted = bucket.push_message(question, response)
    if evicted is not None:
        print(f"\n  -- Pair {i}: EVICTION TRIGGERED --")
        print(f"  Popped  Q: {evicted['question']}")
        print(f"  Popped  A: {evicted['response']}")
        print(f"  Pushed  Q: {question}")
        print(f"  Pushed  A: {response}")
    else:
        print(f"\n  Pair {i} pushed  (stack now {len(bucket.recent_messages)}/5)")
        print(f"  Q: {question}")
        print(f"  A: {response}")

print(f"\n  Final stack size: {len(bucket.recent_messages)} / 5")
print(f"  Bucket repr: {bucket!r}")

# ── Print the fully assembled context string ──────────────────────────────────

print("\n\n")
print("=" * 60)
print("  ASSEMBLED CONTEXT STRING  (what the Active Agent sees)")
print("=" * 60)
print()
print(bucket.to_context_string())
