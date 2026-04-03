"""
test_active_agent.py

Manually assembles a Bucket with a summary, two recent message pairs, one
retrieved memory, and a current prompt -- then calls active_agent.respond()
and prints the response.

This verifies that the agent is reading every slot of the Bucket correctly
and producing a coherent response without any pipeline plumbing involved.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

from active_memory.bucket       import Bucket
from active_memory.active_agent import ActiveAgent

bucket = Bucket(max_recent=5)

bucket.set_summary(
    "The team is building a SaaS product using Elixir and Phoenix LiveView. "
    "Confirmed decisions: CockroachDB as the primary database with raw SQL only "
    "(no Ecto ORM), magic-link authentication with no passwords or OAuth, "
    "GKE deployment with a minimum of three replicas at all times, and "
    "Redis Cluster for all caching with no in-process caching permitted."
)

bucket.push_message(
    question="How should we structure the Phoenix router?",
    response=(
        "Three pipeline scopes: browser for server-rendered pages, api for "
        "JSON endpoints, and live for LiveView routes. Each scope carries its "
        "own plug stack."
    ),
)
bucket.push_message(
    question="What is the pagination strategy for large listings?",
    response=(
        "Cursor-based pagination using a composite cursor of (created_at, id). "
        "All indexes are defined in raw SQL migration files -- no schema macros."
    ),
)

bucket.set_memories([
    "Authentication decision: magic-link only, single-use Redis tokens, "
    "15-minute TTL, no passwords, no OAuth providers."
])

bucket.set_current_prompt(
    "Can you remind me of the database we chose and whether we are using an ORM?"
)

agent = ActiveAgent(model="gemma3:4b")

print("=" * 60)
print(f"  {agent!r}")
print(f"  {bucket!r}")
print("=" * 60)

print("\n  --- Bucket context string ---\n")
print(bucket.to_context_string())

print("\n" + "=" * 60)
print("  AGENT RESPONSE")
print("=" * 60 + "\n")

response = agent.respond(bucket)
print(response)

print("\n" + "=" * 60)
print("  Done.")
print("=" * 60)
