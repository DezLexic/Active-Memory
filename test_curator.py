"""
test_curator.py

Creates a Retrieval instance and a Curator, then evaluates five popped pairs.
A conversation summary is passed alongside each pair to verify that context
improves decisions on short or ambiguous exchanges.

Pairs
-----
  1. Database decision     -- clear, long                  -> STORE expected
  2. Auth constraint       -- clear, long                  -> STORE expected
  3. Short confirmation    -- "Agreed" / "Yes" style;
                              ambiguous without summary     -> STORE expected (with context)
  4. Small-talk            -- no decision even with context -> NO STORE expected
  5. Replica floor         -- brief but decisive            -> STORE expected

After evaluation a retrieval query confirms the right pairs are in Chroma.
"""

import os
os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

import shutil
from retrieval import Retrieval
from curator   import Curator

# ── Fresh store for each run ──────────────────────────────────────────────────

STORE_PATH = "./chroma_store_test_curator"
shutil.rmtree(STORE_PATH, ignore_errors=True)

retrieval = Retrieval(
    chroma_path=STORE_PATH,
    similarity_threshold=0.25,
    max_results=5,
)

curator = Curator(
    model="gemma3:4b",
    retrieval=retrieval,
)

# Summary passed as context for all pairs.
SUMMARY = (
    "The team is building a SaaS product using Elixir and Phoenix LiveView. "
    "Agreed decisions so far: CockroachDB as the primary database with raw SQL only, "
    "magic-link authentication with no passwords or OAuth, "
    "GKE deployment with a minimum of three replicas, "
    "and Redis Cluster for all caching with no in-process caching permitted."
)

print("=" * 60)
print(f"  {curator!r}")
print(f"  {retrieval!r}")
print("=" * 60)
print(f"\n  Context summary ({len(SUMMARY.split())} words):")
print(f"  {SUMMARY[:80].rstrip()}...")
print()

# ── Five test pairs ───────────────────────────────────────────────────────────

pairs = [
    {
        "label": "1 - database decision, long (STORE expected)",
        "summary": SUMMARY,
        "pair": {
            "question": "Which database are we committing to for this project?",
            "response": (
                "CockroachDB as the primary database. Distributed by default, "
                "PostgreSQL-wire-compatible. Raw SQL only -- no Ecto ORM and no "
                "query builders anywhere in the codebase."
            ),
        },
    },
    {
        "label": "2 - auth constraint, long (STORE expected)",
        "summary": SUMMARY,
        "pair": {
            "question": "What authentication method are we using?",
            "response": (
                "Magic-link authentication only. No passwords and no OAuth providers. "
                "Single-use tokens stored in Redis with a 15-minute TTL."
            ),
        },
    },
    {
        "label": "3 - short confirmation, ambiguous without summary (STORE expected with context)",
        "summary": SUMMARY,
        "pair": {
            "question": "So we are locked in on no in-process caching across all pods?",
            "response": "Correct, that is a firm constraint.",
        },
    },
    {
        "label": "4 - small-talk, no decision even with context (NO STORE expected)",
        "summary": SUMMARY,
        "pair": {
            "question": "How is everyone feeling about the progress so far?",
            "response": (
                "Pretty good overall. The team seems aligned and there are no "
                "major blockers right now."
            ),
        },
    },
    {
        "label": "5 - brief but decisive (STORE expected)",
        "summary": SUMMARY,
        "pair": {
            "question": "What is the replica floor we agreed on?",
            "response": "Three replicas minimum. That is non-negotiable.",
        },
    },
]

# ── Evaluate each pair ────────────────────────────────────────────────────────

for entry in pairs:
    label   = entry["label"]
    pair    = entry["pair"]
    summary = entry.get("summary")

    words = len(f"{pair['question']} {pair['response']}".split())
    print(f"  Evaluating: {label}")
    print(f"  Words in pair: {words}  |  Summary passed: {'yes' if summary else 'no'}")

    curator._last_store  = None
    curator._last_reason = None

    curator.evaluate(pair, summary=summary)

    store  = curator._last_store
    reason = curator._last_reason
    tag    = "STORED" if store else "NOT STORED"
    print(f"  -> {tag}")
    print(f"  Reason: {reason}")
    print()

# ── Confirm what landed in Chroma ─────────────────────────────────────────────

print("=" * 60)
print(f"  Retrieval check -- total stored: {retrieval._collection.count()}")
print("=" * 60)

QUERY = "database caching and authentication decisions"
print(f"\n  Query: \"{QUERY}\"\n")

scored = retrieval._retrieve_scored(QUERY)

if scored:
    for i, item in enumerate(scored, 1):
        print(f"  [{i}]  sim={item['similarity']:.4f}  ts={item['timestamp']:.0f}")
        words = item["content"].split()
        line  = "       "
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

print("=" * 60)
print(f"  Done.  {retrieval!r}")
print("=" * 60)
