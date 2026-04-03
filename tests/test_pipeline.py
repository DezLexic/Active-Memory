"""
test_pipeline.py

Runs a 10-message conversation through the Pipeline end-to-end.

Conversation structure
-----------------------
  Turns 1-2  : Establish decision 1 -- language and framework (Elixir / Phoenix)
  Turns 3-4  : Establish decision 2 -- database and query policy (CockroachDB, raw SQL)
  Turns 5-6  : Continue building detail (auth, deployment)
  Turn  7    : Filler -- pagination strategy (pushes turn 1 off the stack)
  Turn  8    : Filler -- router structure   (pushes turn 2 off the stack)
  Turn  9    : Recall challenge -- ask about decision 1
  Turn  10   : Recall challenge -- ask about decision 2

Turns 9-10 verify that the pipeline surfaces the early decisions even after
the raw pairs have been evicted from the recent-message stack -- either via
the rolling summary (Observer) or retrieved memories (Curator -> Retrieval).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

import threading
from active_memory import Pipeline

# ── Pipeline setup ────────────────────────────────────────────────────────────

pipeline = Pipeline(
    model="gemma3:4b",
    chroma_path="./chroma_db_test_pipeline",
    max_recent_messages=5,
)

print("=" * 60)
print(f"  {pipeline!r}")
print("=" * 60 + "\n")

# ── Conversation ──────────────────────────────────────────────────────────────

TURNS = [
    "We are starting a new backend project. What languages are worth considering?",
    "Let's go with Elixir and Phoenix LiveView. That is the decision -- no revisiting.",
    "What database should we use?",
    "CockroachDB, raw SQL only. No ORM, no query builders. That is locked in.",
    "What authentication approach fits this stack?",
    "What does the deployment target look like?",
    "How should we handle cursor-based pagination for large tables?",
    "Walk me through the Phoenix router pipeline structure.",
    "I forgot -- what language and framework did we decide on at the start?",
    "And what was the database decision, including the query policy we agreed to?",
]

# Capture spawned daemon threads so we can join the last pair before printing
# final state.
_spawned: list[threading.Thread] = []
_original_start = threading.Thread.start

def _patched_start(self):
    _spawned.append(self)
    _original_start(self)

threading.Thread.start = _patched_start

for i, message in enumerate(TURNS, 1):
    print(f"  Turn {i:02d}")
    print(f"  USER:  {message}")
    response = pipeline.chat(message)
    print(f"  AGENT: {response}")

    b = pipeline.bucket
    print(
        f"  [bucket: recent={len(b.recent_messages)}/{b._max_recent}  "
        f"memories={len(b.memories)}  "
        f"summary={'set' if b.summary else 'empty'}]"
    )
    print()

threading.Thread.start = _original_start

# ── Wait for the last round of background threads ─────────────────────────────

if _spawned:
    print("  Waiting for background threads to finish...")
    for t in _spawned[-2:]:
        t.join(timeout=60)
    print("  Background threads done.\n")

# ── Final state ───────────────────────────────────────────────────────────────

bucket    = pipeline.bucket
retrieval = pipeline.retrieval

print("=" * 60)
print("  FINAL BUCKET SUMMARY")
print("=" * 60)
summary = bucket.summary.strip() or "(no summary yet)"
for line in summary.splitlines():
    print(f"  {line}")

print()
print("=" * 60)
print(f"  RETRIEVAL QUERY -- total stored: {retrieval._collection.count()}")
print("=" * 60)

QUERY = "language framework database decisions"
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
    print("  (nothing stored yet -- Curator threads may still be running)\n")

print("=" * 60)
print(f"  Done.  {pipeline!r}")
print("=" * 60)
