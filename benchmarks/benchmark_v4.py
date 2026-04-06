# -*- coding: utf-8 -*-
import os
os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

"""
benchmark_v4.py

Controlled comparison between a vanilla agent and the Active Memory pipeline.

  Agent A - Vanilla
    The selected conversation slice is packed into a single system prompt
    and sent to Ollama alongside each recall question.  No compression,
    no retrieval.  The model sees everything but has to hold it all in its
    context window at once.

  Agent B - Active Memory
    All conversation pairs are batch-ingested via pipeline.ingest().
    This builds the summary (Observer) and memory store (Curator -> Retrieval)
    as if the conversation had actually happened.  Each recall question is
    then answered by the Active Agent, which reads from the compressed Bucket.

Context design (see benchmarks/contexts/architecture_v4.py)
  Pairs  0-7    : six technology decisions established naturally
  Pairs  8-54   : development discussion, three noise passages (MySQL/EKS/Redis)
  Pairs 55-107  : extended discussion, three more noise passages + follow-ups
  Recall questions test recall of the original six decisions only.

Sizing: pass n_pairs to ArchitectureContext to control Agent A's context size.
  n_pairs=55   ~7,500 tokens  (original baseline, Agent A typically scores well)
  n_pairs=108  ~18,000 tokens (full stress test, decisions buried in first 7%)

Scoring: a judge Ollama call rates each response 1-5 against ground truth.
Output:  results table, average scores and latencies, drift difference, and
         a section showing what the Active Memory pipeline stored.
"""

import sys
import time
import shutil
import ollama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from active_memory import Pipeline
from contexts.architecture_v4 import ArchitectureContext


# ──────────────────────────────────────────────────────────────────────────────
# 1.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _judge(question: str, ground_truth: str, response: str) -> int:
    """Ask the model to score a response 1-5 against ground truth."""
    prompt = (
        f"Ground truth answer: {ground_truth}\n\n"
        f"Agent response: {response}\n\n"
        "Score this response from 1 to 5 for accuracy and consistency "
        "with the ground truth. "
        "5 means fully consistent and accurate. "
        "1 means wrong or contradictory. "
        "Reply with only a single integer."
    )
    try:
        result = ollama.chat(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = result["message"]["content"].strip()
        for ch in raw:
            if ch.isdigit():
                score = int(ch)
                if 1 <= score <= 5:
                    return score
        return 0
    except Exception as exc:
        print(f"  [judge error] {exc}")
        return 0


def _print_divider(char: str = "-", width: int = 78) -> None:
    print(char * width)


def _truncate(text: str, length: int = 48) -> str:
    text = text.replace("\n", " ").strip()
    return text[:length] + "..." if len(text) > length else text


# ──────────────────────────────────────────────────────────────────────────────
# 2.  CONFIG
# ──────────────────────────────────────────────────────────────────────────────

AGENT_MODEL  = "gemma3:4b"
JUDGE_MODEL  = "gemma3:4b"
CHROMA_PATH  = "./chroma_db_benchmark_v4"

# Ollama instance URLs.
# Active Agent uses global ollama.chat() and always hits AGENT_URL implicitly.
# Routing Observer and Curator to a separate port lets them run in true
# parallel during ingestion (Observer is synchronous; Curator is async) so
# Curator calls never queue behind Observer calls.
AGENT_URL    = "http://localhost:11434"   # Active Agent (implicit, not passed to Pipeline)
OBSERVER_URL = "http://localhost:11434"   # Observer  — synchronous during ingest
CURATOR_URL  = "http://localhost:11434"   # Curator   — async daemon, parallel to Observer

# Context size:
#   None  -> full conversation (~108 pairs, ~18K tokens for Agent A)
#   55    -> original baseline (~7.5K tokens, Agent A typically scores well)
N_PAIRS: int | None = None

ctx = ArchitectureContext(n_pairs=N_PAIRS)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  BANNER
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 78)
print("  BENCHMARK V4  —  Vanilla vs Active Memory Pipeline")
print("=" * 78)
print(f"\n  Model      : {AGENT_MODEL}")
print(f"  Judge      : {JUDGE_MODEL}")
print(f"  Pairs      : {ctx.pair_count}  (~{ctx.token_estimate:,} tokens for Agent A)")
print(f"  Questions  : {len(ctx.recall_questions)}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 4.  AGENT A — VANILLA
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_A = ctx.as_system_prompt()

_print_divider("=")
print("  PHASE 1 — Agent A (Vanilla): answering recall questions")
_print_divider("=")
print()

agent_a_responses: dict[int, str]   = {}
agent_a_latencies: dict[int, float] = {}

for q_num, question in ctx.recall_questions.items():
    print(f"  Q{q_num}: {_truncate(question, 60)}")
    t0 = time.time()
    result = ollama.chat(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_A},
            {"role": "user",   "content": question},
        ],
    )
    latency  = time.time() - t0
    response = result["message"]["content"].strip()
    agent_a_responses[q_num] = response
    agent_a_latencies[q_num] = latency
    print(f"       -> {latency:.1f}s  ({len(response.split())} words)")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 5.  AGENT B — ACTIVE MEMORY: INGEST THEN RECALL
# ──────────────────────────────────────────────────────────────────────────────

_print_divider("=")
print("  PHASE 2 — Agent B (Active Memory): batch ingestion")
_print_divider("=")
print()

# Clear any Chroma data left over from a previous run so counts are accurate.
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

pipeline = Pipeline(
    model=AGENT_MODEL,
    chroma_path=CHROMA_PATH,
    observer_url=OBSERVER_URL,
    curator_url=CURATOR_URL,
)

_PAIR_COUNT = ctx.pair_count
turn_times: list[float] = []

ingest_start = time.time()
for i in range(0, len(ctx.messages), 2):
    user_msg  = ctx.messages[i]["content"]
    asst_msg  = ctx.messages[i + 1]["content"]
    pair_num  = i // 2 + 1
    preview   = _truncate(user_msg, 45)

    t0 = time.time()
    pipeline.ingest(user_msg, asst_msg)
    t_turn = time.time() - t0
    turn_times.append(t_turn)

    bucket       = pipeline.bucket
    depth        = len(bucket.recent_messages)
    stored_count = pipeline.retrieval._collection.count()

    print(
        f"  [{pair_num:>3}/{_PAIR_COUNT}]  "
        f"depth={depth}/{bucket._max_recent}  "
        f"stored={stored_count:>3}  "
        f"{t_turn:>5.1f}s  "
        f"U: \"{preview}\""
    )

ingest_elapsed = time.time() - ingest_start

# Ingest summary stats
avg_turn   = sum(turn_times) / len(turn_times)
max_turn   = max(turn_times)
max_pair   = turn_times.index(max_turn) + 1
slow_turns = [(p + 1, t) for p, t in enumerate(turn_times) if t > avg_turn * 1.5]

print()
_print_divider("-")
print(f"  Ingestion summary")
_print_divider("-")
print(f"  Total time     : {ingest_elapsed:.1f}s  ({_PAIR_COUNT} pairs)")
print(f"  Avg per pair   : {avg_turn:.2f}s")
print(f"  Slowest pair   : {max_turn:.1f}s  (pair {max_pair})")
if slow_turns:
    slow_str = "  ".join(f"#{idx}={t:.1f}s" for idx, t in slow_turns[:8])
    print(f"  Slow pairs     : {slow_str}")
print(f"  Items in Chroma: {pipeline.retrieval._collection.count()}")
print(f"  Bucket summary : {'set' if pipeline.bucket.summary else 'empty'}")
_print_divider("-")
print(f"  {pipeline!r}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 6.  AGENT B — ACTIVE MEMORY: RECALL
# ──────────────────────────────────────────────────────────────────────────────

_print_divider("=")
print("  PHASE 3 — Agent B (Active Memory): answering recall questions")
_print_divider("=")
print()

agent_b_responses: dict[int, str]   = {}
agent_b_latencies: dict[int, float] = {}

for q_num, question in ctx.recall_questions.items():
    print(f"  Q{q_num}: {_truncate(question, 60)}")
    t0 = time.time()
    response = pipeline.chat(question, skip_observer=True)
    latency  = time.time() - t0
    agent_b_responses[q_num] = response
    agent_b_latencies[q_num] = latency
    print(f"       -> {latency:.1f}s  ({len(response.split())} words)")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 7.  SCORING
# ──────────────────────────────────────────────────────────────────────────────

_print_divider("=")
print("  PHASE 4 — Judge scoring")
_print_divider("=")
print()

scores_a: dict[int, int] = {}
scores_b: dict[int, int] = {}

for q_num in ctx.recall_questions:
    gt = ctx.ground_truth[q_num]

    print(f"  Scoring Q{q_num} Agent A ...")
    scores_a[q_num] = _judge(ctx.recall_questions[q_num], gt, agent_a_responses[q_num])

    print(f"  Scoring Q{q_num} Agent B ...")
    scores_b[q_num] = _judge(ctx.recall_questions[q_num], gt, agent_b_responses[q_num])

    print(f"    A={scores_a[q_num]}  B={scores_b[q_num]}")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 8.  RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────

_print_divider("=")
print("  RESULTS TABLE")
_print_divider("=")
print()

COL_Q     = 50
COL_SCORE = 7
COL_LAT   = 8

header = (
    f"  {'Question':<{COL_Q}}  "
    f"{'A Scr':>{COL_SCORE}}  "
    f"{'B Scr':>{COL_SCORE}}  "
    f"{'A Lat':>{COL_LAT}}  "
    f"{'B Lat':>{COL_LAT}}"
)
print(header)
_print_divider("-", len(header))

for q_num in sorted(ctx.recall_questions):
    q_text = f"Q{q_num}: " + _truncate(ctx.recall_questions[q_num], COL_Q - 4)
    sa     = scores_a[q_num]
    sb     = scores_b[q_num]
    la     = agent_a_latencies[q_num]
    lb     = agent_b_latencies[q_num]
    row = (
        f"  {q_text:<{COL_Q}}  "
        f"{sa:>{COL_SCORE}}  "
        f"{sb:>{COL_SCORE}}  "
        f"{la:>{COL_LAT - 1}.1f}s  "
        f"{lb:>{COL_LAT - 1}.1f}s"
    )
    print(row)

_print_divider("-", len(header))

n          = len(ctx.recall_questions)
avg_sa     = sum(scores_a.values()) / n
avg_sb     = sum(scores_b.values()) / n
avg_la     = sum(agent_a_latencies.values()) / n
avg_lb     = sum(agent_b_latencies.values()) / n
drift_diff = avg_sb - avg_sa
leader     = "Agent B (Active Memory)" if drift_diff > 0 else "Agent A (Vanilla)"
drift_abs  = abs(drift_diff)

avg_row = (
    f"  {'AVERAGE':<{COL_Q}}  "
    f"{avg_sa:>{COL_SCORE}.2f}  "
    f"{avg_sb:>{COL_SCORE}.2f}  "
    f"{avg_la:>{COL_LAT - 1}.1f}s  "
    f"{avg_lb:>{COL_LAT - 1}.1f}s"
)
print(avg_row)
_print_divider("=", len(header))

print()
print(f"  Drift leader : {leader}")
print(f"  Drift margin : {drift_abs:.2f} points (out of 5)")
print(f"  Latency A    : {avg_la:.1f}s avg per question")
print(f"  Latency B    : {avg_lb:.1f}s avg per question "
      f"(excludes {ingest_elapsed:.1f}s ingest)")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 9.  WHAT ACTIVE MEMORY STORED
# ──────────────────────────────────────────────────────────────────────────────

_print_divider("=")
print("  ACTIVE MEMORY STORE — what the Curator persisted")
_print_divider("=")

retrieval    = pipeline.retrieval
total_stored = retrieval._collection.count()
print(f"\n  Total items in Chroma: {total_stored}")

BROAD_QUERY = "technology decisions database authentication deployment caching"
print(f"  Query: \"{BROAD_QUERY}\"\n")

scored = retrieval._retrieve_scored(BROAD_QUERY)

if scored:
    for i, item in enumerate(scored, 1):
        print(f"  [{i}]  sim={item['similarity']:.4f}  ts={item['timestamp']:.0f}")
        words = item["content"].split()
        line  = "       "
        for word in words:
            if len(line) + len(word) + 1 > 76:
                print(line)
                line = "       " + word
            else:
                line += " " + word
        if line.strip():
            print(line)
        print()
else:
    print("  (no results above threshold -- Curator threads may still be running)")

# ──────────────────────────────────────────────────────────────────────────────
# 10.  FINAL BUCKET SUMMARY
# ──────────────────────────────────────────────────────────────────────────────

_print_divider("=")
print("  FINAL BUCKET SUMMARY (Observer output)")
_print_divider("=")
print()
summary = pipeline.bucket.summary.strip() or "(no summary yet)"
for line in summary.splitlines():
    print(f"  {line}")
print()
_print_divider("=")
print("  Benchmark complete.")
_print_divider("=")
