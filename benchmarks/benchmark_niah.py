# -*- coding: utf-8 -*-
"""
benchmark_niah.py

Needle-in-a-Haystack (NIAH) benchmark: Vanilla agent vs Active Memory.

  Agent A — Vanilla
    The entire CHAT_LOG is serialised into a single system prompt.
    The model must surface the passkey from the full context window.
    For large files (1000 pairs) this will likely exceed the context
    limit and fail — that is the expected result.

  Agent B — Active Memory
    The CHAT_LOG is batch-ingested via pipeline.ingest(), building the
    rolling summary and vector store as the conversation progresses.
    The recall question is then answered by the Active Agent from the
    compressed Bucket.

Scoring: binary — did the response contain the passkey? (exact match)

Usage
-----
    # Default: benchmarks/contexts/niah_chat_100.py
    python benchmarks/benchmark_niah.py

    # Explicit file
    python benchmarks/benchmark_niah.py benchmarks/contexts/niah_chat_1000.py

    # Via env var
    NIAH_CHAT_FILE=benchmarks/contexts/niah_chat_1000.py python benchmarks/benchmark_niah.py

    # Tune stack size for faster ingestion (in .env or shell)
    ACTIVE_MEMORY_MAX_RECENT=100 ACTIVE_MEMORY_BATCH_REDUCTION=50 python benchmarks/benchmark_niah.py
"""

import os
os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

import sys
import re
import time
import shutil
import importlib.util

# ── Path setup ────────────────────────────────────────────────────────────────
_BENCH_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_BENCH_DIR, "..")
sys.path.insert(0, _PROJECT_ROOT)

from active_memory import Pipeline
from active_memory.config import backend_from_env

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH     = "./chroma_db_benchmark_niah"
MAX_RECENT      = int(os.getenv("ACTIVE_MEMORY_MAX_RECENT",      "20"))
BATCH_REDUCTION = int(os.getenv("ACTIVE_MEMORY_BATCH_REDUCTION", "10"))
RECALL_QUESTION = (
    "What is the access code for the staging vault? "
    "Please recall the exact code."
)

_DEFAULT_NIAH = os.path.join(_BENCH_DIR, "contexts", "niah_chat_100.py")
NIAH_FILE = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.getenv("NIAH_CHAT_FILE", _DEFAULT_NIAH)
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_divider(char: str = "-", width: int = 78) -> None:
    print(char * width)


def _truncate(text: str, length: int = 60) -> str:
    text = text.replace("\n", " ").strip()
    return text[:length] + "..." if len(text) > length else text


def _load_niah_module(path: str):
    spec   = importlib.util.spec_from_file_location("_niah_chat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_passkey(module, path: str) -> str | None:
    """Try module.PASSKEY first; fall back to parsing the header comment."""
    if hasattr(module, "PASSKEY"):
        return module.PASSKEY
    try:
        with open(path, encoding="utf-8") as f:
            header = f.read(512)
        m = re.search(r"#\s*Passkey:\s*'([^']+)'", header)
        if m:
            return m.group(1)
    except OSError:
        pass
    return None


def _chat_log_as_system_prompt(chat_log: list[dict]) -> str:
    lines = ["You are reviewing a prior conversation. Use it to answer questions accurately.\n"]
    for msg in chat_log:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].strip().replace("\n", " ")
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def _found(passkey: str | None, response: str) -> str:
    if passkey is None:
        return "N/A"
    return "YES" if passkey in response else "NO"


# ── Load NIAH file ────────────────────────────────────────────────────────────
_print_divider("=")
print("  BENCHMARK NIAH  —  Needle-in-a-Haystack: Vanilla vs Active Memory")
_print_divider("=")
print()
print(f"  File     : {NIAH_FILE}")

if not os.path.exists(NIAH_FILE):
    print(f"\nERROR: NIAH file not found: {NIAH_FILE}")
    print("Generate one with:")
    print("  python benchmarks/generate_chat.py --pairs 100 --passkey 'ALPHA-7731-DELTA' --position 0.3")
    sys.exit(1)

_module    = _load_niah_module(NIAH_FILE)
CHAT_LOG   = _module.CHAT_LOG
PASSKEY    = _extract_passkey(_module, NIAH_FILE)
PAIR_COUNT = len(CHAT_LOG) // 2

print(f"  Pairs    : {PAIR_COUNT}")
print(f"  Passkey  : {PASSKEY!r}")
print(f"  Question : {RECALL_QUESTION}")
print(f"  MaxRecent: {MAX_RECENT}  |  BatchReduction: {BATCH_REDUCTION}")
print()

if PASSKEY is None:
    print("WARNING: No passkey found in this NIAH file. Binary scoring will show N/A.")
    print()

# ── Phase 1 — Vanilla ─────────────────────────────────────────────────────────
_print_divider("=")
print("  PHASE 1 — Agent A (Vanilla): full context in system prompt")
_print_divider("=")
print()

backend       = backend_from_env()
system_prompt = _chat_log_as_system_prompt(CHAT_LOG)
word_estimate = len(system_prompt.split())

print(f"  Backend  : {backend!r}")
print(f"  Prompt   : ~{word_estimate:,} words  (~{word_estimate * 4 // 3:,} chars)")
print(f"  Asking   : {RECALL_QUESTION}")
print()

t0_a       = time.time()
response_a = backend.chat([
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": RECALL_QUESTION},
])
latency_a  = time.time() - t0_a

print(f"  Response ({latency_a:.1f}s):")
for line in response_a.strip().splitlines():
    print(f"    {line}")
print()

# ── Phase 2 — Active Memory ingestion ─────────────────────────────────────────
_print_divider("=")
print("  PHASE 2 — Agent B (Active Memory): batch ingestion")
_print_divider("=")
print()

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

pipeline = Pipeline(
    backend=backend_from_env(),
    chroma_path=CHROMA_PATH,
    max_recent_messages=MAX_RECENT,
    batch_reduction=BATCH_REDUCTION,
)

turn_times:   list[float] = []
ingest_start = time.time()
pair_width   = len(str(PAIR_COUNT))

for i in range(0, PAIR_COUNT * 2, 2):
    user_msg = CHAT_LOG[i]["content"]
    asst_msg = CHAT_LOG[i + 1]["content"]
    pair_num = i // 2 + 1

    t0 = time.time()
    pipeline.ingest(user_msg, asst_msg)
    t_turn = time.time() - t0
    turn_times.append(t_turn)

    bucket  = pipeline.bucket
    depth   = len(bucket.recent_messages)
    stored  = pipeline.retrieval._collection.count()
    preview = _truncate(user_msg, 45)

    print(
        f"  [{pair_num:>{pair_width}}/{PAIR_COUNT}]  "
        f"depth={depth}/{bucket._max_recent}  "
        f"stored={stored:>4}  "
        f"{t_turn:>5.1f}s  "
        f'U: "{preview}"'
    )

ingest_elapsed = time.time() - ingest_start
avg_turn       = sum(turn_times) / len(turn_times) if turn_times else 0.0
max_turn       = max(turn_times) if turn_times else 0.0
max_pair       = (turn_times.index(max_turn) + 1) if turn_times else 0
slow_turns     = [(p + 1, t) for p, t in enumerate(turn_times) if t > avg_turn * 1.5]

print()
_print_divider("-")
print("  Ingestion summary")
_print_divider("-")
print(f"  Total time     : {ingest_elapsed:.1f}s  ({PAIR_COUNT} pairs)")
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

# ── Phase 3 — Active Memory recall ────────────────────────────────────────────
_print_divider("=")
print("  PHASE 3 — Agent B (Active Memory): recall")
_print_divider("=")
print()
print(f"  Asking: {RECALL_QUESTION}")
print()

t0_b       = time.time()
response_b = pipeline.chat(RECALL_QUESTION, skip_observer=True)
latency_b  = time.time() - t0_b

print(f"  Response ({latency_b:.1f}s):")
for line in response_b.strip().splitlines():
    print(f"    {line}")
print()

# ── Phase 4 — Results ─────────────────────────────────────────────────────────
_print_divider("=")
print("  PHASE 4 — Results")
_print_divider("=")
print()

found_a = _found(PASSKEY, response_a)
found_b = _found(PASSKEY, response_b)

if PASSKEY is not None:
    if found_a == "YES" and found_b == "YES":
        winner = "Both agents"
    elif found_b == "YES":
        winner = "Agent B (Active Memory)"
    elif found_a == "YES":
        winner = "Agent A (Vanilla)"
    else:
        winner = "Neither agent"
else:
    winner = None

print(f"  Passkey            : {PASSKEY!r}")
print()
print(f"  Agent A (Vanilla)")
print(f"    Passkey found    : {found_a}")
print(f"    Latency          : {latency_a:.1f}s")
print(f"    Response excerpt : {_truncate(response_a, 60)}")
print()
print(f"  Agent B (Active Memory)")
print(f"    Passkey found    : {found_b}")
print(f"    Latency recall   : {latency_b:.1f}s  (excludes {ingest_elapsed:.1f}s ingestion)")
print(f"    Response excerpt : {_truncate(response_b, 60)}")
print()

_print_divider("-")
if winner:
    print(f"  Winner             : {winner}")
if PASSKEY is not None:
    speedup = latency_a / latency_b if latency_b > 0 else float("inf")
    print(
        f"  Recall latency     : A={latency_a:.1f}s  B={latency_b:.1f}s  "
        f"(B is {speedup:.1f}x {'faster' if speedup >= 1 else 'slower'} at recall)"
    )
    print(f"  Ingestion overhead : {ingest_elapsed:.1f}s  (amortised: {ingest_elapsed / PAIR_COUNT:.2f}s/pair)")
_print_divider("-")
print()

_print_divider("=")
print("  Benchmark NIAH complete.")
_print_divider("=")
