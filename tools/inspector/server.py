"""
tools/inspector/server.py

FastAPI server + browser UI for live LoCoMo benchmark inspection.

Runs the LoCoMo QA benchmark in a background thread and exposes endpoints to
start, pause, resume, and reset the run, plus a /api/state endpoint that
returns the full pipeline state — recent messages, topic tree, retrieved
memories, Chroma counts — for display in the UI.

Boot
----
    pip install -e .[inspector]
    uvicorn tools.inspector.server:app --reload --port 8000
    # then open http://localhost:8000

The benchmark loop runs against gemma3:4b on a local Ollama by default; edit
the constants at the top of this file to retarget.

Design notes
------------
- Single global benchmark thread.  Only one run at a time.
- threading.Event for pause/resume — the loop calls _check_pause() at safe
  points (between pairs and between questions) and blocks while the event
  is cleared.
- Snapshots are deep-copied because pipeline.bucket.topic_tree is mutated
  live by the Observer thread.
- pipeline.retrieval._retrieve_scored() is used for inspection retrievals
  during Q&A.  It does NOT increment retrieval_count, so polling does not
  pollute Chroma metadata.
- Q&A uses pipeline.build_context(q) + pipeline._backend.chat(ctx) directly.
  We never call pipeline.update() during Q&A so questions never enter the
  Bucket — matches the published LoCoMo evaluation protocol.
"""

from __future__ import annotations

import copy
import os
import pathlib
import shutil
import sys
import threading
import time
from typing import Any

# ── Path setup — let the server import active_memory and loader.py ───────────

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "benchmarks" / "locomo"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from active_memory import Pipeline
from active_memory.backends import OllamaBackend
from loader import load_locomo, CATEGORY_LABELS  # type: ignore[import-not-found]


# ── Configuration ────────────────────────────────────────────────────────────

DATA_PATH    = str(_REPO_ROOT / "benchmarks" / "locomo" / "data" / "locomo10.json")
STATIC_PATH  = str(_REPO_ROOT / "tools" / "inspector" / "static")
DEFAULT_MODEL = "gemma3:4b"
AGENT_URL     = "http://localhost:11434"
OBSERVER_URL  = "http://localhost:11434"
CURATOR_URL   = "http://localhost:11434"
CHROMA_PATH_TEMPLATE = "./chroma_db_inspector_{idx}"

LOG_TAIL_MAX = 200


# ── Global state ─────────────────────────────────────────────────────────────

def _idle_state() -> dict[str, Any]:
    return {
        "status":    "idle",   # idle | running | paused | done | error
        "conv_idx":  None,
        "conv_id":   None,
        "phase":     None,     # "ingest" | "qa" | None
        "progress": {
            "pairs_done":      0,
            "pairs_total":     0,
            "questions_done":  0,
            "questions_total": 0,
        },
        "snapshot":  None,
        "questions": [],
        "log":       [],
        "error":     None,
    }


_state: dict[str, Any] = _idle_state()
_state_lock   = threading.Lock()
_pause_event  = threading.Event()
_pause_event.set()
_stop_flag    = threading.Event()
_thread:   threading.Thread | None = None
_pipeline: Pipeline | None = None


# ── Helpers — copied (not imported) from benchmark_locomo.py ─────────────────
# benchmark_locomo.py runs argparse at import time; copying the two helpers
# keeps the inspector independent of that script's CLI shape.

def _make_pairs(turns: list[dict]) -> list[tuple[str, str]]:
    """Pair sequential turns into (user, assistant). Drops a trailing odd turn."""
    pairs: list[tuple[str, str]] = []
    for i in range(0, len(turns) - 1, 2):
        u = turns[i]["text"].strip()
        a = turns[i + 1]["text"].strip()
        if u and a:
            pairs.append((u, a))
    return pairs


def _judge(
    question: str,
    ground_truth: str,
    response: str,
    model: str = DEFAULT_MODEL,
    base_url: str = AGENT_URL,
) -> int:
    """Score `response` against `ground_truth` (1-5).  Returns 0 on failure."""
    import ollama

    prompt = (
        f"Question: {question}\n\n"
        f"Ground truth answer: {ground_truth}\n\n"
        f"Agent response: {response}\n\n"
        "Score this response 1-5 for factual accuracy and consistency with "
        "the ground truth answer.\n"
        "5 = fully correct and consistent.\n"
        "1 = wrong, contradictory, or completely missing the answer.\n"
        "Reply with only a single integer and nothing else."
    )
    try:
        client = ollama.Client(host=base_url)
        result = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = result["message"]["content"].strip()
        for ch in raw:
            if ch.isdigit():
                score = int(ch)
                if 1 <= score <= 5:
                    return score
        return 0
    except Exception:
        return 0


# ── State helpers ────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    """Append a short event line to the rolling log (must be called with lock)."""
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    log = _state["log"]
    log.append(line)
    if len(log) > LOG_TAIL_MAX:
        del log[: len(log) - LOG_TAIL_MAX]


def _snapshot(pipeline: Pipeline, last_query: str | None = None) -> dict[str, Any]:
    """Deep-copy a snapshot of the live pipeline state for the UI."""
    bucket = pipeline.bucket
    snap: dict[str, Any] = {
        "recent_messages": copy.deepcopy(bucket.recent_messages),
        "depth":           len(bucket.recent_messages),
        "max_recent":      bucket._max_recent,
        "topic_tree":      copy.deepcopy(bucket.topic_tree),
        "topic_summary":   bucket.get_summary_text(),
        "chroma_count":    pipeline.retrieval._collection.count(),
    }
    if last_query:
        # _retrieve_scored does NOT increment retrieval_count — safe to poll.
        try:
            snap["retrieved"] = pipeline.retrieval._retrieve_scored(last_query)
        except Exception as exc:
            snap["retrieved"] = []
            snap["retrieval_error"] = f"{type(exc).__name__}: {exc}"
        snap["last_query"] = last_query
    else:
        snap["retrieved"] = []
        snap["last_query"] = None
    return snap


def _check_pause() -> bool:
    """
    Block while pause is active. Returns True if the run should continue,
    False if a stop has been requested (caller should bail out).
    """
    if _stop_flag.is_set():
        return False
    if not _pause_event.is_set():
        with _state_lock:
            _state["status"] = "paused"
            _log("paused")
        # Wait until either resumed or stopped.
        while not _pause_event.is_set():
            if _stop_flag.is_set():
                return False
            _pause_event.wait(timeout=0.5)
        with _state_lock:
            if _state["status"] == "paused":
                _state["status"] = "running"
                _log("resumed")
    return not _stop_flag.is_set()


# ── Benchmark thread ─────────────────────────────────────────────────────────

def _run_benchmark(conv_idx: int) -> None:
    global _pipeline
    chroma_path = CHROMA_PATH_TEMPLATE.format(idx=conv_idx)

    try:
        conversations = load_locomo(DATA_PATH)
        if conv_idx < 0 or conv_idx >= len(conversations):
            raise IndexError(
                f"conv_idx {conv_idx} out of range — dataset has "
                f"{len(conversations)} conversations."
            )
        conv  = conversations[conv_idx]
        pairs = _make_pairs(conv.turns)

        with _state_lock:
            _state["status"]    = "running"
            _state["conv_idx"]  = conv_idx
            _state["conv_id"]   = conv.id
            _state["phase"]     = "ingest"
            _state["progress"]["pairs_total"]     = len(pairs)
            _state["progress"]["questions_total"] = len(conv.questions)
            _log(f"loaded conversation {conv.id} ({len(pairs)} pairs, "
                 f"{len(conv.questions)} questions)")

        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)

        _pipeline = Pipeline(
            backend=OllamaBackend(model=DEFAULT_MODEL, base_url=AGENT_URL),
            chroma_path=chroma_path,
            observer_backend=OllamaBackend(model=DEFAULT_MODEL, base_url=OBSERVER_URL),
            curator_backend=OllamaBackend(model=DEFAULT_MODEL, base_url=CURATOR_URL),
        )

        # ── Phase 1: ingest ───────────────────────────────────────────────────
        for i, (u, a) in enumerate(pairs, 1):
            if not _check_pause():
                return
            _pipeline.ingest(u, a)
            with _state_lock:
                _state["progress"]["pairs_done"] = i
                _state["snapshot"] = _snapshot(_pipeline)
                _log(f"ingested pair {i}/{len(pairs)}")

        # ── Phase 2: Q&A ──────────────────────────────────────────────────────
        with _state_lock:
            _state["phase"] = "qa"

        for i, q in enumerate(conv.questions, 1):
            if not _check_pause():
                return

            ctx       = _pipeline.build_context(q.question)
            retrieved = copy.deepcopy(_pipeline._bucket.memories)
            response  = _pipeline._backend.chat(ctx)
            score     = _judge(q.question, q.answer, response,
                               model=DEFAULT_MODEL, base_url=AGENT_URL)

            with _state_lock:
                _state["questions"].append({
                    "idx":          i,
                    "category":     q.category_label,
                    "category_id":  q.category,
                    "question":     q.question,
                    "ground_truth": q.answer,
                    "response":     response,
                    "score":        score,
                    "retrieved":    retrieved,
                })
                _state["progress"]["questions_done"] = i
                _state["snapshot"] = _snapshot(_pipeline, last_query=q.question)
                _log(f"q{i}/{len(conv.questions)} score={score}")

        with _state_lock:
            _state["status"] = "done"
            _state["phase"]  = None
            _log("benchmark complete")

    except Exception as exc:
        with _state_lock:
            _state["status"] = "error"
            _state["error"]  = f"{type(exc).__name__}: {exc}"
            _log(f"ERROR: {exc}")


def _spawn(conv_idx: int) -> None:
    """Start a fresh benchmark thread.  Caller must ensure no thread is alive."""
    global _thread
    _stop_flag.clear()
    _pause_event.set()
    _thread = threading.Thread(
        target=_run_benchmark,
        args=(conv_idx,),
        name="locomo-bench",
        daemon=True,
    )
    _thread.start()


def _kill_running_thread() -> None:
    """Signal stop, unblock pause, and join the benchmark thread (if any)."""
    global _thread
    if _thread is not None and _thread.is_alive():
        _stop_flag.set()
        _pause_event.set()  # unblock any pause wait so the loop can exit
        _thread.join(timeout=10)
    _thread = None
    _stop_flag.clear()


def _wipe_inspector_chroma() -> None:
    """Remove every chroma_db_inspector_*/ directory under the CWD."""
    cwd = pathlib.Path.cwd()
    for path in cwd.glob("chroma_db_inspector_*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Active Memory — LoCoMo Inspector")
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")


class StartRequest(BaseModel):
    conv_idx: int


@app.get("/")
def root() -> FileResponse:
    return FileResponse(os.path.join(STATIC_PATH, "index.html"))


@app.get("/api/conversations")
def list_conversations() -> list[dict[str, Any]]:
    """Return summary metadata for every conversation in the dataset."""
    try:
        convs = load_locomo(DATA_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"locomo10.json not found at {DATA_PATH}. "
                "Clone https://github.com/snap-research/locomo and place "
                "data/locomo10.json under benchmarks/locomo/data/."
            ),
        ) from exc
    out: list[dict[str, Any]] = []
    for c in convs:
        pairs = _make_pairs(c.turns)
        out.append({
            "idx":         c.idx,
            "id":          c.id,
            "turns":       len(c.turns),
            "pairs":       len(pairs),
            "questions":   len(c.questions),
            "speaker_a":   c.speaker_a,
            "speaker_b":   c.speaker_b,
        })
    return out


@app.get("/api/categories")
def list_categories() -> dict[int, str]:
    """Return the LoCoMo category id → label mapping."""
    return CATEGORY_LABELS


@app.post("/api/start")
def start(req: StartRequest) -> dict[str, Any]:
    global _thread
    if _thread is not None and _thread.is_alive():
        raise HTTPException(
            status_code=409,
            detail=f"Benchmark already running (status={_state['status']}). "
                   "Reset before starting a new run.",
        )
    # Fresh state.
    with _state_lock:
        _state.clear()
        _state.update(_idle_state())
        _state["status"] = "running"
        _log(f"start requested for conversation {req.conv_idx}")
    _spawn(req.conv_idx)
    return {"ok": True}


@app.post("/api/pause")
def pause() -> dict[str, Any]:
    if _state["status"] not in ("running",):
        return {"ok": True, "noop": True, "status": _state["status"]}
    _pause_event.clear()
    return {"ok": True}


@app.post("/api/resume")
def resume() -> dict[str, Any]:
    if _state["status"] not in ("paused",):
        return {"ok": True, "noop": True, "status": _state["status"]}
    _pause_event.set()
    return {"ok": True}


@app.post("/api/reset")
def reset() -> dict[str, Any]:
    global _pipeline
    _kill_running_thread()
    _pipeline = None
    _wipe_inspector_chroma()
    with _state_lock:
        _state.clear()
        _state.update(_idle_state())
    _pause_event.set()
    return {"ok": True}


@app.get("/api/state")
def get_state() -> dict[str, Any]:
    """Return a shallow copy of the global state dict (snapshot)."""
    # Snapshot under the lock so the UI never sees a half-mutated dict.
    with _state_lock:
        # The snapshot field is already a deep copy taken at write time;
        # everything else is JSON-friendly so a shallow copy is enough.
        return {
            "status":    _state["status"],
            "conv_idx":  _state["conv_idx"],
            "conv_id":   _state["conv_id"],
            "phase":     _state["phase"],
            "progress":  dict(_state["progress"]),
            "snapshot":  _state["snapshot"],
            "questions": list(_state["questions"]),
            "log":       list(_state["log"]),
            "error":     _state["error"],
        }
