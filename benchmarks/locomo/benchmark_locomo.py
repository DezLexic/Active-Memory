"""
benchmark_locomo.py

LoCoMo QA benchmark for a single conversation.

Evaluates Active Memory against the LoCoMo dataset's QA task:
  1. Ingest the conversation into Active Memory via pipeline.ingest().
  2. Answer each annotated question with pipeline.build_context() + backend.chat().
     (The two-method pattern is used deliberately — questions never pollute memory.)
  3. Score each answer 1-5 with a judge model against ground truth.
  4. Report results broken down by category (single-hop / multi-hop / temporal /
     open-domain) for direct comparison with published LoCoMo numbers.

Usage
-----
    python benchmarks/locomo/benchmark_locomo.py --conversation 0
    python benchmarks/locomo/benchmark_locomo.py --conversation 3 --verbose

Dataset
-------
    Clone https://github.com/snap-research/locomo and place data/locomo10.json at
    benchmarks/locomo/data/locomo10.json before running.
"""

from __future__ import annotations

import os
import sys
import time
import shutil
import argparse

os.environ.setdefault("OLLAMA_KEEP_ALIVE", "10m")

# ── Path setup — allow running from project root without installing ────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.dirname(__file__))  # for loader.py

from loader import load_locomo, LoCoMoConversation, CATEGORY_LABELS
from active_memory import Pipeline
from active_memory.backends import OllamaBackend


# ── Config ────────────────────────────────────────────────────────────────────

CONVERSATION_IDX = 0
DATA_PATH        = os.path.join(os.path.dirname(__file__), "data", "locomo10.json")
CHROMA_PATH      = "./chroma_db_locomo_{idx}"
AGENT_MODEL      = "gemma3:4b"
JUDGE_MODEL      = "gemma3:4b"
AGENT_URL        = "http://localhost:11434"
OBSERVER_URL     = "http://localhost:11434"
CURATOR_URL      = "http://localhost:11434"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_divider(char: str = "-", width: int = 78) -> None:
    print(char * width)


def _truncate(text: str, length: int = 55) -> str:
    text = text.replace("\n", " ").strip()
    return text[:length] + "..." if len(text) > length else text


def _safe_avg(scores: list[int]) -> float | None:
    """Return average or None if the list is empty."""
    return sum(scores) / len(scores) if scores else None


def _make_pairs(turns: list[dict]) -> list[tuple[str, str]]:
    """
    Convert the flat turn list into (user, assistant) pairs.

    Turns are paired sequentially: turn[0]=user, turn[1]=assistant,
    turn[2]=user, turn[3]=assistant, etc.  If the total is odd, the last
    unpaired turn is dropped.
    """
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
    model: str = JUDGE_MODEL,
    base_url: str = AGENT_URL,
) -> int:
    """
    Ask the judge model to score `response` against `ground_truth` (1-5).

    Returns 0 on failure so the caller can distinguish scored from errored.
    """
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
    except Exception as exc:
        print(f"  [judge error] {exc}")
        return 0


# ── Core benchmark function ───────────────────────────────────────────────────

def run_conversation(
    conv_idx: int,
    data_path: str = DATA_PATH,
    agent_model: str = AGENT_MODEL,
    judge_model: str = JUDGE_MODEL,
    agent_url: str = AGENT_URL,
    observer_url: str = OBSERVER_URL,
    curator_url: str = CURATOR_URL,
    verbose: bool = True,
) -> dict:
    """
    Run the LoCoMo benchmark for one conversation.

    Parameters
    ----------
    conv_idx        Which conversation to run (0-9).
    data_path       Path to locomo10.json.
    agent_model     Ollama model tag for the Active Agent.
    judge_model     Ollama model tag for the judge.
    agent_url       Ollama base URL for the Active Agent.
    observer_url    Ollama base URL for the Observer.
    curator_url     Ollama base URL for the Curator.
    verbose         When True, print progress and results to stdout.

    Returns
    -------
    dict with keys:
        conv_id          str   — conversation index as a string
        turns_ingested   int   — total turns in the conversation
        pairs_ingested   int   — Q/A pairs pushed into the pipeline
        questions        int   — number of QA questions answered
        ingest_time      float — seconds spent on ingestion
        scores_by_cat    dict[int, list[int]]  — per-category judge scores
        all_scores       list[int]             — every score, flat
    """
    # ── Load dataset ──────────────────────────────────────────────────────────
    conversations = load_locomo(data_path)
    if conv_idx < 0 or conv_idx >= len(conversations):
        raise IndexError(
            f"conv_idx {conv_idx} is out of range. "
            f"Dataset has {len(conversations)} conversations (0-{len(conversations)-1})."
        )
    conv: LoCoMoConversation = conversations[conv_idx]
    pairs = _make_pairs(conv.turns)

    chroma_path = CHROMA_PATH.format(idx=conv_idx)

    if verbose:
        _print_divider("=")
        print(f"  LOCOMO BENCHMARK — Conversation {conv.id}")
        _print_divider("=")
        print(f"\n  Agent model : {agent_model}")
        print(f"  Judge model : {judge_model}")
        print(f"  Turns       : {len(conv.turns)}  →  {len(pairs)} pairs")
        print(f"  Questions   : {len(conv.questions)}")
        print(f"  Chroma path : {chroma_path}")
        print()

    # ── Phase 1: Ingest ───────────────────────────────────────────────────────
    if verbose:
        _print_divider("=")
        print("  PHASE 1 — Ingestion")
        _print_divider("=")
        print()

    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    pipeline = Pipeline(
        backend=OllamaBackend(model=agent_model, base_url=agent_url),
        chroma_path=chroma_path,
        observer_backend=OllamaBackend(model=agent_model, base_url=observer_url),
        curator_backend=OllamaBackend(model=agent_model, base_url=curator_url),
    )

    turn_times: list[float] = []
    ingest_start = time.time()

    for i, (user_text, asst_text) in enumerate(pairs, 1):
        t0 = time.time()
        pipeline.ingest(user_text, asst_text)
        t_turn = time.time() - t0
        turn_times.append(t_turn)

        if verbose:
            bucket       = pipeline.bucket
            depth        = len(bucket.recent_messages)
            stored_count = pipeline.retrieval._collection.count()
            print(
                f"  [{i:>4}/{len(pairs)}]  "
                f"depth={depth}/{bucket._max_recent}  "
                f"stored={stored_count:>3}  "
                f"{t_turn:>5.1f}s  "
                f"U: \"{_truncate(user_text, 40)}\""
            )

    ingest_elapsed = time.time() - ingest_start

    if verbose and turn_times:
        avg_t = sum(turn_times) / len(turn_times)
        max_t = max(turn_times)
        max_i = turn_times.index(max_t) + 1
        print()
        _print_divider("-")
        print(f"  Ingestion complete")
        print(f"  Total time   : {ingest_elapsed:.1f}s  ({len(pairs)} pairs)")
        print(f"  Avg per pair : {avg_t:.2f}s")
        print(f"  Slowest pair : {max_t:.1f}s  (pair {max_i})")
        print(f"  Chroma items : {pipeline.retrieval._collection.count()}")
        print(f"  Summary set  : {'yes' if pipeline.bucket.summary else 'no'}")
        _print_divider("-")
        print()

    # ── Phase 2: Answer questions ─────────────────────────────────────────────
    if verbose:
        _print_divider("=")
        print("  PHASE 2 — Answering questions")
        _print_divider("=")
        print()

    responses: list[str] = []
    answer_times: list[float] = []

    for i, q in enumerate(conv.questions, 1):
        if verbose:
            print(
                f"  [{i:>3}/{len(conv.questions)}] "
                f"[{q.category_label:<11}]  "
                f"{_truncate(q.question, 50)}"
            )

        t0 = time.time()
        # build_context injects memories without committing to the stack;
        # calling the backend directly avoids polluting the conversation store.
        ctx      = pipeline.build_context(q.question)
        response = pipeline._backend.chat(ctx)
        t_ans    = time.time() - t0

        responses.append(response)
        answer_times.append(t_ans)

        if verbose:
            print(f"         {t_ans:.1f}s  →  {_truncate(response, 60)}")

    if verbose:
        print()

    # ── Phase 3: Judge scoring ────────────────────────────────────────────────
    if verbose:
        _print_divider("=")
        print("  PHASE 3 — Judge scoring")
        _print_divider("=")
        print()

    scores_by_cat: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
    all_scores: list[int] = []

    for i, (q, resp) in enumerate(zip(conv.questions, responses), 1):
        if verbose:
            print(
                f"  [{i:>3}/{len(conv.questions)}] "
                f"[{q.category_label:<11}]  "
                f"{_truncate(q.question, 45)}",
                end="  ",
                flush=True,
            )

        score = _judge(
            question=q.question,
            ground_truth=q.answer,
            response=resp,
            model=judge_model,
            base_url=agent_url,
        )

        all_scores.append(score)
        if q.category in scores_by_cat:
            scores_by_cat[q.category].append(score)

        if verbose:
            bar = "█" * score + "░" * (5 - score)
            print(f"score={score}  [{bar}]")

    if verbose:
        print()

    # ── Results ───────────────────────────────────────────────────────────────
    if verbose:
        _print_divider("=")
        print("  RESULTS BY CATEGORY")
        _print_divider("=")
        print()

        for cat_id, label in CATEGORY_LABELS.items():
            cat_scores = scores_by_cat.get(cat_id, [])
            avg = _safe_avg(cat_scores)
            if avg is None:
                print(f"  {label:<12} (cat {cat_id}): —  (no questions)")
            else:
                print(
                    f"  {label:<12} (cat {cat_id}): "
                    f"avg {avg:.1f} / 5   "
                    f"({len(cat_scores)} questions)"
                )

        overall = _safe_avg(all_scores)
        print()
        print(
            f"  {'Overall':<17}: "
            f"avg {overall:.1f} / 5   "
            f"({len(all_scores)} questions)"
        )
        print()
        _print_divider("-")
        print(f"  Turns ingested  : {len(conv.turns)}")
        print(f"  Pairs ingested  : {len(pairs)}")
        print(f"  Ingest time     : {ingest_elapsed:.1f}s")
        if answer_times:
            print(f"  Avg answer time : {sum(answer_times)/len(answer_times):.1f}s")
        _print_divider("=")
        print()

    return {
        "conv_id":       conv.id,
        "turns_ingested": len(conv.turns),
        "pairs_ingested": len(pairs),
        "questions":     len(conv.questions),
        "ingest_time":   ingest_elapsed,
        "scores_by_cat": scores_by_cat,
        "all_scores":    all_scores,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LoCoMo QA benchmark for a single conversation."
    )
    parser.add_argument(
        "--conversation", "-c",
        type=int,
        default=CONVERSATION_IDX,
        help="Conversation index to benchmark (0-9). Default: 0.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DATA_PATH,
        help=f"Path to locomo10.json. Default: {DATA_PATH}",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default=AGENT_MODEL,
        help=f"Ollama model for the Active Agent. Default: {AGENT_MODEL}",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=JUDGE_MODEL,
        help=f"Ollama model for the judge. Default: {JUDGE_MODEL}",
    )
    parser.add_argument(
        "--agent-url",
        type=str,
        default=AGENT_URL,
        help=f"Ollama base URL for the Active Agent. Default: {AGENT_URL}",
    )
    parser.add_argument(
        "--observer-url",
        type=str,
        default=OBSERVER_URL,
        help=f"Ollama base URL for the Observer. Default: {OBSERVER_URL}",
    )
    parser.add_argument(
        "--curator-url",
        type=str,
        default=CURATOR_URL,
        help=f"Ollama base URL for the Curator. Default: {CURATOR_URL}",
    )

    args = parser.parse_args()

    run_conversation(
        conv_idx=args.conversation,
        data_path=args.data,
        agent_model=args.agent_model,
        judge_model=args.judge_model,
        agent_url=args.agent_url,
        observer_url=args.observer_url,
        curator_url=args.curator_url,
        verbose=True,
    )


if __name__ == "__main__":
    main()
