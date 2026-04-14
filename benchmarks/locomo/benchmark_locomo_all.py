"""
benchmark_locomo_all.py

Runs the LoCoMo QA benchmark across all 10 conversations and prints an
aggregate results table.

Each conversation gets its own isolated Chroma directory so runs cannot
contaminate each other.  Each run cleans its own Chroma directory before
starting, so rerunning produces fresh results every time.

Usage
-----
    python benchmarks/locomo/benchmark_locomo_all.py

    # Use a different model
    python benchmarks/locomo/benchmark_locomo_all.py --agent-model llama3:8b

    # Run only conversations 0-2 (useful for quick testing)
    python benchmarks/locomo/benchmark_locomo_all.py --conversations 0 1 2

Dataset
-------
    Clone https://github.com/snap-research/locomo and place data/locomo10.json at
    benchmarks/locomo/data/locomo10.json before running.
"""

from __future__ import annotations

import os
import sys
import argparse

os.environ.setdefault("OLLAMA_KEEP_ALIVE", "10m")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.dirname(__file__))

from loader import CATEGORY_LABELS
from benchmark_locomo import (
    run_conversation,
    DATA_PATH,
    AGENT_MODEL,
    JUDGE_MODEL,
    AGENT_URL,
    OBSERVER_URL,
    CURATOR_URL,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_divider(char: str = "-", width: int = 78) -> None:
    print(char * width)


def _safe_avg(scores: list[int]) -> float | None:
    return sum(scores) / len(scores) if scores else None


def _fmt(val: float | None) -> str:
    """Format a score as '3.4' or '  —' if None."""
    return f"{val:.1f}" if val is not None else "  —"


# ── Aggregate runner ──────────────────────────────────────────────────────────

def run_all(
    conv_indices: list[int],
    data_path: str = DATA_PATH,
    agent_model: str = AGENT_MODEL,
    judge_model: str = JUDGE_MODEL,
    agent_url: str = AGENT_URL,
    observer_url: str = OBSERVER_URL,
    curator_url: str = CURATOR_URL,
) -> None:
    """
    Run the LoCoMo benchmark for each conversation in conv_indices and print
    a combined summary table.
    """
    _print_divider("=")
    print("  LOCOMO BENCHMARK — ALL CONVERSATIONS")
    _print_divider("=")
    print(f"\n  Agent model : {agent_model}")
    print(f"  Judge model : {judge_model}")
    print(f"  Conversations: {conv_indices}")
    print()

    all_results: list[dict] = []

    for idx in conv_indices:
        print(f"\n{'=' * 78}")
        print(f"  Running conversation {idx} ...")
        print(f"{'=' * 78}\n")
        try:
            result = run_conversation(
                conv_idx=idx,
                data_path=data_path,
                agent_model=agent_model,
                judge_model=judge_model,
                agent_url=agent_url,
                observer_url=observer_url,
                curator_url=curator_url,
                verbose=True,
            )
            all_results.append(result)
        except Exception as exc:
            print(f"  [ERROR] Conversation {idx} failed: {exc}")
            all_results.append({
                "conv_id":        str(idx),
                "turns_ingested": 0,
                "pairs_ingested": 0,
                "questions":      0,
                "ingest_time":    0.0,
                "scores_by_cat":  {1: [], 2: [], 3: [], 4: []},
                "all_scores":     [],
                "error":          str(exc),
            })

    if not all_results:
        print("No results to summarise.")
        return

    # ── Aggregate table ───────────────────────────────────────────────────────
    print("\n")
    _print_divider("=")
    print("  AGGREGATE RESULTS")
    _print_divider("=")
    print()

    # Column widths
    W_CONV  = 6
    W_QS    = 10
    W_CAT   = 7    # one column per category (1-4)
    W_OVR   = 9
    W_INGEST= 8

    cat_ids = sorted(CATEGORY_LABELS.keys())

    # Header
    header_parts = [
        f"  {'Conv':<{W_CONV}}",
        f"{'Qs':>{W_QS}}",
    ]
    for cat_id in cat_ids:
        label = f"Cat{cat_id}"
        header_parts.append(f"{label:>{W_CAT}}")
    header_parts.append(f"{'Overall':>{W_OVR}}")
    header_parts.append(f"{'Ingest':>{W_INGEST}}")
    header = "  ".join(header_parts)
    print(header)
    _print_divider("-", len(header) + 2)

    # Per-conversation rows
    agg_by_cat: dict[int, list[int]] = {c: [] for c in cat_ids}
    agg_all: list[int] = []
    agg_ingest: list[float] = []

    for r in all_results:
        error_flag = "  [failed]" if r.get("error") else ""
        row_parts = [
            f"  {r['conv_id']:<{W_CONV}}",
            f"{r['questions']:>{W_QS}}",
        ]
        for cat_id in cat_ids:
            cat_scores = r["scores_by_cat"].get(cat_id, [])
            avg = _safe_avg(cat_scores)
            row_parts.append(f"{_fmt(avg):>{W_CAT}}")
            agg_by_cat[cat_id].extend(cat_scores)

        overall = _safe_avg(r["all_scores"])
        row_parts.append(f"{_fmt(overall):>{W_OVR}}")
        row_parts.append(f"{int(r['ingest_time']):>{W_INGEST - 1}}s")

        agg_all.extend(r["all_scores"])
        if r["ingest_time"] > 0:
            agg_ingest.append(r["ingest_time"])

        print("  ".join(row_parts) + error_flag)

    # Totals / averages row
    _print_divider("-", len(header) + 2)
    avg_parts = [
        f"  {'AVERAGE':<{W_CONV}}",
        f"{len(agg_all):>{W_QS}}",
    ]
    for cat_id in cat_ids:
        avg = _safe_avg(agg_by_cat[cat_id])
        avg_parts.append(f"{_fmt(avg):>{W_CAT}}")
    overall_avg = _safe_avg(agg_all)
    avg_parts.append(f"{_fmt(overall_avg):>{W_OVR}}")
    avg_ingest  = _safe_avg(agg_ingest)
    avg_parts.append(f"{_fmt(avg_ingest):>{W_INGEST - 1}}s" if avg_ingest else f"{'—':>{W_INGEST}}")
    print("  ".join(avg_parts))
    _print_divider("=", len(header) + 2)

    # Per-category summary block
    print()
    print("  CATEGORY BREAKDOWN (all conversations combined)")
    print()
    for cat_id, label in CATEGORY_LABELS.items():
        scores = agg_by_cat[cat_id]
        avg    = _safe_avg(scores)
        if avg is None:
            print(f"  {label:<12} (cat {cat_id}): —  (no questions)")
        else:
            print(
                f"  {label:<12} (cat {cat_id}): "
                f"avg {avg:.2f} / 5   "
                f"({len(scores)} questions)"
            )

    overall_avg = _safe_avg(agg_all)
    print()
    if overall_avg is not None:
        print(f"  Overall              : avg {overall_avg:.2f} / 5   ({len(agg_all)} questions)")
    _print_divider("=")
    print("  Benchmark complete.")
    _print_divider("=")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LoCoMo QA benchmark across all 10 conversations."
    )
    parser.add_argument(
        "--conversations",
        nargs="+",
        type=int,
        default=list(range(10)),
        help="Conversation indices to run. Default: 0 through 9.",
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

    run_all(
        conv_indices=sorted(set(args.conversations)),
        data_path=args.data,
        agent_model=args.agent_model,
        judge_model=args.judge_model,
        agent_url=args.agent_url,
        observer_url=args.observer_url,
        curator_url=args.curator_url,
    )


if __name__ == "__main__":
    main()
