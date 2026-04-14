"""
generate_chat.py

Generates an artificial chat log for NIAH (needle-in-a-haystack) benchmarking.
Uses the configured LLM backend (reads .env via backend_from_env).

Large counts are split into batches so the model is never asked to produce
thousands of tokens in one call.  Each batch logs its progress to stdout.

Each batch call runs in a daemon thread so Ctrl+C is always responsive and
a configurable per-batch timeout aborts a hung call cleanly.  Completed
batches are saved to disk immediately, so a partial run is never lost.

Usage
-----
    # Defaults: 75 pairs, write to benchmarks/contexts/niah_chat_75.py
    python benchmarks/generate_chat.py

    # Custom size (20 batches of 50)
    python benchmarks/generate_chat.py --pairs 1000

    # With a passkey needle injected at 30% depth
    python benchmarks/generate_chat.py --pairs 100 --passkey "ALPHA-7731-DELTA" --position 0.3

    # Smaller batches if your model truncates at 50 pairs
    python benchmarks/generate_chat.py --pairs 1000 --batch-size 25

    # Longer timeout for slow hardware (seconds per batch, default 300)
    python benchmarks/generate_chat.py --pairs 1000 --timeout 600

    # Custom output file
    python benchmarks/generate_chat.py --pairs 50 --output benchmarks/contexts/my_chat.py

Arguments
---------
    --pairs       Number of user/assistant pairs to generate. Default 75.
    --batch-size  Pairs per LLM call. Default 50. Lower if model truncates.
    --timeout     Seconds to wait per batch before aborting. Default 300.
    --retries     Times to retry a timed-out batch before giving up. Default 1.
    --output      Output file path. Default: benchmarks/contexts/niah_chat_<N>.py
    --passkey     Optional passkey string to inject as a needle in the haystack.
    --position    Where to inject the passkey (0.0 = start, 1.0 = end). Default 0.5.
"""

from __future__ import annotations

import argparse
import re
import sys
import os
import threading

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_prompt(num_pairs: int) -> str:
    return f"""You are generating a realistic, long-form technical chat log between a software engineer (user) and an AI assistant (assistant).

OUTPUT FORMAT — follow this exactly, no deviations:
- Output raw Python list syntax only
- Each message is a dict with "role" and "content" keys
- role is either "user" or "assistant"
- content is a Python string using parenthesized string concatenation for long lines
- Add a comment with the message index after the opening brace: {{   # <index>
- No markdown, no explanation, no ```python fences — raw Python list entries only

EXAMPLE OF CORRECT FORMAT:
        {{   # 0
            "role": "user",
            "content": (
                "I want to settle the frontend architecture before we write a line of "
                "code. I've been burned by SPA sprawl before -- separate build pipelines, "
                "client-server state sync bugs, two deployment artifacts to keep in step."
            ),
        }},
        {{   # 1
            "role": "assistant",
            "content": (
                "LiveView has matured. Real-time updates, minimal JavaScript, one "
                "deployment artifact, server-side rendering by default. "
                "What's the backend language you have in mind?"
            ),
        }},

CONTENT GUIDELINES:
- The conversation should meander naturally across multiple technical topics
- Topics to weave through: database schema design, deployment strategy, API design,
  authentication, caching, testing philosophy, CI/CD, monitoring, frontend architecture
- Each message should be 2-6 sentences. Vary the length naturally.
- The assistant sometimes asks follow-up questions, sometimes gives opinions
- Include occasional tangents, topic revisits, and mild disagreements
- Do NOT resolve every topic cleanly — let threads drop and resurface
- Generate exactly {num_pairs} message pairs ({num_pairs * 2} total dict entries)
- Start your output with [
- End your output with ]
"""


def _build_continuation_prompt(num_pairs: int, start_index: int, last_messages: str) -> str:
    return f"""You are continuing a long technical chat log between a software engineer (user) and an AI assistant (assistant).

The conversation so far ended with these messages:
{last_messages}

IMPORTANT: The messages above are context only — do NOT include them in your output.
Your first output message must have index {start_index} and continue from where the context ends.

OUTPUT FORMAT — follow this exactly, no deviations:
- Output raw Python list syntax only
- Each message is a dict with "role" and "content" keys
- role is either "user" or "assistant"
- content is a Python string using parenthesized string concatenation for long lines
- Add a comment with the message index after the opening brace: {{   # <index>
- Indices start at {start_index} and count up
- No markdown, no explanation, no ```python fences — raw Python list entries only

CONTENT GUIDELINES:
- Continue naturally from the last message — don't restart or summarize
- Keep meandering through technical topics: database schema, deployment, API design,
  authentication, caching, testing, CI/CD, monitoring, frontend architecture
- Each message should be 2-6 sentences. Vary the length naturally.
- Include tangents, topic revisits, and mild disagreements
- Generate exactly {num_pairs} message pairs ({num_pairs * 2} total dict entries)
- Start your output with [
- End your output with ]
"""


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def _extract_last_messages(raw_output: str, n: int = 3) -> str:
    """Return the raw text of the last `n` message dicts from raw_output."""
    matches = list(re.finditer(r'\{', raw_output))
    if len(matches) < n:
        return raw_output
    start = matches[-n].start()
    return raw_output[start:].rstrip().rstrip("]").rstrip(",").strip()


def _merge_chunks(chunks: list[str]) -> str:
    """Merge multiple raw '[...]' batch strings into a single '[...]' string."""
    inner_parts = []
    for chunk in chunks:
        inner = chunk.strip()
        if inner.startswith("["):
            inner = inner[1:]
        if inner.endswith("]"):
            inner = inner[:-1]
        inner_parts.append(inner.strip().rstrip(","))
    return "[\n" + ",\n".join(inner_parts) + "\n]"


_QUOTE_MAP = str.maketrans({
    '\u201c': '\\"',  # "  left double quotation mark  → escaped \" (valid inside "...")
    '\u201d': '\\"',  # "  right double quotation mark → escaped \"
    '\u201e': '\\"',  # „  double low-9 quotation mark → escaped \"
    '\u2033': '\\"',  # ″  double prime                → escaped \"
    '\u2018': "'",    # '  left single quotation mark
    '\u2019': "'",    # '  right single quotation mark
    '\u201b': "'",    # ‛  single high-reversed-9 quotation mark
    '\u2032': "'",    # ′  prime
})


def _normalize_quotes(text: str) -> str:
    """Replace Unicode smart/curly quotes with straight ASCII equivalents."""
    return text.translate(_QUOTE_MAP)


def _clean_chunk(raw: str) -> str:
    """
    Strip markdown code fences, normalize smart quotes, and discard any
    explanatory preamble/postamble.  Returns just the '[...]' list content.

    Models often wrap output in ```python ... ``` and emit Unicode curly
    quotes despite being told not to.
    """
    # Remove ``` fences with or without a language tag (```python, ```py, etc.)
    raw = re.sub(r'```[a-z]*\n?', '', raw).strip()
    # Replace curly/smart quotes with straight ASCII equivalents so the
    # generated Python file is valid and parseable.
    raw = _normalize_quotes(raw)
    # Extract from first '[' to last ']' to discard any stray leading/trailing text
    start = raw.find('[')
    end   = raw.rfind(']')
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return raw


# ---------------------------------------------------------------------------
# Timeout-safe backend call
# ---------------------------------------------------------------------------

def _chat_with_timeout(backend, messages: list[dict], timeout_secs: int) -> str:
    """
    Call backend.chat() in a daemon thread so the main thread stays
    interruptible by Ctrl+C and a timeout can be enforced.

    Raises TimeoutError if the call doesn't complete within timeout_secs.
    Re-raises any exception thrown by the backend.
    """
    result: list[str | None] = [None]
    error:  list[BaseException | None] = [None]

    def _call() -> None:
        try:
            result[0] = backend.chat(messages)
        except Exception as exc:  # noqa: BLE001
            error[0] = exc

    thread = threading.Thread(target=_call, daemon=True)
    thread.start()

    try:
        thread.join(timeout=timeout_secs)
    except KeyboardInterrupt:
        # Re-raise so the caller's KeyboardInterrupt handler fires.
        raise

    if thread.is_alive():
        raise TimeoutError(
            f"Batch timed out after {timeout_secs}s. "
            "Ollama may be overloaded. Try --timeout 600 or restart Ollama."
        )

    if error[0] is not None:
        raise error[0]  # type: ignore[misc]

    return result[0]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Passkey injection
# ---------------------------------------------------------------------------

def _inject_passkey(raw_output: str, passkey: str, position: float) -> str:
    """
    Finds the message dict closest to `position` (0.0–1.0) and injects
    the passkey as a natural aside in a user turn.
    """
    matches = list(re.finditer(r'\{\s*#\s*(\d+)', raw_output))
    if not matches:
        print("Warning: could not find message markers for passkey injection.")
        return raw_output

    target_idx   = int(len(matches) * position)
    target_match = matches[min(target_idx, len(matches) - 1)]

    passkey_block = f"""        {{   # passkey_injection
            "role": "user",
            "content": (
                "Oh before I forget -- the access code for the staging vault is "
                "{passkey}. "
                "Remind me of that if I ask later. Anyway, back to what we were saying."
            ),
        }},
"""
    insert_pos = target_match.start()
    return raw_output[:insert_pos] + passkey_block + raw_output[insert_pos:]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(filepath: str) -> None:
    """Quick sanity check — exec the file and count messages."""
    try:
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        namespace: dict = {}
        exec(source, namespace)  # noqa: S102
        log = namespace.get("CHAT_LOG", [])
        print(f"Validation OK — {len(log)} messages parsed successfully.")
    except Exception as exc:
        print(f"Validation WARNING — could not parse output: {exc}")
        print("You may need to manually clean up the generated file.")


# ---------------------------------------------------------------------------
# Partial save
# ---------------------------------------------------------------------------

def _save(output_file: str, batches: list[str], num_pairs: int, batch_size: int,
          passkey: str | None, passkey_position: float, partial: bool = False) -> None:
    """Write whatever batches exist to disk. Marks the file as partial if incomplete."""
    raw_output = _merge_chunks(batches)

    if passkey and not partial:
        print(f"Injecting passkey at position {passkey_position:.0%}...")
        raw_output = _inject_passkey(raw_output, passkey, passkey_position)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        if partial:
            f.write("# PARTIAL — generation was interrupted before completion\n")
        else:
            f.write("# Auto-generated chat log for NIAH stress test\n")
        f.write(f"# Pairs: {num_pairs} | Batch size: {batch_size}\n")
        if passkey and not partial:
            f.write(f"# Passkey: '{passkey}' | Position: {passkey_position:.0%}\n")
        f.write("\n")
        if passkey and not partial:
            f.write(f'PASSKEY = "{passkey}"\n')
            f.write("\n")
        f.write("CHAT_LOG = ")
        f.write(raw_output)
        f.write("\n")

    label = "Partial file" if partial else "Written"
    print(f"{label} → {output_file}")
    _validate(output_file)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_chat_log(
    output_file: str,
    num_pairs: int,
    passkey: str | None,
    passkey_position: float,
    batch_size: int = 50,
    timeout_secs: int = 300,
    retries: int = 1,
) -> None:
    from active_memory.config import backend_from_env

    backend = backend_from_env()
    print(f"Backend: {backend!r}")

    total_batches = (num_pairs + batch_size - 1) // batch_size
    print(f"Generating {num_pairs} pairs in {total_batches} batch(es) of up to {batch_size}...")
    print(f"Per-batch timeout: {timeout_secs}s  |  Retries: {retries}  |  Press Ctrl+C to abort and save progress.\n")

    batches:    list[str] = []
    pairs_done: int       = 0

    try:
        for batch_num in range(1, total_batches + 1):
            this_batch = min(batch_size, num_pairs - pairs_done)

            if batch_num == 1:
                prompt = _build_prompt(this_batch)
            else:
                last_msgs = _extract_last_messages(batches[-1])
                prompt    = _build_continuation_prompt(this_batch, pairs_done * 2, last_msgs)

            msg_start = pairs_done * 2
            msg_end   = (pairs_done + this_batch) * 2 - 1
            print(
                f"  [batch {batch_num}/{total_batches}] generating {this_batch} pairs "
                f"(messages {msg_start}–{msg_end})...",
                end=" ",
                flush=True,
            )

            chunk = None
            for attempt in range(retries + 1):
                try:
                    chunk = _chat_with_timeout(
                        backend,
                        [{"role": "user", "content": prompt}],
                        timeout_secs,
                    )
                    chunk = _clean_chunk(chunk)
                    break  # success
                except TimeoutError as exc:
                    if attempt < retries:
                        print(f"\n  TIMEOUT — retrying ({attempt + 1}/{retries})...", end=" ", flush=True)
                    else:
                        print(f"\n  TIMEOUT — no more retries.")
                        print(f"  {exc}")
                        if batches:
                            print(f"\nSaving {pairs_done} completed pairs before exiting...")
                            _save(output_file, batches, num_pairs, batch_size,
                                  passkey, passkey_position, partial=True)
                        sys.exit(1)

            batches.append(chunk)
            pairs_done += this_batch
            print(f"done  ({pairs_done}/{num_pairs} pairs complete)")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        if batches:
            print(f"Saving {pairs_done} completed pairs...")
            _save(output_file, batches, num_pairs, batch_size,
                  passkey, passkey_position, partial=True)
        else:
            print("No batches completed — nothing to save.")
        sys.exit(0)

    _save(output_file, batches, num_pairs, batch_size, passkey, passkey_position, partial=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an artificial chat log for NIAH benchmarking."
    )
    parser.add_argument(
        "--pairs", "-n",
        type=int,
        default=75,
        help="Number of user/assistant pairs to generate. Default: 75.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Pairs per LLM call. Default: 50. Lower if model truncates output.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Seconds to wait per batch before aborting. Default: 300.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Times to retry a timed-out batch before giving up. Default: 1.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=(
            "Output file path. "
            "Default: benchmarks/contexts/niah_chat_<N>.py"
        ),
    )
    parser.add_argument(
        "--passkey",
        type=str,
        default=None,
        help="Passkey string to inject as a needle. Omit to skip injection.",
    )
    parser.add_argument(
        "--position",
        type=float,
        default=0.5,
        help="Fractional position to inject passkey (0.0–1.0). Default: 0.5.",
    )

    args = parser.parse_args()

    output_file = args.output or os.path.join(
        os.path.dirname(__file__), "contexts", f"niah_chat_{args.pairs}.py"
    )

    generate_chat_log(
        output_file=output_file,
        num_pairs=args.pairs,
        passkey=args.passkey,
        passkey_position=args.position,
        batch_size=args.batch_size,
        timeout_secs=args.timeout,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
