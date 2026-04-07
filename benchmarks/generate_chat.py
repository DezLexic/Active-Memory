"""
generate_chat.py

Generates an artificial chat log for NIAH (needle-in-a-haystack) benchmarking.
Uses the configured LLM backend (reads .env via backend_from_env).

Usage
-----
    # Defaults: 75 pairs, write to benchmarks/contexts/niah_chat_75.py
    python benchmarks/generate_chat.py

    # Custom size
    python benchmarks/generate_chat.py --pairs 120

    # With a passkey needle injected at 30% depth
    python benchmarks/generate_chat.py --pairs 100 --passkey "ALPHA-7731-DELTA" --position 0.3

    # Custom output file
    python benchmarks/generate_chat.py --pairs 50 --output benchmarks/contexts/my_chat.py

Arguments
---------
    --pairs     Number of user/assistant pairs to generate. Default 75.
    --output    Output file path. Default: benchmarks/contexts/niah_chat_<N>.py
    --passkey   Optional passkey string to inject as a needle in the haystack.
    --position  Where to inject the passkey (0.0 = start, 1.0 = end). Default 0.5.
"""

from __future__ import annotations

import argparse
import re
import sys
import os

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


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


def _inject_passkey(raw_output: str, passkey: str, position: float) -> str:
    """
    Finds the message dict closest to `position` (0.0–1.0) and injects
    the passkey as a natural aside in a user turn.
    """
    matches = list(re.finditer(r'\{\s*#\s*(\d+)', raw_output))
    if not matches:
        print("Warning: could not find message markers for passkey injection.")
        return raw_output

    target_idx  = int(len(matches) * position)
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


def _validate(filepath: str) -> None:
    """Quick sanity check — exec the file and count messages."""
    try:
        with open(filepath) as f:
            source = f.read()
        namespace: dict = {}
        exec(source, namespace)  # noqa: S102
        log = namespace.get("CHAT_LOG", [])
        print(f"Validation OK — {len(log)} messages parsed successfully.")
    except Exception as exc:
        print(f"Validation WARNING — could not parse output: {exc}")
        print("You may need to manually clean up the generated file.")


def generate_chat_log(
    output_file: str,
    num_pairs: int,
    passkey: str | None,
    passkey_position: float,
) -> None:
    from active_memory.config import backend_from_env

    backend = backend_from_env()
    print(f"Backend: {backend!r}")
    print(f"Generating {num_pairs} pairs ({num_pairs * 2} messages)...")

    prompt = _build_prompt(num_pairs)
    raw_output = backend.chat([{"role": "user", "content": prompt}]).strip()

    if passkey:
        print(f"Injecting passkey at position {passkey_position:.0%}...")
        raw_output = _inject_passkey(raw_output, passkey, passkey_position)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Auto-generated chat log for NIAH stress test\n")
        f.write(f"# Pairs: {num_pairs}\n")
        if passkey:
            f.write(f"# Passkey: '{passkey}' | Position: {passkey_position:.0%}\n")
        f.write("\n")
        f.write("CHAT_LOG = ")
        f.write(raw_output)
        f.write("\n")

    print(f"Written to {output_file}")
    _validate(output_file)


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
    )


if __name__ == "__main__":
    main()
