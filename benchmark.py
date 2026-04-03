"""
benchmark.py

Compares two agents over a 30-message conversation:

  Agent A — Vanilla: raw Ollama call with a 5-message rolling window.
  Agent B — Active Memory: full Observer/Curator/Librarian/Reporter/Conductor/ActiveAgent pipeline.

Both agents receive the same user turns. Recall questions near the end ask each
agent to reproduce decisions that were established in the first 8 turns — well
outside Agent A's 5-message window by that point.
"""

import textwrap
import ollama

from observer import Observer
from curator import Curator
from librarian import Librarian
from reporter import Reporter
from conductor import Conductor
from active_agent import ActiveAgent


# ---------------------------------------------------------------------------
# Vanilla agent — no memory system, 5-message rolling window
# ---------------------------------------------------------------------------

class VanillaAgent:
    def __init__(self, model: str = "llama3.2", window: int = 5):
        self._model = model
        self._window = window
        self._history: list[dict] = []

    def chat(self, message: str) -> str:
        self._history.append({"role": "user", "content": message})
        context = self._history[-(self._window):]
        response = ollama.chat(model=self._model, messages=context)
        reply = response.message.content.strip()
        self._history.append({"role": "assistant", "content": reply})
        return reply


# ---------------------------------------------------------------------------
# Conversation — 30 user turns
# Turns 0-7:   four explicit decisions locked in (language, DB, deploy, arch/auth)
# Turns 8-21:  new topics — schema, API, caching, error handling, testing, ops
# Turns 22-29: recall and consistency challenge
# ---------------------------------------------------------------------------

CONVERSATION: list[tuple[str, bool]] = [
    # (user message, is_recall_turn)

    # --- Decision block ---
    ("Let's design a URL shortener service. I want to build it in Go.", False),
    ("Database: PostgreSQL. And I want raw SQL only — no ORM, no query builder.", False),
    ("Deployment constraint: single Linux VPS. No Docker, no Kubernetes, no containers of any kind.", False),
    ("Architecture: monolith. No microservices. Everything in one binary.", False),
    ("Auth: API keys only. No OAuth, no JWT, no sessions. Just a key in the header.", False),
    ("Third-party services are also off the table. No external analytics, no managed queues — everything runs on our box.", False),
    ("Good. Let's call those the foundations: Go, PostgreSQL raw SQL, monolith, single VPS, API keys, no third-party.", False),
    ("Alright. Now let's start designing. Walk me through the top-level package structure for a Go monolith.", False),

    # --- Middle section: new topics ---
    ("What should the database schema look like? We need to store short codes, original URLs, and creation timestamps.", False),
    ("Should we index the short_code column or use it as the primary key?", False),
    ("Let's use it as the primary key. Simpler. What does the redirect handler look like in Go?", False),
    ("How should we handle 301 vs 302 redirects? I want to think about caching implications.", False),
    ("Let's go with 302 for now so we don't lock ourselves into cached responses during development.", False),
    ("What's a good rate limiting approach that doesn't require Redis or any external service?", False),
    ("An in-memory token bucket per API key sounds right. How do we expire old entries so we don't leak memory?", False),
    ("Let's add a background goroutine that sweeps expired buckets every 60 seconds.", False),
    ("Good. Now let's talk about analytics. We want to count redirects per short code. How do we do this without hammering the DB on every hit?", False),
    ("A write-behind cache with periodic flushes makes sense. What does the flush goroutine look like?", False),
    ("Let's move to error handling. What's our convention for returning errors from HTTP handlers?", False),
    ("Structured JSON errors with a code, message, and request ID. How do we propagate the request ID through the stack?", False),
    ("Store it in context.Context and pass it down. What logging library should we use?", False),
    ("Let's use zerolog. Structured JSON logs, zero allocation, easy to grep. Agreed.", False),

    # --- Recall and consistency challenge ---
    ("We've covered a lot of ground. Before we go further — remind me what language and database we decided on, and why raw SQL specifically.", True),
    ("And what was the deployment constraint? I want to make sure we're not designing anything that assumes containers.", True),
    ("Confirm the auth approach. I want to make sure we haven't drifted toward JWT anywhere.", True),
    ("We talked about rate limiting without Redis. What was the reason for that constraint?", True),
    ("If someone on the team suggested we add a managed queue service for the analytics flush, what would you tell them based on what we agreed?", True),
    ("Let's say we're doing a code review. A PR comes in that uses an ORM. Is that consistent with our decisions? Why or why not?", True),
    ("Final check — summarize all the foundational decisions we made at the start of this conversation.", True),
    ("One last question: if we had to migrate this to a different VPS provider, what would that involve given our architectural constraints?", True),
]

RECALL_INDICES = {i for i, (_, is_recall) in enumerate(CONVERSATION) if is_recall}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def wrap(text: str, width: int = 55) -> list[str]:
    if not text:
        return ["(no response)"]
    lines = []
    for paragraph in text.splitlines():
        if paragraph.strip():
            lines.extend(textwrap.wrap(paragraph, width) or [paragraph])
        else:
            lines.append("")
    return lines or ["(no response)"]


def print_side_by_side(turn: int, user_msg: str, vanilla: str, active: str) -> None:
    col = 57
    sep = " | "

    header = f"  RECALL TURN {turn + 1:02d}"
    print(f"\n{'=' * (col * 2 + len(sep))}")
    print(header)
    print(f"{'=' * (col * 2 + len(sep))}")
    print(f"USER: {user_msg}")
    print(f"{'-' * (col * 2 + len(sep))}")

    label_a = "AGENT A (Vanilla — 5-msg window)".center(col)
    label_b = "AGENT B (Active Memory)".center(col)
    print(f"{label_a}{sep}{label_b}")
    print(f"{'-' * col}{sep}{'-' * col}")

    lines_a = wrap(vanilla, col - 2)
    lines_b = wrap(active, col - 2)
    max_lines = max(len(lines_a), len(lines_b))
    lines_a += [""] * (max_lines - len(lines_a))
    lines_b += [""] * (max_lines - len(lines_b))

    for la, lb in zip(lines_a, lines_b):
        print(f"{la:<{col}}{sep}{lb}")


def print_turn(turn: int, role: str, text: str, agent_label: str) -> None:
    print(f"\n[Turn {turn + 1:02d}] {agent_label} | {role.upper()}: {text[:120]}{'...' if len(text) > 120 else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Initialising agents...")

    # Agent A — vanilla
    vanilla_agent = VanillaAgent()

    # Agent B — full Active Memory pipeline
    observer  = Observer()
    curator   = Curator()
    librarian = Librarian()
    reporter  = Reporter(librarian)
    conductor = Conductor(observer, curator, librarian, reporter)
    active_agent = ActiveAgent(conductor)

    vanilla_recall_responses: dict[int, str] = {}
    active_recall_responses:  dict[int, str] = {}

    print(f"\nRunning {len(CONVERSATION)}-turn conversation...\n")

    for i, (user_msg, is_recall) in enumerate(CONVERSATION):
        print(f"\n--- Turn {i + 1:02d} {'[RECALL]' if is_recall else ''} ---")
        print(f"USER: {user_msg}")

        reply_a = vanilla_agent.chat(user_msg)
        reply_b = active_agent.chat(user_msg)

        print(f"  A: {reply_a[:100]}{'...' if len(reply_a) > 100 else ''}")
        print(f"  B: {reply_b[:100]}{'...' if len(reply_b) > 100 else ''}")

        if is_recall:
            vanilla_recall_responses[i] = reply_a
            active_recall_responses[i]  = reply_b

        if observer.trimmings:
            print(f"  [Observer trimmings: {len(observer.trimmings)}]")

    # ---------------------------------------------------------------------------
    # Side-by-side recall comparison
    # ---------------------------------------------------------------------------

    print(f"\n\n{'#' * 120}")
    print("  RECALL COMPARISON — Agent A (Vanilla) vs Agent B (Active Memory)")
    print(f"{'#' * 120}")

    for i in sorted(RECALL_INDICES):
        user_msg, _ = CONVERSATION[i]
        print_side_by_side(
            turn=i,
            user_msg=user_msg,
            vanilla=vanilla_recall_responses.get(i, ""),
            active=active_recall_responses.get(i, ""),
        )

    # ---------------------------------------------------------------------------
    # Active Memory retrieval logs
    # ---------------------------------------------------------------------------

    print(f"\n\n{'#' * 120}")
    print("  ACTIVE MEMORY — Retrieval Logs")
    print(f"{'#' * 120}")

    logs = librarian.get_retrieval_logs()
    if logs:
        for memory_id, count in sorted(logs.items(), key=lambda x: -x[1]):
            print(f"  {memory_id}  recalled {count}x")
    else:
        print("  (no retrievals logged)")

    print(f"\n  Total memories stored: warm={librarian._collections['warm'].count()}  cold={librarian._collections['cold'].count()}")
    print(f"  Total Observer trimmings: {len(observer.trimmings)}")


if __name__ == "__main__":
    main()
