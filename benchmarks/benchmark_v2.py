# -*- coding: utf-8 -*-
import os
os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

"""
benchmark_v2.py

Two-part benchmark:

  1. Generate a 60-message fake chat log about building a SaaS product.
     - Messages  1-10:  five locked-in decisions (runtime, DB, frontend, deploy, auth)
     - Messages 11-45:  development discussion that dilutes early context
     - Messages 46-60:  recall / consistency questions (user-only; no scripted replies)

  2. Run two agents against every recall question and compare:

     Agent A — Vanilla:
       The full conversation history is passed as a proper messages array on every
       call — exactly how a real production model works today.  All 46 context
       messages are included as individual {role, content} objects, with the recall
       question appended as the final user message.  No summarisation, no retrieval,
       just the raw history handed to the model verbatim.

     Agent B — Active Memory:
       All 45 context messages are batch-processed through the full pipeline
       (Observer → Curator → Librarian, orchestrated by Conductor) to build
       the memory store.  Each recall question is then answered via ActiveAgent
       which assembles its context window from live summary + retrieved memories.
"""

import re
import time
import textwrap
import ollama

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from archive.observer_v1   import Observer
from archive.curator_v1    import Curator
from archive.librarian      import Librarian
from archive.reporter       import Reporter
from archive.conductor      import Conductor
from archive.active_agent_v1 import ActiveAgent


# ──────────────────────────────────────────────────────────────────────────────
# 1. CHAT LOG
#    Indices 0-44  → context  (user/assistant alternating, establishing 5 decisions
#                               then gradually drifting to new topics)
#    Indices 45-59 → recall   (user-only; no scripted assistant reply)
# ──────────────────────────────────────────────────────────────────────────────

CHAT_LOG: list[dict] = [

    # ── Decisions block (turns 1-10) ─────────────────────────────────────────

    {   # 0
        "role": "user",
        "content": (
            "Let's spec out the new SaaS product. Runtime and language first: "
            "I want Bun with TypeScript end-to-end. Not Node, not Deno. Bun specifically "
            "because of its built-in bundler, fast cold starts, and native TS support."
        ),
    },
    {   # 1
        "role": "assistant",
        "content": (
            "Bun + TypeScript it is. We get a single executable, native test runner, and "
            "we skip the tsconfig ceremony entirely. I'll assume bun:http for the server "
            "and bun:test for unit tests. What's the database?"
        ),
    },
    {   # 2
        "role": "user",
        "content": (
            "SQLite for the database. And I want raw SQL only — no Drizzle, no Prisma, "
            "no Kysley, no query builders at all. Just bun:sqlite with hand-written, "
            "parameterized queries. I've been burned by ORM abstractions before."
        ),
    },
    {   # 3
        "role": "assistant",
        "content": (
            "Raw SQL with bun:sqlite gives us full control and zero abstraction overhead. "
            "We'll keep a /db folder with migration files numbered sequentially. "
            "Parameterized queries everywhere — no string interpolation. "
            "What does the frontend look like?"
        ),
    },
    {   # 4
        "role": "user",
        "content": (
            "Frontend: HTMX with server-rendered HTML. No React, no Vue, no Svelte, "
            "no JS framework of any kind. Vanilla JS only for the handful of things "
            "HTMX can't handle. I want hypermedia all the way down."
        ),
    },
    {   # 5
        "role": "assistant",
        "content": (
            "HTMX pairs perfectly with a Bun HTTP server — we return HTML fragments "
            "on hx-get/hx-post, target specific DOM nodes, and the browser handles "
            "the swap. We'll serve a thin base layout and keep all dynamic content "
            "as partials. Deployment target?"
        ),
    },
    {   # 6
        "role": "user",
        "content": (
            "Fly.io, single machine. No autoscaling, no multi-region, no load balancer. "
            "One Fly machine with a persistent volume for the SQLite file. "
            "We stay single-instance until load actually demands otherwise."
        ),
    },
    {   # 7
        "role": "assistant",
        "content": (
            "Single Fly machine with a volume mount for /data/app.db — clean and simple. "
            "We set scale.min = 1, scale.max = 1 in fly.toml so it never autoscales. "
            "Fly handles TLS termination so we don't need to. Auth approach?"
        ),
    },
    {   # 8
        "role": "user",
        "content": (
            "Passkeys only — WebAuthn. No passwords, no OAuth providers, no magic links, "
            "no TOTP. Just passkeys. Every user registers a passkey; every login uses it. "
            "I know it limits older browsers but that's a deliberate trade-off."
        ),
    },
    {   # 9
        "role": "assistant",
        "content": (
            "WebAuthn passkeys it is. We store credential_id and public_key_cbor in "
            "SQLite, handle the registration and authentication ceremonies server-side, "
            "and rely on the browser's credential manager. Baseline support is good on "
            "modern browsers. Five decisions locked: Bun/TS, SQLite raw SQL, HTMX, "
            "Fly.io single instance, passkeys. Let's build."
        ),
    },

    # ── Development discussion (turns 11-45) ──────────────────────────────────

    {   # 10
        "role": "user",
        "content": "Okay. Let's start with the database schema. We need users, sessions, and passkey credentials.",
    },
    {   # 11
        "role": "assistant",
        "content": (
            "Three tables to start:\n"
            "  users(id TEXT PK, email TEXT UNIQUE, created_at INTEGER)\n"
            "  passkey_credentials(id TEXT PK, user_id TEXT FK, credential_id BLOB UNIQUE, "
            "public_key BLOB, sign_count INTEGER, created_at INTEGER)\n"
            "  sessions(id TEXT PK, user_id TEXT FK, expires_at INTEGER, created_at INTEGER)\n"
            "Migration 001_init.sql. Want me to draft the CREATE TABLE statements?"
        ),
    },
    {   # 12
        "role": "user",
        "content": "Yes, draft 001_init.sql. Also add a products table — users can create simple product listings.",
    },
    {   # 13
        "role": "assistant",
        "content": (
            "Added products(id TEXT PK, user_id TEXT FK, name TEXT, description TEXT, "
            "price_cents INTEGER, active INTEGER DEFAULT 1, created_at INTEGER). "
            "All foreign keys with ON DELETE CASCADE. "
            "PRAGMA foreign_keys = ON goes at the top of every connection open."
        ),
    },
    {   # 14
        "role": "user",
        "content": "What should our folder structure look like for the Bun project?",
    },
    {   # 15
        "role": "assistant",
        "content": (
            "src/\n"
            "  server.ts          ← entry point, route table\n"
            "  routes/            ← one file per resource (users, products, auth)\n"
            "  views/             ← HTML template functions (returns string)\n"
            "  db/\n"
            "    client.ts        ← opens the bun:sqlite connection\n"
            "    migrations/      ← 001_init.sql, 002_...\n"
            "  lib/               ← webauthn helpers, session helpers\n"
            "  public/            ← static assets (htmx.min.js, style.css)\n"
            "tests/               ← bun:test files\n"
            "fly.toml"
        ),
    },
    {   # 16
        "role": "user",
        "content": "How do we handle routing in Bun? I don't want an Express-style framework.",
    },
    {   # 17
        "role": "assistant",
        "content": (
            "Use Bun.serve() with a fetch handler and a simple manual router — "
            "just a Map<string, Handler> keyed on 'METHOD /path'. "
            "For dynamic segments like /products/:id, a small matchRoute() function "
            "with a regex or split-and-compare. No need for a framework; "
            "the router fits in ~40 lines."
        ),
    },
    {   # 18
        "role": "user",
        "content": "Let's talk about the WebAuthn registration flow. Walk me through the ceremony.",
    },
    {   # 19
        "role": "assistant",
        "content": (
            "Registration ceremony:\n"
            "1. Client calls POST /auth/register/begin with the user's email.\n"
            "2. Server generates a random challenge (32 bytes), stores it in the session, "
            "returns PublicKeyCredentialCreationOptions JSON.\n"
            "3. Browser calls navigator.credentials.create(), user touches the authenticator.\n"
            "4. Client POSTs the attestation response to /auth/register/complete.\n"
            "5. Server verifies: origin, challenge, rpIdHash, attestation. "
            "Stores credential_id + public_key_cbor in passkey_credentials.\n"
            "Use the @simplewebauthn/server package for the crypto heavy-lifting."
        ),
    },
    {   # 20
        "role": "user",
        "content": "And the authentication flow?",
    },
    {   # 21
        "role": "assistant",
        "content": (
            "Auth ceremony:\n"
            "1. Client hits POST /auth/login/begin. Server looks up credentials for the email "
            "(or sends an empty allowCredentials list for discoverable credentials), "
            "generates a new challenge, stashes it in session.\n"
            "2. Browser calls navigator.credentials.get(), user taps authenticator.\n"
            "3. Client POSTs assertion to /auth/login/complete.\n"
            "4. Server verifies signature against stored public key, increments sign_count "
            "(clone detection), creates a session row, sets a session cookie.\n"
            "Session cookie: HttpOnly, Secure, SameSite=Strict."
        ),
    },
    {   # 22
        "role": "user",
        "content": "What's our error handling convention for HTTP responses?",
    },
    {   # 23
        "role": "assistant",
        "content": (
            "All errors return JSON: { error: string, code: string, request_id: string }. "
            "HTTP status codes are semantic (400 bad input, 401 unauth, 404 not found, 500 server). "
            "request_id is a UUID generated per request, attached to logs. "
            "Client-side HTMX swaps an #error-banner partial on non-2xx responses "
            "using hx-swap-oob."
        ),
    },
    {   # 24
        "role": "user",
        "content": "Logging strategy — what format and where do logs go?",
    },
    {   # 25
        "role": "assistant",
        "content": (
            "Structured JSON logs to stdout: { ts, level, request_id, method, path, "
            "status, duration_ms, msg }. "
            "Fly.io ships stdout to its built-in log aggregator, accessible via fly logs. "
            "No log library needed — a small log() helper that writes "
            "JSON.stringify({ ts: Date.now(), ...fields }) + newline to process.stdout."
        ),
    },
    {   # 26
        "role": "user",
        "content": "How should we handle database migrations at deploy time?",
    },
    {   # 27
        "role": "assistant",
        "content": (
            "A migrate.ts script that:\n"
            "1. Opens the DB.\n"
            "2. Creates a schema_migrations table if it doesn't exist.\n"
            "3. Reads all *.sql files from db/migrations/, sorted by filename.\n"
            "4. Skips any already recorded in schema_migrations.\n"
            "5. Runs new ones in a transaction, records them.\n"
            "Add `bun run migrate` as a release_command in fly.toml so it runs before "
            "the new process starts, with zero downtime for non-breaking changes."
        ),
    },
    {   # 28
        "role": "user",
        "content": "Rate limiting — how do we prevent abuse without Redis?",
    },
    {   # 29
        "role": "assistant",
        "content": (
            "In-process sliding window counter using a Map<string, number[]> keyed on IP "
            "or session ID. Each entry is an array of timestamps. On each request: "
            "filter out timestamps older than the window (e.g. 60s), check length, "
            "push current timestamp. A background setInterval every 5 minutes clears "
            "entries with no recent timestamps to prevent memory growth. "
            "Works fine for a single Fly instance — no shared state needed."
        ),
    },
    {   # 30
        "role": "user",
        "content": "Let's think about testing. What do we test and how?",
    },
    {   # 31
        "role": "assistant",
        "content": (
            "Three layers:\n"
            "1. Unit tests (bun:test) for pure functions — route matcher, rate limiter logic, "
            "migration runner, WebAuthn helpers.\n"
            "2. Integration tests that spin up Bun.serve() on an ephemeral port with an "
            "in-memory SQLite DB (pass ':memory:' to bun:sqlite). "
            "Test full request/response cycles.\n"
            "3. No E2E browser tests initially — HTMX fragments are server-rendered; "
            "the integration tests cover the contract."
        ),
    },
    {   # 32
        "role": "user",
        "content": "How should we structure the HTMX partials? I want to keep the server templates clean.",
    },
    {   # 33
        "role": "assistant",
        "content": (
            "Each partial is a TypeScript function that returns an HTML string. "
            "views/partials/product-card.ts exports productCard(p: Product): string. "
            "The full page views import and compose partials. "
            "No templating engine — just tagged template literals with a small html`` "
            "tag helper that auto-escapes interpolated values to prevent XSS. "
            "Keep partials small and single-purpose."
        ),
    },
    {   # 34
        "role": "user",
        "content": "What's the fly.toml config going to look like?",
    },
    {   # 35
        "role": "assistant",
        "content": (
            "[app]\n"
            "name = 'your-app'\n"
            "primary_region = 'lax'\n\n"
            "[build]\n"
            "dockerfile = 'Dockerfile'\n\n"
            "[[mounts]]\n"
            "source = 'sqlite_data'\n"
            "destination = '/data'\n\n"
            "[http_service]\n"
            "internal_port = 3000\n"
            "force_https = true\n\n"
            "[http_service.concurrency]\n"
            "type = 'requests'\n"
            "hard_limit = 200\n\n"
            "[[vm]]\n"
            "memory = '256mb'\n"
            "cpu_kind = 'shared'\n"
            "cpus = 1\n\n"
            "[deploy]\n"
            "release_command = 'bun run migrate'"
        ),
    },
    {   # 36
        "role": "user",
        "content": "Should we add caching for product listings? They'll be read-heavy.",
    },
    {   # 37
        "role": "assistant",
        "content": (
            "Yes — an in-process LRU cache (Map with a max-size eviction policy) "
            "keyed on user_id. Cache the rendered HTML fragment, not the data objects. "
            "TTL of 30 seconds. Invalidate on any product write for that user_id. "
            "Since we're single-instance, no cache coherence problem. "
            "A simple 200-entry LRU keeps the hot path allocation trivial."
        ),
    },
    {   # 38
        "role": "user",
        "content": "What about background jobs — email notifications, cleanup tasks?",
    },
    {   # 39
        "role": "assistant",
        "content": (
            "No job queue — keep it in-process. Use setInterval for periodic cleanup "
            "(expired sessions, rate limit maps). For email, call a transactional email "
            "API (Postmark or Resend) inline in the request handler but fire-and-forget "
            "with a try/catch so a failed send doesn't block the response. "
            "We don't need durable job queues at this scale."
        ),
    },
    {   # 40
        "role": "user",
        "content": "How do we handle the Dockerfile for Bun on Fly?",
    },
    {   # 41
        "role": "assistant",
        "content": (
            "FROM oven/bun:1 as base\n"
            "WORKDIR /app\n"
            "COPY package.json bun.lockb ./\n"
            "RUN bun install --frozen-lockfile\n"
            "COPY . .\n"
            "ENV NODE_ENV=production\n"
            "EXPOSE 3000\n"
            "CMD [\"bun\", \"run\", \"src/server.ts\"]"
        ),
    },
    {   # 42
        "role": "user",
        "content": "CI/CD — what's the pipeline look like?",
    },
    {   # 43
        "role": "assistant",
        "content": (
            "GitHub Actions:\n"
            "1. On push to main: bun install, bun test, type-check.\n"
            "2. On passing tests: fly deploy --remote-only (uses FLY_API_TOKEN secret).\n"
            "3. Fly runs the release_command (migrate) before swapping traffic.\n"
            "One workflow file, two jobs (test, deploy). "
            "No staging environment — we catch issues in tests and rely on SQLite's "
            "transactional safety for migrations."
        ),
    },
    {   # 44
        "role": "user",
        "content": "Last topic before we wrap up the design phase — what's our approach to input validation?",
    },
    {   # 45 — last scripted assistant reply; index 45 is still context
        "role": "assistant",
        "content": (
            "Validate at the boundary. Each route handler calls a validate() function "
            "that takes the raw body and a schema object (plain TS interface with "
            "type assertions — no Zod, no Yup). It returns { ok: true, data } or "
            "{ ok: false, errors: string[] }. On failure we return 400 immediately. "
            "No validation inside db/ or lib/ — trust that the route layer cleaned up."
        ),
    },

    # ── Recall questions (indices 46-60, user-only) ──────────────────────────

    {   # 46
        "role": "user",
        "content": (
            "Okay, let's do a consistency check before we write any code. "
            "What runtime and language did we decide on, and what were the reasons?"
        ),
    },
    {   # 47
        "role": "user",
        "content": (
            "Walk me through the database decision. "
            "What are we using, and what did we specifically rule out and why?"
        ),
    },
    {   # 48
        "role": "user",
        "content": (
            "A new developer joins the project and suggests replacing our frontend "
            "approach with Next.js. Based on what we agreed, how do you respond?"
        ),
    },
    {   # 49
        "role": "user",
        "content": (
            "Remind me of the deployment setup. "
            "What platform, how many instances, and what was the scaling policy?"
        ),
    },
    {   # 50
        "role": "user",
        "content": (
            "What's our auth approach? "
            "Be specific about what we chose and what we explicitly ruled out."
        ),
    },
    {   # 51
        "role": "user",
        "content": (
            "Someone on the team wants to add Prisma to speed up query writing. "
            "Is that consistent with our decisions? What do you tell them?"
        ),
    },
    {   # 52
        "role": "user",
        "content": (
            "We're about to write the README. "
            "List every foundational tech decision we locked in at the start — "
            "runtime, database, frontend, deployment, auth — with a brief reason for each."
        ),
    },
    {   # 53
        "role": "user",
        "content": (
            "A vendor pitches us their managed Redis service for rate limiting and caching. "
            "Given our architectural decisions, what's the right response?"
        ),
    },
    {   # 54
        "role": "user",
        "content": (
            "If we ever need to scale beyond a single instance, "
            "what parts of our current architecture would need to change and why?"
        ),
    },
    {   # 55
        "role": "user",
        "content": (
            "Someone suggests adding Google OAuth as a login option alongside passkeys "
            "to improve conversion. Is that on the table?"
        ),
    },
    {   # 56
        "role": "user",
        "content": (
            "Quick check: what file format do we use for database migrations, "
            "and what tool runs them?"
        ),
    },
    {   # 57
        "role": "user",
        "content": (
            "We're doing a code review. A PR comes in that uses fetch() to call "
            "an external analytics API on every page view. "
            "Is that consistent with our project decisions?"
        ),
    },
    {   # 58
        "role": "user",
        "content": (
            "Describe the tech stack to a new engineer in three sentences. "
            "Make sure you include the five core decisions we established at the start."
        ),
    },
    {   # 59
        "role": "user",
        "content": (
            "Final question: if we had to rewrite this product from scratch in six months, "
            "which of our five core decisions would you keep and which would you reconsider, "
            "and why?"
        ),
    },
]

# Indices of recall questions in CHAT_LOG
CONTEXT_SLICE = CHAT_LOG[:46]       # messages 0-45 are context
RECALL_SLICE  = CHAT_LOG[46:]       # messages 46-59 are recall questions

assert len(CHAT_LOG) == 60, f"Expected 60 messages, got {len(CHAT_LOG)}"
assert len(RECALL_SLICE) == 14, f"Expected 14 recall questions, got {len(RECALL_SLICE)}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. GROUND TRUTH
#    One authoritative answer per recall question, keyed by RECALL_SLICE index.
#    Derived strictly from the decisions recorded in CONTEXT_SLICE above.
#    Used by the judge to score each agent response 1-5.
# ──────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH: dict[int, str] = {
    0: (
        "Bun as the runtime with TypeScript end-to-end. Not Node.js and not Deno. "
        "Reasons: Bun's built-in bundler, fast cold starts, and native TypeScript "
        "support without complex tsconfig setup."
    ),
    1: (
        "SQLite for the database with raw SQL only via bun:sqlite. No ORMs and no "
        "query builders — Drizzle, Prisma, and Kysley were all explicitly ruled out. "
        "All queries must be hand-written and parameterized."
    ),
    2: (
        "Decline the suggestion. Our frontend decision is HTMX with server-rendered "
        "HTML and no JavaScript framework of any kind. React, Vue, and Svelte were all "
        "explicitly ruled out. Next.js is a React framework and contradicts our decision. "
        "Only vanilla JS is permitted for things HTMX cannot handle."
    ),
    3: (
        "Fly.io, single machine, one instance. No autoscaling, no multi-region, no "
        "load balancer. The SQLite database lives on a Fly persistent volume. We stay "
        "single-instance until load actually demands otherwise. scale.min=1, scale.max=1."
    ),
    4: (
        "WebAuthn passkeys only. Passwords, OAuth providers, magic links, and TOTP "
        "were all explicitly ruled out. Every user registers a passkey and uses it to "
        "log in. Limiting older browser support is a deliberate accepted trade-off."
    ),
    5: (
        "No, Prisma is not consistent. We explicitly decided raw SQL only with "
        "bun:sqlite and ruled out Drizzle, Prisma, Kysley, and all query builders. "
        "The response should be: write hand-written parameterized SQL queries."
    ),
    6: (
        "1) Runtime: Bun + TypeScript (fast cold starts, built-in bundler, native TS). "
        "2) Database: SQLite raw SQL via bun:sqlite, no ORM (full control, zero "
        "abstraction overhead). "
        "3) Frontend: HTMX + server-rendered HTML, no JS framework (hypermedia-driven). "
        "4) Deployment: Fly.io single machine, no autoscaling (simplicity first). "
        "5) Auth: WebAuthn passkeys only, no passwords or OAuth (modern, passwordless)."
    ),
    7: (
        "Decline. Rate limiting is in-process via a sliding window counter (Map of "
        "timestamps per IP/session), and caching is an in-process LRU keyed on user_id. "
        "Being single-instance means no shared state is needed. Redis would add "
        "unnecessary external complexity. Our architecture deliberately keeps these "
        "concerns in-process."
    ),
    8: (
        "SQLite would need to be replaced with a distributed database such as PostgreSQL "
        "because SQLite does not support concurrent multi-writer access across machines. "
        "The in-process rate limiter and LRU cache would need external shared storage "
        "because they rely on single-instance memory. The Fly persistent volume for the "
        "SQLite file is also tied to one machine."
    ),
    9: (
        "No. Auth is passkeys only via WebAuthn. OAuth providers including Google OAuth "
        "were explicitly ruled out, along with passwords, magic links, and TOTP. Adding "
        "Google OAuth would directly contradict a core foundational decision."
    ),
    10: (
        "Plain SQL files numbered sequentially, e.g. 001_init.sql, stored in "
        "db/migrations/. A custom migrate.ts script runs them. It is invoked as a "
        "release_command in fly.toml via 'bun run migrate', so it runs automatically "
        "before each new deployment."
    ),
    11: (
        "Not explicitly prohibited by the five core decisions — there is no blanket "
        "ban on external API calls in this project. The concern is performance: calling "
        "an external service synchronously on every page view adds latency. A better "
        "pattern would be fire-and-forget or batched async, consistent with how we "
        "handle email notifications."
    ),
    12: (
        "Our stack in three sentences: We use Bun with TypeScript end-to-end and "
        "SQLite with raw hand-written SQL via bun:sqlite — no ORM. The frontend is "
        "HTMX with server-rendered HTML partials and no JavaScript framework, deployed "
        "on a single Fly.io machine with a persistent volume for the database. "
        "Authentication is WebAuthn passkeys only — no passwords, no OAuth."
    ),
    13: (
        "Keep: Bun/TypeScript (mature, fast, good DX), SQLite raw SQL (reliable and "
        "perfectly sufficient for single-instance scale), WebAuthn passkeys (future-proof "
        "auth worth the initial complexity). "
        "Possibly reconsider: HTMX if team growth brings engineers who are more "
        "productive with React-based tooling; Fly.io single instance if the product "
        "needs multi-region or high-availability before natural scale is reached."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. AGENTS
# ──────────────────────────────────────────────────────────────────────────────

# ── Agent A: vanilla (full conversation history as proper message objects) ────

def ask_vanilla(question: str) -> str:
    """
    Pass the complete context history as individual message objects, then
    append the recall question as the final user turn.  This is exactly how
    a real production LLM application works: the entire prior conversation
    is concatenated into the messages array on every API call.
    No system prompt, no summarisation — just raw history.
    """
    messages = list(CONTEXT_SLICE) + [{"role": "user", "content": question}]
    response = ollama.chat(model="gemma3:4b", messages=messages)
    return response.message.content.strip()


# ── Agent B: Active Memory (batch pre-process, then answer via ActiveAgent) ───

def build_active_memory_pipeline() -> tuple[ActiveAgent, Observer, Librarian]:
    """Wire up the full pipeline with an isolated Chroma store."""
    observer  = Observer(model="gemma3:4b", max_words=200)
    curator   = Curator(model="gemma3:4b")
    librarian = Librarian(path="./chroma_store_v2")
    reporter  = Reporter(librarian)
    conductor = Conductor(observer, curator, librarian, reporter)
    agent     = ActiveAgent(conductor, model="gemma3:4b")
    return agent, observer, librarian


def batch_process_context(agent: ActiveAgent, messages: list[dict]) -> None:
    """
    Drive all context messages through the pipeline without using the agent's
    response for the scripted assistant turns — we just want to build the
    memory store from the conversation history.
    """
    print("  Batch-processing context messages through the Active Memory pipeline...")
    for i, msg in enumerate(messages):
        # For user messages, call agent.chat() so the full pipeline runs
        # (Observer → Curator → Librarian) and we get a real response.
        # For scripted assistant messages, feed them directly through the
        # Conductor so the Observer updates its summary without generating
        # a new LLM response.
        if msg["role"] == "user":
            _ = agent.chat(msg["content"])   # response discarded; side-effects kept
        else:
            # Feed the scripted assistant text as an assistant turn to keep
            # the Observer's summary accurate without a second LLM call.
            agent._conductor.process_message(msg["content"], "assistant")

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(messages)} context messages")

    print(f"  Done. {len(messages)} messages processed.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 3. JUDGE
# ──────────────────────────────────────────────────────────────────────────────

def judge_score(ground_truth: str, response: str) -> int:
    """
    Ask a third Ollama call to score `response` against `ground_truth` on a
    1-5 scale.  Returns an int; falls back to 3 if the model's output cannot
    be parsed.
    """
    prompt = (
        f"Given this correct answer: {ground_truth}\n\n"
        f"Score this response from 1 to 5 for consistency and accuracy where "
        f"5 is perfectly consistent and 1 is completely wrong or contradictory. "
        f"Return only a number.\n\n"
        f"Response to score: {response}"
    )
    raw = ollama.chat(
        model="gemma3:4b",
        messages=[{"role": "user", "content": prompt}],
    ).message.content.strip()

    # Accept "4", "4/5", "Score: 4", "I'd give it a 4", etc.
    match = re.search(r"\b([1-5])\b", raw)
    return int(match.group(1)) if match else 3


# ──────────────────────────────────────────────────────────────────────────────
# 4. DISPLAY
# ──────────────────────────────────────────────────────────────────────────────

COL_WIDTH = 60
SEP = " │ "
FULL_WIDTH = COL_WIDTH * 2 + len(SEP)


def wrap_col(text: str, width: int = COL_WIDTH - 2) -> list[str]:
    if not text:
        return ["(no response)"]
    lines: list[str] = []
    for para in text.splitlines():
        if para.strip():
            lines.extend(textwrap.wrap(para, width) or [para])
        else:
            lines.append("")
    return lines or ["(no response)"]


def print_comparison(
    q_index: int,
    question: str,
    answer_a: str,
    answer_b: str,
    score_a: int,
    score_b: int,
    lat_a: float,
    lat_b: float,
) -> None:
    print(f"\n{'═' * FULL_WIDTH}")
    print(f"  RECALL Q{q_index:02d}  (message {45 + q_index} of 60)")
    print(f"{'═' * FULL_WIDTH}")

    # Question wrapped to full width
    for line in textwrap.wrap(f"USER: {question}", FULL_WIDTH - 2):
        print(f"  {line}")
    print(f"{'─' * FULL_WIDTH}")

    # Column headers
    label_a = "AGENT A  (Vanilla — Full History)".center(COL_WIDTH)
    label_b = "AGENT B  (Active Memory)".center(COL_WIDTH)
    print(f"{label_a}{SEP}{label_b}")
    print(f"{'─' * COL_WIDTH}{SEP}{'─' * COL_WIDTH}")

    # Side-by-side responses
    rows_a = wrap_col(answer_a)
    rows_b = wrap_col(answer_b)
    height = max(len(rows_a), len(rows_b))
    rows_a += [""] * (height - len(rows_a))
    rows_b += [""] * (height - len(rows_b))

    for ra, rb in zip(rows_a, rows_b):
        print(f"{ra:<{COL_WIDTH}}{SEP}{rb}")

    # Score + latency footer
    score_bar  = _score_bar(score_a)
    score_bar2 = _score_bar(score_b)
    print(f"{'─' * FULL_WIDTH}")
    col_a = f"  Score: {score_a}/5  {score_bar}   Latency: {lat_a:5.1f}s"
    col_b = f"  Score: {score_b}/5  {score_bar2}   Latency: {lat_b:5.1f}s"
    print(f"{col_a:<{COL_WIDTH}}{SEP}{col_b}")


def _score_bar(score: int, width: int = 5) -> str:
    """Tiny filled-block bar: ████░ for score 4/5."""
    filled = "█" * score
    empty  = "░" * (width - score)
    return filled + empty


def print_metrics_summary(
    recall_questions: list[dict],
    scores_a: list[int],
    scores_b: list[int],
    lats_a:   list[float],
    lats_b:   list[float],
) -> None:
    """Print the final aggregated results table."""

    avg_score_a = sum(scores_a) / len(scores_a)
    avg_score_b = sum(scores_b) / len(scores_b)
    avg_lat_a   = sum(lats_a)   / len(lats_a)
    avg_lat_b   = sum(lats_b)   / len(lats_b)
    peak_lat_a  = max(lats_a)
    peak_lat_b  = max(lats_b)

    W = FULL_WIDTH
    print(f"\n\n{'#' * W}")
    print("  METRICS SUMMARY")
    print(f"{'#' * W}\n")

    # ── Overall comparison ────────────────────────────────────────────────────
    pad = 38
    print(f"  {'':>{pad}}  {'Agent A (Vanilla)':>20}  {'Agent B (Active Memory)':>24}")
    print(f"  {'─' * pad}  {'─' * 20}  {'─' * 24}")
    print(f"  {'Average drift score (1–5)':>{pad}}  {avg_score_a:>17.2f}/5  {avg_score_b:>21.2f}/5")
    print(f"  {'Average latency':>{pad}}  {avg_lat_a:>18.1f}s  {avg_lat_b:>22.1f}s")
    print(f"  {'Peak latency':>{pad}}  {peak_lat_a:>18.1f}s  {peak_lat_b:>22.1f}s")

    # ── Per-question breakdown ────────────────────────────────────────────────
    print(f"\n  {'─' * W}")
    print(f"  PER-QUESTION DRIFT SCORES")
    print(f"  {'─' * W}")

    q_col  = 4     # "Q01"
    gt_col = 46    # truncated ground truth
    sc_col = 9     # "Score A"
    lt_col = 9     # "Lat A"

    hdr = (
        f"  {'Q':<{q_col}}  "
        f"{'Ground Truth (excerpt)':<{gt_col}}  "
        f"{'Score A':>{sc_col}}  "
        f"{'Score B':>{sc_col}}  "
        f"{'Lat A':>{lt_col}}  "
        f"{'Lat B':>{lt_col}}"
    )
    print(hdr)
    print(f"  {'─' * (q_col + gt_col + sc_col * 2 + lt_col * 2 + 12)}")

    for i, (msg, sa, sb, la, lb) in enumerate(
        zip(recall_questions, scores_a, scores_b, lats_a, lats_b), start=1
    ):
        gt_text = GROUND_TRUTH.get(i - 1, "")
        gt_excerpt = (gt_text[:gt_col - 3] + "...") if len(gt_text) > gt_col else gt_text
        bar_a = _score_bar(sa)
        bar_b = _score_bar(sb)
        print(
            f"  Q{i:02d}  "
            f"{gt_excerpt:<{gt_col}}  "
            f"{sa}/5 {bar_a}  "
            f"{sb}/5 {bar_b}  "
            f"{la:>{lt_col - 1}.1f}s  "
            f"{lb:>{lt_col - 1}.1f}s"
        )

    # Averages row
    print(f"  {'─' * (q_col + gt_col + sc_col * 2 + lt_col * 2 + 12)}")
    print(
        f"  {'AVG':<{q_col + 2}}"
        f"{'':>{gt_col + 2}}"
        f"{avg_score_a:>{sc_col - 2}.2f}/5   "
        f"{avg_score_b:>{sc_col - 2}.2f}/5   "
        f"{avg_lat_a:>{lt_col - 1}.1f}s  "
        f"{avg_lat_b:>{lt_col - 1}.1f}s"
    )
    print(
        f"  {'PEAK':<{q_col + 2}}"
        f"{'':>{gt_col + 2}}"
        f"{'':>{sc_col + 2}}"
        f"{'':>{sc_col + 2}}"
        f"{peak_lat_a:>{lt_col - 1}.1f}s  "
        f"{peak_lat_b:>{lt_col - 1}.1f}s"
    )
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * FULL_WIDTH)
    print("  benchmark_v2.py — Active Memory vs Vanilla (Full History)")
    print("=" * FULL_WIDTH)
    print(f"\n  Chat log        : {len(CHAT_LOG)} messages total")
    print(f"  Context block   : messages 1–{len(CONTEXT_SLICE)} ({len(CONTEXT_SLICE)} messages)")
    print(f"  Recall questions: {len(RECALL_SLICE)} (messages {len(CONTEXT_SLICE)+1}–60)")
    print(f"  Ground truth    : {len(GROUND_TRUTH)} entries\n")

    # ── PHASE 1: Build Agent B memory store ───────────────────────────────────
    print("─" * FULL_WIDTH)
    print("  PHASE 1 — Building Active Memory store from context messages")
    print("─" * FULL_WIDTH)
    active_agent, observer, librarian = build_active_memory_pipeline()
    batch_process_context(active_agent, CONTEXT_SLICE)

    warm_count = librarian._collections["warm"].count()
    cold_count = librarian._collections["cold"].count()
    print(f"  Memory store: {warm_count} warm  |  {cold_count} cold  |  {len(observer.trimmings)} observer trimmings\n")

    # ── PHASE 2: Run recall questions, record answers + latencies ─────────────
    print("─" * FULL_WIDTH)
    print("  PHASE 2 — Running recall questions through both agents")
    print("─" * FULL_WIDTH)

    vanilla_answers: list[str]   = []
    active_answers:  list[str]   = []
    lats_a:          list[float] = []
    lats_b:          list[float] = []

    for i, msg in enumerate(RECALL_SLICE):
        question = msg["content"]
        print(f"  Recall Q{i + 1:02d} ... ", end="", flush=True)

        t0 = time.time()
        answer_a = ask_vanilla(question)
        lats_a.append(time.time() - t0)

        t0 = time.time()
        answer_b = active_agent.chat(question)
        lats_b.append(time.time() - t0)

        vanilla_answers.append(answer_a)
        active_answers.append(answer_b)
        print(f"done  (A: {lats_a[-1]:.1f}s  B: {lats_b[-1]:.1f}s)")

    # ── PHASE 3: Judge each response against ground truth ─────────────────────
    print(f"\n{'─' * FULL_WIDTH}")
    print("  PHASE 3 — Scoring responses against ground truth")
    print("─" * FULL_WIDTH)

    scores_a: list[int] = []
    scores_b: list[int] = []

    for i, (ans_a, ans_b) in enumerate(zip(vanilla_answers, active_answers)):
        gt = GROUND_TRUTH.get(i, "")
        print(f"  Judging Q{i + 1:02d} ... ", end="", flush=True)

        score_a = judge_score(gt, ans_a)
        score_b = judge_score(gt, ans_b)

        scores_a.append(score_a)
        scores_b.append(score_b)
        print(f"A={score_a}/5  B={score_b}/5")

    # ── PHASE 4: Side-by-side comparison ──────────────────────────────────────
    print(f"\n\n{'#' * FULL_WIDTH}")
    print("  RESULTS — Recall Question Comparison")
    print(f"{'#' * FULL_WIDTH}")

    for i, (msg, a_ans, b_ans, sa, sb, la, lb) in enumerate(
        zip(RECALL_SLICE, vanilla_answers, active_answers,
            scores_a, scores_b, lats_a, lats_b),
        start=1,
    ):
        print_comparison(i, msg["content"], a_ans, b_ans, sa, sb, la, lb)

    # ── PHASE 5: Retrieval log ─────────────────────────────────────────────────
    print(f"\n\n{'#' * FULL_WIDTH}")
    print("  ACTIVE MEMORY — Retrieval Log (what got surfaced and how often)")
    print(f"{'#' * FULL_WIDTH}\n")

    retrieval_log = librarian.get_retrieval_logs()
    if retrieval_log:
        def _fetch_by_id(mid: str) -> str:
            for tier in ("warm", "cold"):
                result = librarian._collections[tier].get(ids=[mid])
                docs = result.get("documents") or []
                if docs and docs[0]:
                    text = docs[0]
                    return (text[:100] + "...") if len(text) > 100 else text
            return "(not found)"

        for memory_id, count in sorted(retrieval_log.items(), key=lambda x: -x[1]):
            preview = _fetch_by_id(memory_id)
            print(f"  [{count:2d}x]  {memory_id[:8]}…  {preview}")
    else:
        print("  (no retrievals logged — recall trigger may not have fired)")

    print(f"\n  Total warm memories: {librarian._collections['warm'].count()}")
    print(f"  Total cold memories: {librarian._collections['cold'].count()}")
    print(f"  Observer trimmings generated: {len(observer.trimmings)}")

    # ── PHASE 6: Metrics summary ───────────────────────────────────────────────
    print_metrics_summary(RECALL_SLICE, scores_a, scores_b, lats_a, lats_b)


if __name__ == "__main__":
    main()
