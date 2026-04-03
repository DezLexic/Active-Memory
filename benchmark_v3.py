# -*- coding: utf-8 -*-
import os
os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

"""
benchmark_v3.py

Clean-room benchmark comparing two context-management strategies:

  Agent A — Vanilla
    For each recall question the entire 80-message chat log is formatted into a
    single system-prompt block and sent to the model verbatim alongside the
    question.  No compression, no retrieval — raw context dump.

  Agent B — Active Memory
    All 80 messages are batch-ingested through Observer → Curator → Librarian
    with no conversational responses generated.  This builds the memory store
    from the full conversation.  Each recall question is then answered by the
    Active Agent, which assembles a compressed context window from the live
    Observer summary plus any memories the Conductor retrieves.

Six decisions are locked in during messages 1-15.  Messages 16-80 are normal
development discussion that progressively dilutes the early context.

After both agents answer every question a judge Ollama call scores each
response 1-5 against a ground-truth dictionary.  A final results table
shows per-question scores and latencies plus overall averages.
"""

import re
import time
import textwrap
import ollama

from observer     import Observer
from curator      import Curator
from librarian    import Librarian
from reporter     import Reporter
from conductor    import Conductor
from active_agent import ActiveAgent


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CHAT LOG  (80 messages — pure conversation, no recall questions)
#
#     Indices  0-14  — six decisions established, with explicit reasoning
#     Indices 15-79  — development discussion that dilutes the early context
# ──────────────────────────────────────────────────────────────────────────────

CHAT_LOG: list[dict] = [

    # ── Decision block ────────────────────────────────────────────────────────

    {   # 0
        "role": "user",
        "content": (
            "Before a single line of code gets written I want the tech stack locked. "
            "Decision one: Elixir with the Phoenix framework. The BEAM VM's concurrency "
            "model is unmatched for a product that needs to stay responsive under load, "
            "supervision trees give us fault tolerance without extra infrastructure, and "
            "Phoenix LiveView lets us build real-time UI features without shipping a "
            "separate JavaScript SPA. The entire stack will be Elixir top to bottom. "
            "No React, no Vue, no Next.js. LiveView handles all interactive UI."
        ),
    },
    {   # 1
        "role": "assistant",
        "content": (
            "Elixir + Phoenix is a defensible long-term choice. BEAM process isolation "
            "means a crash in one context doesn't cascade, hot code reloading simplifies "
            "deploys, and LiveView eliminates the client-server state synchronisation "
            "problem entirely. Ruling out SPA frameworks removes a whole class of "
            "complexity. Decision one noted: Elixir/Phoenix/LiveView, no SPA. "
            "What's the database?"
        ),
    },
    {   # 2
        "role": "user",
        "content": (
            "CockroachDB for the database. It's distributed by default, "
            "PostgreSQL-wire-compatible so all our SQL tooling works, and it gives us "
            "geo-distribution capability when we eventually need it. "
            "And this is non-negotiable: raw SQL only. No Ecto, no Ecto.Repo, "
            "no query builders of any kind. Every database interaction goes through "
            "hand-written, parameterised SQL. I've watched ORMs paper over performance "
            "problems until they become catastrophic. Not on this project."
        ),
    },
    {   # 3
        "role": "assistant",
        "content": (
            "CockroachDB is PostgreSQL-compatible at the wire level so our migration "
            "tooling, prepared statements, and explain plans all work as expected. "
            "Raw SQL only is a discipline worth enforcing — queries stay readable, "
            "execution plans are predictable, and there's no ORM magic to debug at 3am. "
            "Ecto is off the table even for simple queries. Decision two: CockroachDB. "
            "Decision three: raw SQL only, no Ecto. Auth approach?"
        ),
    },
    {   # 4
        "role": "user",
        "content": (
            "Magic links only for authentication. Users enter their email address, "
            "receive a single-use time-limited link, click it, and they're in. "
            "No passwords — I don't want the credential storage liability or the "
            "password-reset flow maintenance. No OAuth — I don't want our auth path "
            "dependent on Google or GitHub uptime. Just our own magic link "
            "implementation with tokens stored in Redis with a short TTL."
        ),
    },
    {   # 5
        "role": "assistant",
        "content": (
            "Magic links are underrated. No breach exposure from stored credentials, "
            "no password reset flows to maintain, and the UX smooths out quickly once "
            "users are familiar with it. Single-use tokens in Redis with a short TTL "
            "and single-consumption invalidation is the right implementation pattern. "
            "Decision four: magic links only, no passwords, no OAuth. "
            "Where does this thing run?"
        ),
    },
    {   # 6
        "role": "user",
        "content": (
            "Kubernetes on GKE. Hard minimum of three replicas — I want high availability "
            "as a default, not a future upgrade. The team has deep Kubernetes experience "
            "already, and GKE's managed control plane means we don't own etcd backups "
            "or API server upgrades. No serverless, no Cloud Run, no container-as-a-"
            "service shortcuts. Proper Kubernetes with Deployments, Services, and "
            "an Ingress."
        ),
    },
    {   # 7
        "role": "assistant",
        "content": (
            "GKE with a mandatory three-replica floor is the right call for a product "
            "where downtime costs money. Managed control plane keeps operational "
            "overhead low while giving full Kubernetes semantics — rolling updates, "
            "pod disruption budgets, horizontal pod autoscaling when we're ready. "
            "Cloud Run and serverless are ruled out. "
            "Decision five: GKE Kubernetes, minimum three replicas. Caching strategy?"
        ),
    },
    {   # 8
        "role": "user",
        "content": (
            "Redis Cluster for all caching. And this is explicit: no in-process caching "
            "of any kind. The moment you have more than one replica, in-process caches "
            "diverge and you spend your life debugging stale-read bugs that only "
            "reproduce under load. Redis Cluster gives every replica a single consistent "
            "cache layer. It also handles magic link token storage and can back our "
            "session data. If it needs to be cached or shared across pods, it goes "
            "in Redis."
        ),
    },
    {   # 9
        "role": "assistant",
        "content": (
            "Redis Cluster is the correct answer for a multi-replica deployment. "
            "In-process caching in a K8s pod is a footgun at scale — you can't "
            "invalidate a key you can't reach. A shared Redis layer also enables "
            "pub/sub for LiveView channel broadcasts across nodes, which we'll need. "
            "Decision six: Redis Cluster, no in-process caching. "
            "Six decisions locked: Elixir/Phoenix/LiveView, CockroachDB, raw SQL only, "
            "magic links, GKE K8s three-replica minimum, Redis Cluster."
        ),
    },
    {   # 10
        "role": "user",
        "content": (
            "Correct. Let me state them cleanly: "
            "(1) Elixir + Phoenix with LiveView for all UI — no SPA frameworks. "
            "(2) CockroachDB as the database. "
            "(3) Raw SQL only, Ecto is banned from the codebase. "
            "(4) Magic links for auth — no passwords, no OAuth. "
            "(5) GKE Kubernetes with a hard three-replica minimum — no serverless. "
            "(6) Redis Cluster for all caching and shared state — no in-process cache. "
            "These are the foundations. Nothing overrides them without a full team "
            "architecture discussion."
        ),
    },
    {   # 11
        "role": "assistant",
        "content": (
            "Recorded. Six non-negotiable decisions: "
            "1. Elixir/Phoenix/LiveView — full-stack Elixir, no SPA. "
            "2. CockroachDB — distributed, PostgreSQL-compatible. "
            "3. Raw SQL only — Ecto is explicitly prohibited. "
            "4. Magic links — no passwords, no OAuth, tokens in Redis. "
            "5. GKE Kubernetes — three-replica hard floor, no serverless or Cloud Run. "
            "6. Redis Cluster — shared cache for all replicas, no in-process caching. "
            "Nothing changes without a full team architecture review. Ready to build?"
        ),
    },
    {   # 12
        "role": "user",
        "content": (
            "One operational constraint to add: all configuration via environment "
            "variables, twelve-factor style. No config files checked into the repo. "
            "Database URLs, Redis connection strings, secret keys — all via env vars, "
            "injected through Kubernetes Secrets and ConfigMaps."
        ),
    },
    {   # 13
        "role": "assistant",
        "content": (
            "Twelve-factor config noted. All runtime config from the environment — "
            "nothing sensitive in the repo. Kubernetes Secrets for credentials, "
            "ConfigMaps for non-sensitive config values. Phoenix's runtime.exs reads "
            "from System.fetch_env! so a missing variable fails loudly at startup "
            "rather than silently at runtime."
        ),
    },
    {   # 14
        "role": "user",
        "content": (
            "Good. Stack is locked. Let's start designing the system."
        ),
    },

    # ── Development discussion (indices 15-69) ────────────────────────────────

    {   # 15
        "role": "user",
        "content": "Let's start with the database schema. What tables do we need for the MVP?",
    },
    {   # 16
        "role": "assistant",
        "content": (
            "Core MVP tables:\n"
            "  users(id UUID PK, email TEXT UNIQUE NOT NULL, created_at TIMESTAMPTZ)\n"
            "  magic_link_tokens(id UUID PK, user_id UUID FK, token TEXT UNIQUE, "
            "expires_at TIMESTAMPTZ, used_at TIMESTAMPTZ)\n"
            "  sessions(id UUID PK, user_id UUID FK, created_at TIMESTAMPTZ, "
            "expires_at TIMESTAMPTZ)\n"
            "All IDs as UUIDs — CockroachDB generates them efficiently and they're "
            "safe to expose in URLs."
        ),
    },
    {   # 17
        "role": "user",
        "content": "What should the products table look like? Users can list items for sale.",
    },
    {   # 18
        "role": "assistant",
        "content": (
            "products(id UUID PK, user_id UUID FK REFERENCES users(id) ON DELETE CASCADE, "
            "title TEXT NOT NULL, description TEXT, price_cents INTEGER NOT NULL, "
            "currency CHAR(3) DEFAULT 'USD', status TEXT DEFAULT 'active', "
            "created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ). "
            "Store prices as integer cents — avoid floating-point money bugs entirely. "
            "Status as TEXT with a CHECK constraint: CHECK (status IN ('active','paused','sold'))."
        ),
    },
    {   # 19
        "role": "user",
        "content": "How do we structure the Phoenix router? What does the top level look like?",
    },
    {   # 20
        "role": "assistant",
        "content": (
            "Three pipeline scopes: :browser (session, flash, CSRF), :api (JSON, no session), "
            "and :live (LiveView socket). "
            "Main routes:\n"
            "  GET  /auth/request      — magic link request form (LiveView)\n"
            "  GET  /auth/confirm/:token — token exchange, sets session\n"
            "  GET  /auth/logout\n"
            "  live /dashboard          — authenticated main view\n"
            "  live /products/new\n"
            "  live /products/:id/edit\n"
            "  scope /api/v1            — REST endpoints for mobile clients"
        ),
    },
    {   # 21
        "role": "user",
        "content": "Walk me through the LiveView module structure for the dashboard.",
    },
    {   # 22
        "role": "assistant",
        "content": (
            "DashboardLive implements mount/2, handle_params/3, and handle_event/3. "
            "mount/2 checks assigns.current_user (set by on_mount auth hook), loads "
            "initial data via a raw SQL query, assigns to socket. "
            "Use LiveView streams for product lists to avoid re-rendering the full list "
            "on each update. Components live in DashboardLive.Components — small, "
            "stateless functional components for cards, modals, form inputs."
        ),
    },
    {   # 23
        "role": "user",
        "content": "How does the magic link flow work end to end?",
    },
    {   # 24
        "role": "assistant",
        "content": (
            "Full flow:\n"
            "1. User submits email on RequestLive.\n"
            "2. Handler upserts user row (INSERT ... ON CONFLICT DO NOTHING), "
            "generates a 32-byte secure random token, stores it in Redis with a "
            "15-minute TTL, emails the link.\n"
            "3. User clicks the link → GET /auth/confirm/:token.\n"
            "4. Controller fetches token from Redis, validates not expired, marks used "
            "(DELETE from Redis), creates session row in CockroachDB, sets session cookie.\n"
            "5. Redirect to /dashboard."
        ),
    },
    {   # 25
        "role": "user",
        "content": "How do we handle database migrations with CockroachDB?",
    },
    {   # 26
        "role": "assistant",
        "content": (
            "Use golang-migrate or a custom Elixir migration runner — plain .sql files "
            "numbered sequentially in priv/migrations/. A schema_migrations table tracks "
            "applied migrations. Run migrations as part of the K8s init container on "
            "deploy so they complete before new pods receive traffic. CockroachDB "
            "supports online schema changes so most migrations run without locking; "
            "check the docs for the handful of exceptions (column type changes, etc)."
        ),
    },
    {   # 27
        "role": "user",
        "content": "What Elixir library do we use to talk to Redis, and how do we set up connection pooling?",
    },
    {   # 28
        "role": "assistant",
        "content": (
            "Redix for the Redis client — it's a pure Elixir implementation with solid "
            "Cluster support via Redix.Cluster. Wrap it in a supervised pool: start "
            "a pool of Redix connections under the application supervisor, expose a "
            "thin Redis module with get/2, set/3, del/2 helpers that check out a "
            "connection from the pool. Keep pool size proportional to expected "
            "concurrent Redis operations per pod — start with 10 connections."
        ),
    },
    {   # 29
        "role": "user",
        "content": "What does the K8s Deployment manifest look like for the Phoenix app?",
    },
    {   # 30
        "role": "assistant",
        "content": (
            "Key fields:\n"
            "  replicas: 3  (hard minimum)\n"
            "  strategy: RollingUpdate — maxUnavailable: 0, maxSurge: 1\n"
            "  containers[0].image: gcr.io/project/app:SHA\n"
            "  envFrom: [{secretRef: app-secrets}, {configMapRef: app-config}]\n"
            "  livenessProbe: GET /health, initialDelaySeconds: 10\n"
            "  readinessProbe: GET /ready, failureThreshold: 3\n"
            "  resources: requests cpu 100m mem 256Mi, limits cpu 500m mem 512Mi\n"
            "Add a PodDisruptionBudget: minAvailable 2 so rolling deploys never "
            "drop below two healthy pods."
        ),
    },
    {   # 31
        "role": "user",
        "content": "How do we manage environment config in K8s — what goes in Secrets vs ConfigMaps?",
    },
    {   # 32
        "role": "assistant",
        "content": (
            "Secrets (base64-encoded, should be managed by External Secrets Operator "
            "pulling from GCP Secret Manager): DATABASE_URL, REDIS_URL, SECRET_KEY_BASE, "
            "SMTP credentials, any API keys.\n"
            "ConfigMaps (non-sensitive): PHX_HOST, PORT, POOL_SIZE, LOG_LEVEL, "
            "REGION, FEATURE_FLAG_REDIS_KEY_PREFIX.\n"
            "Both injected via envFrom in the Deployment spec. "
            "Phoenix runtime.exs reads everything with System.fetch_env! — "
            "missing vars crash startup, which is correct behaviour."
        ),
    },
    {   # 33
        "role": "user",
        "content": "Testing strategy — what layers do we test and with what tools?",
    },
    {   # 34
        "role": "assistant",
        "content": (
            "Three layers:\n"
            "1. Unit (ExUnit): pure functions, context modules, helper modules. "
            "Fast, no DB or Redis.\n"
            "2. Integration (ExUnit + Ecto sandbox): full request cycles via "
            "Phoenix.ConnTest and LiveView test helpers. Use a test CockroachDB "
            "instance with a fresh schema per test run. Sandbox mode gives each "
            "test its own transaction that rolls back after.\n"
            "3. Contract tests for the REST API using ExUnit + a mock Redis. "
            "No E2E browser tests initially."
        ),
    },
    {   # 35
        "role": "user",
        "content": "CI pipeline — what does the GitHub Actions workflow look like?",
    },
    {   # 36
        "role": "assistant",
        "content": (
            "Two jobs:\n"
            "test: checkout → setup-beam → mix deps.get → mix compile --warnings-as-errors "
            "→ mix credo → mix test --cover.\n"
            "deploy (needs: test, on: push to main): "
            "docker build → push to GCR with SHA tag → "
            "kubectl set image deployment/app app=gcr.io/project/app:SHA → "
            "kubectl rollout status deployment/app.\n"
            "Secrets: GOOGLE_CREDENTIALS, GKE_CLUSTER, GKE_ZONE stored as "
            "GitHub Actions secrets."
        ),
    },
    {   # 37
        "role": "user",
        "content": "How should we handle errors in LiveView? What's the pattern for user-facing error states?",
    },
    {   # 38
        "role": "assistant",
        "content": (
            "Pattern: each LiveView assign has a companion :error assign. "
            "Operations that can fail return {:ok, result} | {:error, reason}. "
            "On :error, assign the reason and render an inline error component "
            "rather than crashing the process. LiveView processes are supervised — "
            "a crash triggers a reconnect with a flash message, but it's jarring. "
            "Reserve process crashes for truly unexpected states. "
            "For async operations use assign_async/3 which gives :loading/:ok/:error "
            "states built in."
        ),
    },
    {   # 39
        "role": "user",
        "content": "We'll need background jobs for sending emails and cleaning up expired tokens. What do we use?",
    },
    {   # 40
        "role": "assistant",
        "content": (
            "Oban for background jobs. It's backed by the database (CockroachDB in our "
            "case — Oban supports PostgreSQL-compatible databases) so jobs survive "
            "process crashes and restarts. Workers run in the same K8s pods — no "
            "separate worker fleet. Two queues to start: mailer (concurrency 5) and "
            "cleanup (concurrency 2). Cleanup worker runs on a cron schedule to "
            "delete expired magic link tokens and sessions from CockroachDB."
        ),
    },
    {   # 41
        "role": "user",
        "content": "What about file uploads — users will want to attach images to product listings.",
    },
    {   # 42
        "role": "assistant",
        "content": (
            "LiveView's live_upload handles the client side — chunk uploads directly "
            "from the browser. Server side: use ex_aws + SweetXml to upload to GCS "
            "(Google Cloud Storage) after validation. Store only the GCS object key "
            "in CockroachDB, not the full URL — construct URLs at render time from "
            "the key + bucket config. Validate MIME type and file size server-side "
            "before accepting. CDN (Cloud CDN) sits in front of GCS for fast delivery."
        ),
    },
    {   # 43
        "role": "user",
        "content": "How do we expose Prometheus metrics from the Phoenix app?",
    },
    {   # 44
        "role": "assistant",
        "content": (
            "Use the peep library which integrates with Elixir's Telemetry pipeline. "
            "Attach Telemetry handlers in application start for Phoenix endpoint events "
            "(http.request.stop), VM metrics (memory, schedulers), and custom business "
            "events (magic link sent, product created). Expose a /metrics endpoint "
            "scoped to an internal-only K8s Service so Prometheus can scrape it but "
            "it's not publicly reachable. Add a ServiceMonitor CRD for Prometheus "
            "Operator auto-discovery."
        ),
    },
    {   # 45
        "role": "user",
        "content": "Rate limiting — how do we prevent abuse of the magic link endpoint?",
    },
    {   # 46
        "role": "assistant",
        "content": (
            "Hammer library for rate limiting — it supports Redis as the backend, "
            "which means rate limit counts are shared across all three pods. "
            "Limit magic link requests to 5 per email per 10 minutes. "
            "On exceeding the limit return a 429 with a Retry-After header. "
            "Also rate limit by IP as a secondary defence: 20 requests per minute "
            "per IP across all unauthenticated endpoints. Store both counters in "
            "Redis so the limits are enforced cluster-wide, not per-pod."
        ),
    },
    {   # 47
        "role": "user",
        "content": "Should we build an umbrella app or keep it as a single Phoenix application?",
    },
    {   # 48
        "role": "assistant",
        "content": (
            "Single application for the MVP. Umbrella projects add compile-time "
            "boundary enforcement which is valuable, but the overhead of managing "
            "inter-app dependencies slows down early-stage development. Start with a "
            "well-organised single app — contexts (Accounts, Products, Marketplace) "
            "as bounded modules with clear public APIs. Refactor to umbrella only if "
            "you genuinely need separate deployment units or team boundary enforcement."
        ),
    },
    {   # 49
        "role": "user",
        "content": "How do we handle LiveView form validation? We can't use Ecto.Changeset.",
    },
    {   # 50
        "role": "assistant",
        "content": (
            "Actually Ecto.Changeset is fine here — the prohibition is on Ecto.Repo "
            "and ORM query building, not on Ecto's embedded schema and validation "
            "primitives. Ecto.Changeset with embedded_schema gives us cast/3, "
            "validate_required/3, validate_length/3 etc without touching the database. "
            "If that still feels like cheating, roll a small hand-written validator "
            "module that returns {:ok, data} | {:error, %{field => [errors]}} — "
            "it's 50 lines and gives full control."
        ),
    },
    {   # 51
        "role": "user",
        "content": "How do we handle session refresh — keeping users logged in for more than 15 minutes?",
    },
    {   # 52
        "role": "assistant",
        "content": (
            "Sessions are separate from magic link tokens. On successful magic link "
            "authentication, create a session row in CockroachDB with a 30-day expiry "
            "and set an HttpOnly Secure SameSite=Strict cookie containing the session ID. "
            "On each request, validate the session ID against CockroachDB — if valid "
            "and not expired, the user is authenticated. Sliding window: update "
            "expires_at to now+30d on each active request so the session stays alive "
            "while the user is active."
        ),
    },
    {   # 53
        "role": "user",
        "content": "What CockroachDB-specific SQL quirks do we need to watch out for?",
    },
    {   # 54
        "role": "assistant",
        "content": (
            "Key differences from standard PostgreSQL:\n"
            "1. No SERIAL — use UUID or gen_random_uuid() for primary keys.\n"
            "2. RETURNING clause works but RETURNING NOTHING is CockroachDB-specific.\n"
            "3. Distributed transactions have higher latency — keep transactions short.\n"
            "4. No support for some PostgreSQL extensions (PostGIS, pg_trgm vary by version).\n"
            "5. Schema changes are online but multi-step (add column, backfill, set default). "
            "Check the schema change best practices guide before any ALTER TABLE.\n"
            "6. EXPLAIN ANALYZE output differs from pg — use the CockroachDB UI for query plans."
        ),
    },
    {   # 55
        "role": "user",
        "content": "We'll need feature flags for gradual rollouts. What's the plan?",
    },
    {   # 56
        "role": "assistant",
        "content": (
            "FunWithFlags as the feature flag library — it supports Redis as a backend "
            "which keeps flags consistent across all pods. Flags stored in Redis with "
            "a local ETS cache (TTL 30s) to avoid hammering Redis on every request. "
            "Use actor-based flags keyed on user.id for gradual rollouts: enable for "
            "10% of users, then 50%, then 100%. Gate flags at the LiveView mount level "
            "so the flag check happens once per page load, not per render."
        ),
    },
    {   # 57
        "role": "user",
        "content": "How do we version the REST API for mobile clients?",
    },
    {   # 58
        "role": "assistant",
        "content": (
            "URL-based versioning: /api/v1/... for the initial version. "
            "Accept-Version header versioning is cleaner semantically but harder to "
            "test and cache. URL versioning is pragmatic and explicit. "
            "Keep v1 stable — breaking changes go to /api/v2 with a deprecation notice "
            "on v1. The Phoenix router isolates versions in separate scopes with their "
            "own plugs, so v1 and v2 share zero controller code unless deliberately "
            "extracted into shared context modules."
        ),
    },
    {   # 59
        "role": "user",
        "content": "Load testing — how do we validate the three-replica setup handles expected traffic?",
    },
    {   # 60
        "role": "assistant",
        "content": (
            "k6 for load testing — scriptable in JavaScript, integrates with Grafana "
            "for real-time dashboards. Test scenarios: sustained load at expected peak, "
            "burst to 5x normal, magic link flow under load, product listing pagination. "
            "Run against a staging K8s cluster (identical spec to prod, three replicas) "
            "not against local. Watch for: Redis connection pool exhaustion, "
            "CockroachDB connection limits, LiveView channel memory per connected socket. "
            "Set SLO targets before running: p99 < 500ms, error rate < 0.1%."
        ),
    },
    {   # 61
        "role": "user",
        "content": "Structured logging — what format and what fields should every log line have?",
    },
    {   # 62
        "role": "assistant",
        "content": (
            "JSON to stdout, shipped to Cloud Logging by the GKE node agent. "
            "Required fields on every line: timestamp (ISO8601), level, request_id "
            "(UUID per request, propagated through Logger metadata), "
            "user_id (if authenticated), pod_name (from HOSTNAME env var), "
            "message. HTTP request logs add: method, path, status, duration_ms. "
            "Use Logger.metadata/1 to set request_id at the plug level so it "
            "propagates automatically to all log calls within that request lifecycle."
        ),
    },
    {   # 63
        "role": "user",
        "content": "How do we handle the K8s Ingress and TLS?",
    },
    {   # 64
        "role": "assistant",
        "content": (
            "GKE-managed ingress with the Google-managed SSL certificate — no cert-manager "
            "needed, Google rotates certs automatically. Ingress spec: "
            "annotations: kubernetes.io/ingress.class: gce, "
            "networking.gke.io/managed-certificates: app-cert. "
            "Backend: Service of type NodePort pointing at the Phoenix pods on port 4000. "
            "Add a FrontendConfig for HTTP-to-HTTPS redirect. Health check path is /health "
            "— make sure the Phoenix endpoint returns 200 on that path even before the "
            "app is fully initialised."
        ),
    },
    {   # 65
        "role": "user",
        "content": "What's our approach to database connection pooling in the app?",
    },
    {   # 66
        "role": "assistant",
        "content": (
            "Since we're not using Ecto we manage the connection pool ourselves. "
            "Use postgrex directly (it's what Ecto uses under the hood) — start a "
            "supervised Postgrex connection pool via DBConnection. Pool size: start at "
            "10 per pod, tune based on CockroachDB's max_connections limit and observed "
            "concurrency. Expose a thin Repo module (not Ecto.Repo — just a module) "
            "that calls DBConnection.execute/4 with the pool ref. "
            "Parameterised queries only — never interpolate user input into SQL strings."
        ),
    },
    {   # 67
        "role": "user",
        "content": "How should we approach documentation — code comments, architectural docs, runbooks?",
    },
    {   # 68
        "role": "assistant",
        "content": (
            "Three layers: ExDoc for inline module and function docs (auto-published to "
            "an internal docs site on every main branch deploy), an Architecture "
            "Decision Records folder (docs/adr/) for decisions like the ones we made "
            "today — each decision gets its own numbered Markdown file with context, "
            "decision, and consequences, and a runbooks/ folder in the repo for "
            "operational procedures: deploy rollback, database failover, Redis "
            "connection issues. Runbooks are Markdown, versioned alongside the code."
        ),
    },
    {   # 69
        "role": "user",
        "content": (
            "Good. That covers the main design areas. "
            "Let's move on to the implementation sprint."
        ),
    },

    # ── Implementation sprint discussion (indices 70-79) ─────────────────────

    {   # 70
        "role": "assistant",
        "content": (
            "Let's kick off the sprint. First milestone: scaffold the Phoenix app, "
            "configure the Postgrex connection pool, write migration 001 with the "
            "core tables, and get /health returning 200. Everything else builds on "
            "that foundation."
        ),
    },
    {   # 71
        "role": "user",
        "content": (
            "We need to support English, French, and German at launch. "
            "How do we handle internationalisation?"
        ),
    },
    {   # 72
        "role": "assistant",
        "content": (
            "Gettext — it's built into Phoenix. Each locale gets a .po file under "
            "priv/gettext/. Wrap all user-facing strings in gettext/1 or dgettext/2. "
            "Phoenix selects locale from Accept-Language or a session preference. "
            "In LiveView pass locale as a socket assign and reference it in templates. "
            "Keep translation files in the repo — no external translation service "
            "needed at MVP stage."
        ),
    },
    {   # 73
        "role": "user",
        "content": "Users need to search product listings by keyword. What's the approach?",
    },
    {   # 74
        "role": "assistant",
        "content": (
            "PostgreSQL full-text search via raw SQL — CockroachDB supports it. "
            "Add a tsvector column to products, populate it via trigger using "
            "to_tsvector('english', title || ' ' || coalesce(description, '')). "
            "Query with to_tsquery and the @@ operator. All raw SQL, consistent "
            "with our no-ORM policy. No external search engine needed for MVP — "
            "CockroachDB FTS handles tens of thousands of products without issue."
        ),
    },
    {   # 75
        "role": "user",
        "content": "How do we paginate large product lists without degrading at scale?",
    },
    {   # 76
        "role": "assistant",
        "content": (
            "Cursor-based pagination — offsets degrade linearly with table size. "
            "Use (created_at, id) as the compound cursor: "
            "WHERE (created_at, id) < ($cursor_ts, $cursor_id) "
            "ORDER BY created_at DESC, id DESC LIMIT 25. "
            "Return the last row's values as next_cursor in the response payload. "
            "Pure raw SQL, performs identically at page 1 or page 50,000."
        ),
    },
    {   # 77
        "role": "user",
        "content": "How do we notify sellers when a product sells?",
    },
    {   # 78
        "role": "assistant",
        "content": (
            "Email notification via Oban. When a purchase is recorded, enqueue a "
            "SendSaleNotification job in the mailer queue. The worker sends via the "
            "configured SMTP provider. No push notifications at MVP — email is "
            "sufficient and avoids APNs/FCM device registration complexity. "
            "Add a notify_on_sale boolean to users so sellers can opt out. "
            "Oban retries handle transient SMTP failures automatically."
        ),
    },
    {   # 79
        "role": "user",
        "content": (
            "That covers everything for the design phase. "
            "Let's get the repo initialised and start the first sprint."
        ),
    },
]

assert len(CHAT_LOG) == 80, f"Expected 80 messages, got {len(CHAT_LOG)}"


# ──────────────────────────────────────────────────────────────────────────────
# 2.  RECALL QUESTIONS  (separate from the chat log)
#
#     Ten questions keyed 0-9.  These are never part of CHAT_LOG — the log is
#     pure conversation.  Both agents are asked every question independently;
#     Agent A gets the full CHAT_LOG as context, Agent B uses its memory store.
# ──────────────────────────────────────────────────────────────────────────────

RECALL_QUESTIONS: dict[int, str] = {
    0: (
        "Implementation sprint is starting. Before we assign tickets, "
        "confirm: what language and framework did we decide on, and what "
        "were the specific reasons given for each choice?"
    ),
    1: (
        "Walk me through the full database decision — what database we chose, "
        "why we chose it, and what we explicitly ruled out for data access."
    ),
    2: (
        "A new backend engineer joins and says they want to use Ecto because "
        "it's the standard Elixir data layer and they're productive with it. "
        "Based on our architectural decisions, what do you tell them?"
    ),
    3: (
        "Detail our authentication approach. What method did we choose, "
        "what alternatives did we explicitly rule out, and what were the "
        "reasons given for ruling them out?"
    ),
    4: (
        "The DevOps lead says Cloud Run would be cheaper and simpler than "
        "our current plan. What is our deployment decision, and does "
        "Cloud Run fit within it?"
    ),
    5: (
        "Explain our caching strategy in full. What did we decide to use, "
        "and why did we explicitly rule out in-process caching?"
    ),
    6: (
        "A frontend engineer joins the team and says they want to build the "
        "product dashboard in React because they're more productive with it "
        "than with LiveView. Based on what we agreed, what do you tell them?"
    ),
    7: (
        "Write the 'Architecture Decisions' section of our project README. "
        "It should cover all six foundational decisions we locked in at the "
        "start of this conversation, with the reasoning for each."
    ),
    8: (
        "A user logs in via magic link and we need to store their session so "
        "they stay authenticated across requests. Which parts of our decided "
        "infrastructure handle this, and how does the data flow?"
    ),
    9: (
        "Quick-fire round: give me one sentence for each of the six decisions "
        "we locked in at the start — language/framework, database, data access "
        "policy, authentication, deployment, and caching."
    ),
}

assert len(RECALL_QUESTIONS) == 10, f"Expected 10 recall questions, got {len(RECALL_QUESTIONS)}"


# ──────────────────────────────────────────────────────────────────────────────
# 3.  GROUND TRUTH  (one authoritative answer per recall question)
#     Keyed 0-9, matching RECALL_QUESTIONS.
#     Written strictly from the decisions recorded in CHAT_LOG.
# ──────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH: dict[int, str] = {
    0: (
        "Elixir as the language with the Phoenix framework. LiveView handles all "
        "interactive UI — no separate SPA framework. Reasons: BEAM VM concurrency "
        "model, fault tolerance via OTP supervision trees, Phoenix LiveView for "
        "real-time UI without JavaScript complexity."
    ),
    1: (
        "CockroachDB for the database — chosen for its distributed-by-default "
        "architecture, PostgreSQL wire compatibility, and geo-distribution capability. "
        "Raw SQL only for data access. Ecto was explicitly ruled out, along with all "
        "query builders. All queries must be hand-written and parameterised."
    ),
    2: (
        "Tell them no. Ecto was explicitly prohibited in our architectural decisions. "
        "Our policy is raw SQL only via hand-written parameterised queries. "
        "Ecto.Repo and ORM query building are banned from the codebase. "
        "Note: Ecto.Changeset for validation is acceptable since it doesn't touch "
        "the database, but Ecto.Repo is not."
    ),
    3: (
        "Magic links only. Users enter their email and receive a single-use "
        "time-limited link. Passwords were ruled out to avoid credential storage "
        "liability and password-reset flow maintenance. OAuth was ruled out to avoid "
        "dependency on third-party provider uptime (Google, GitHub, etc). "
        "Magic link tokens are stored in Redis with a short TTL."
    ),
    4: (
        "Our deployment decision is GKE Kubernetes with a hard minimum of three "
        "replicas. Cloud Run does not fit — serverless and container-as-a-service "
        "platforms were explicitly ruled out. We require proper Kubernetes with "
        "Deployments and Services, using GKE's managed control plane."
    ),
    5: (
        "Redis Cluster for all caching and shared state. In-process caching was "
        "explicitly ruled out because in a multi-replica K8s deployment each pod "
        "would have a divergent cache, causing stale-read bugs that are difficult "
        "to reproduce. Redis Cluster provides a single shared cache layer "
        "accessible consistently to all replicas."
    ),
    6: (
        "Tell them no. Our frontend decision is Phoenix LiveView exclusively for all "
        "interactive UI. React, Vue, and Next.js were explicitly ruled out. The "
        "entire stack is Elixir top to bottom. No SPA frameworks, hybrid approaches, "
        "or per-feature React components are permitted."
    ),
    7: (
        "Architecture Decisions section should cover all six: "
        "1) Elixir/Phoenix/LiveView — BEAM concurrency, fault tolerance, real-time UI "
        "without JS complexity, no SPA frameworks. "
        "2) CockroachDB — distributed, PostgreSQL-compatible, geo-distribution capable. "
        "3) Raw SQL only — Ecto prohibited, all queries hand-written and parameterised. "
        "4) Magic links — no passwords (credential liability), no OAuth (provider dependency). "
        "5) GKE Kubernetes, three-replica minimum — HA by default, no serverless. "
        "6) Redis Cluster — shared cache for all replicas, in-process caching prohibited."
    ),
    8: (
        "Magic link authentication creates a session row in CockroachDB with a 30-day "
        "expiry. A session cookie (HttpOnly, Secure, SameSite=Strict) containing the "
        "session ID is set. On each request the session ID is validated against "
        "CockroachDB. Magic link tokens during the auth ceremony are stored in Redis "
        "with a 15-minute TTL and deleted on consumption. Both CockroachDB (session "
        "persistence) and Redis (token storage) are involved."
    ),
    9: (
        "1. Language/framework: Elixir + Phoenix with LiveView — full-stack Elixir, "
        "no SPA frameworks. "
        "2. Database: CockroachDB — distributed, PostgreSQL-compatible. "
        "3. Data access: raw SQL only — Ecto and all query builders prohibited. "
        "4. Auth: magic links only — no passwords, no OAuth. "
        "5. Deployment: GKE Kubernetes, hard minimum three replicas, no serverless. "
        "6. Caching: Redis Cluster only — no in-process caching."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# 4.  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

MODEL       = "gemma3:4b"
CHROMA_PATH = "./chroma_store_v3"


# ──────────────────────────────────────────────────────────────────────────────
# 5.  PIPELINE BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_pipeline() -> tuple[ActiveAgent, Conductor, Observer, Librarian]:
    observer  = Observer(model=MODEL)
    curator   = Curator(model=MODEL)
    librarian = Librarian(path=CHROMA_PATH)
    reporter  = Reporter(librarian)
    conductor = Conductor(observer, curator, librarian, reporter)
    agent     = ActiveAgent(conductor, model=MODEL)
    return agent, conductor, observer, librarian


# ──────────────────────────────────────────────────────────────────────────────
# 6.  BATCH INGESTION  (Agent B — build memory store, no chat responses)
#
#     conductor.process_message() runs the Observer → Curator → Librarian
#     sequence internally: it updates the rolling summary, passes any new
#     trimmings through the Curator for classification, and stores approved
#     ones in the Librarian.  No conversational LLM response is generated.
#     The return value (assembled context string) is discarded.
# ──────────────────────────────────────────────────────────────────────────────

def batch_ingest(conductor: Conductor, messages: list[dict]) -> None:
    n = len(messages)
    for i, msg in enumerate(messages, 1):
        conductor.process_message(msg["content"], msg["role"])
        if i % 10 == 0 or i == n:
            print(f"    [{i:>3}/{n}] messages ingested")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  AGENT A  (vanilla — full chat log as system prompt)
# ──────────────────────────────────────────────────────────────────────────────

def _format_log_as_system_prompt(messages: list[dict]) -> str:
    lines = [
        "The following is the complete conversation history for a software project.",
        "Answer the question that follows using only information present in this history.",
        "Be specific and stay consistent with every decision that was recorded.\n",
    ]
    for i, msg in enumerate(messages, 1):
        label = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"[{i:02d}] {label}: {msg['content']}")
    return "\n".join(lines)


# Build once at module load; reused for every recall question.
_SYSTEM_PROMPT_A: str = _format_log_as_system_prompt(CHAT_LOG)


def ask_vanilla(question: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT_A},
            {"role": "user",   "content": question},
        ],
    )
    return response.message.content.strip()


# ──────────────────────────────────────────────────────────────────────────────
# 8.  JUDGE
# ──────────────────────────────────────────────────────────────────────────────

def judge_score(ground_truth: str, response: str) -> int:
    """
    Score `response` against `ground_truth` on a 1-5 scale using a third
    Ollama call.  Returns an int; falls back to 3 if output cannot be parsed.
    """
    prompt = (
        f"Given this correct answer: {ground_truth}\n\n"
        f"Score this response from 1 to 5 for consistency and accuracy where "
        f"5 is perfectly consistent and 1 is completely wrong or contradictory. "
        f"Return only a number.\n\n"
        f"Response to score: {response}"
    )
    raw = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    ).message.content.strip()

    match = re.search(r"\b([1-5])\b", raw)
    return int(match.group(1)) if match else 3


# ──────────────────────────────────────────────────────────────────────────────
# 9.  RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────

def _bar(score: int, width: int = 5) -> str:
    return "█" * score + "░" * (width - score)


def print_results_table(
    questions: list[str],
    scores_a:  list[int],
    scores_b:  list[int],
    lats_a:    list[float],
    lats_b:    list[float],
) -> None:
    Q_W  = 38   # question excerpt column
    SC_W = 9    # score column  "  3/5 ███░░"
    LT_W = 8    # latency column "  12.3s"

    # Column widths: Q | ScoreA | ScoreB | LatA | LatB
    divider = (
        "+"
        + "-" * (Q_W  + 2) + "+"
        + "-" * (SC_W + 2) + "+"
        + "-" * (SC_W + 2) + "+"
        + "-" * (LT_W + 2) + "+"
        + "-" * (LT_W + 2) + "+"
    )

    def row(q: str, sa: str, sb: str, la: str, lb: str) -> str:
        return (
            f"| {q:<{Q_W}} "
            f"| {sa:^{SC_W}} "
            f"| {sb:^{SC_W}} "
            f"| {la:^{LT_W}} "
            f"| {lb:^{LT_W}} |"
        )

    print("\n" + divider)
    print(row(
        "Question (excerpt)",
        "Agent A", "Agent B",
        "Lat A",   "Lat B",
    ))
    print(divider)

    for i, (q, sa, sb, la, lb) in enumerate(
        zip(questions, scores_a, scores_b, lats_a, lats_b), start=1
    ):
        q_text  = f"Q{i:02d}: {q[:Q_W - 5].rstrip()}..."
        sa_text = f"{sa}/5 {_bar(sa)}"
        sb_text = f"{sb}/5 {_bar(sb)}"
        la_text = f"{la:.1f}s"
        lb_text = f"{lb:.1f}s"
        print(row(q_text, sa_text, sb_text, la_text, lb_text))

    print(divider)

    avg_sa = sum(scores_a) / len(scores_a)
    avg_sb = sum(scores_b) / len(scores_b)
    avg_la = sum(lats_a)   / len(lats_a)
    avg_lb = sum(lats_b)   / len(lats_b)
    peak_la = max(lats_a)
    peak_lb = max(lats_b)

    print(row(
        "AVERAGE",
        f"{avg_sa:.2f}/5",
        f"{avg_sb:.2f}/5",
        f"{avg_la:.1f}s",
        f"{avg_lb:.1f}s",
    ))
    print(row(
        "PEAK LATENCY",
        "", "",
        f"{peak_la:.1f}s",
        f"{peak_lb:.1f}s",
    ))
    print(divider)

    # Verdict
    score_delta = avg_sb - avg_sa
    lat_delta   = avg_la - avg_lb
    print()
    if score_delta > 0:
        print(f"  Drift score:   Agent B leads by +{score_delta:.2f} points on average.")
    elif score_delta < 0:
        print(f"  Drift score:   Agent A leads by +{abs(score_delta):.2f} points on average.")
    else:
        print("  Drift score:   Both agents scored identically on average.")

    if lat_delta > 0:
        print(f"  Latency:       Agent B was faster by {lat_delta:.1f}s on average.")
    elif lat_delta < 0:
        print(f"  Latency:       Agent A was faster by {abs(lat_delta):.1f}s on average.")
    else:
        print("  Latency:       Both agents had identical average latency.")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 10. LIBRARIAN STORAGE LOG
# ──────────────────────────────────────────────────────────────────────────────

def print_librarian_storage_log(librarian: Librarian) -> None:
    """
    Dump every memory the Curator decided was worth storing.
    Warm memories are things the Curator flagged as immediately relevant;
    cold memories are things it considered worth archiving for later retrieval.
    """
    WIDE = 80

    print(f"\n{'=' * WIDE}")
    print("  LIBRARIAN STORAGE LOG — what the Curator chose to keep")
    print(f"{'=' * WIDE}")

    total = 0
    for tier in ("warm", "cold"):
        result    = librarian._collections[tier].get()
        ids       = result.get("ids")       or []
        documents = result.get("documents") or []

        label = "WARM" if tier == "warm" else "COLD"
        print(f"\n  ── {label} MEMORIES ({len(ids)}) " + "─" * (WIDE - 18 - len(str(len(ids)))))

        if not ids:
            print("  (none)")
            continue

        for i, (mid, doc) in enumerate(zip(ids, documents), 1):
            total += 1
            # Print entry header: index, truncated ID
            print(f"\n  [{i:02d}]  id: {mid[:8]}…")
            # Word-wrap the content to WIDE-6 chars, indented
            indent = "        "
            width  = WIDE - len(indent)
            import textwrap as _tw
            for line in _tw.wrap(doc, width):
                print(f"{indent}{line}")

    print(f"\n  Total memories stored: {total}  "
          f"(warm: {librarian._collections['warm'].count()}  "
          f"cold: {librarian._collections['cold'].count()})")
    print("=" * WIDE)


# ──────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    WIDE = 80
    print("=" * WIDE)
    print("  benchmark_v3.py")
    print("  Agent A: Vanilla (full 80-msg log as system prompt)")
    print("  Agent B: Active Memory (batch ingestion → Conductor retrieval)")
    print("=" * WIDE)
    print(f"\n  Chat log        : {len(CHAT_LOG)} messages (pure conversation)")
    print(f"  Recall questions: {len(RECALL_QUESTIONS)} (separate dict)")
    print(f"  Model           : {MODEL}")
    print(f"  Chroma store    : {CHROMA_PATH}\n")

    # ── Phase 1: Build Agent B memory store ───────────────────────────────────
    print("-" * WIDE)
    print("  PHASE 1 — Batch ingestion: building Active Memory store")
    print("-" * WIDE)
    agent_b, conductor, observer, librarian = build_pipeline()
    batch_ingest(conductor, CHAT_LOG)
    warm = librarian._collections["warm"].count()
    cold = librarian._collections["cold"].count()
    print(f"\n  Store built: {warm} warm memories, {cold} cold memories")
    print(f"  Observer trimmings generated: {len(observer.trimmings)}\n")

    # ── Phase 2: Run recall questions through both agents ─────────────────────
    print("-" * WIDE)
    print("  PHASE 2 — Recall questions: Agent A and Agent B")
    print("-" * WIDE)

    answers_a: list[str]   = []
    answers_b: list[str]   = []
    lats_a:    list[float] = []
    lats_b:    list[float] = []

    for i, question in RECALL_QUESTIONS.items():
        i += 1   # display as 1-indexed
        print(f"\n  Q{i:02d}: {question[:70]}...")

        t0 = time.time()
        ans_a = ask_vanilla(question)
        lats_a.append(time.time() - t0)

        t0 = time.time()
        ans_b = agent_b.chat(question)
        lats_b.append(time.time() - t0)

        answers_a.append(ans_a)
        answers_b.append(ans_b)

        print(f"       Agent A ({lats_a[-1]:.1f}s): {ans_a[:80].replace(chr(10),' ')}...")
        print(f"       Agent B ({lats_b[-1]:.1f}s): {ans_b[:80].replace(chr(10),' ')}...")

    # ── Phase 3: Judge every response ─────────────────────────────────────────
    print(f"\n{'-' * WIDE}")
    print("  PHASE 3 — Scoring responses against ground truth")
    print("-" * WIDE)

    scores_a: list[int] = []
    scores_b: list[int] = []

    for i, (ans_a, ans_b) in enumerate(zip(answers_a, answers_b)):
        gt = GROUND_TRUTH[i]
        print(f"  Judging Q{i + 1:02d} ...", end=" ", flush=True)
        sa = judge_score(gt, ans_a)
        sb = judge_score(gt, ans_b)
        scores_a.append(sa)
        scores_b.append(sb)
        print(f"A = {sa}/5   B = {sb}/5")

    # ── Phase 4: Results table ─────────────────────────────────────────────────
    print(f"\n{'=' * WIDE}")
    print("  RESULTS")
    print("=" * WIDE)

    questions = [RECALL_QUESTIONS[k] for k in sorted(RECALL_QUESTIONS)]
    print_results_table(questions, scores_a, scores_b, lats_a, lats_b)

    # ── Phase 5: Librarian storage log ────────────────────────────────────────
    print_librarian_storage_log(librarian)


if __name__ == "__main__":
    main()
