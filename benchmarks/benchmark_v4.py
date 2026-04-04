# -*- coding: utf-8 -*-
import os
os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

"""
benchmark_v4.py

Controlled comparison between a vanilla agent and the Active Memory pipeline.

  Agent A - Vanilla
    For each recall question the entire 120-message chat log is packed into a
    single system prompt and sent to Ollama alongside the question.  No
    compression, no retrieval.  The model sees everything but has to hold it
    all in its context window at once.

  Agent B - Active Memory
    All 120 messages are batch-ingested via pipeline.chat().  This builds the
    summary (Observer) and memory store (Curator -> Retrieval) as if the
    conversation had actually happened.  Each recall question is then answered
    by the Active Agent, which reads from the compressed Bucket context.

Chat log design
---------------
  Messages 1-15    : six decisions emerge naturally from conversation.
                     They are not labelled or announced -- they surface as
                     conclusions from discussion.  A reader has to pay
                     attention to catch them.
  Messages 16-110  : development discussion with active noise.  At three
                     points a team member suggests an alternative to an
                     earlier decision.  The response to each suggestion is
                     vague enough that a degraded agent could misread it as
                     a reversal.  None of the original decisions are actually
                     overturned.
  Messages 111-120 : ten recall questions.  Each requires the agent to
                     recall the original decision, the reasoning behind it,
                     and reason about a challenge or complication introduced
                     during the noise passages.

Scoring: a judge Ollama call rates each response 1-5 against ground truth.
Output:  results table, average scores and latencies, drift difference, and
         a section showing what the Active Memory pipeline stored.
"""

import sys
import time
import ollama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from active_memory import Pipeline


# ──────────────────────────────────────────────────────────────────────────────
# 1.  CHAT LOG  (120 messages)
#
#     Index   0-14  : six decisions, emerging naturally from discussion
#     Index  15-109 : development discussion, with noise at ~28-31, 55-58, 83-86
#     Index 110-119 : ten recall questions (user role; answered by agents)
#
#     Noise passages:
#       ~28-31  : MySQL floated as alternative database -- vague resolution
#       ~55-58  : EKS/Kubernetes raised by new hire -- left open
#       ~83-86  : Redis revisited as caching alternative -- sounds persuasive
# ──────────────────────────────────────────────────────────────────────────────

CHAT_LOG: list[dict] = [

    # ── Decisions emerge naturally (messages 1-15, indices 0-14) ─────────────

    {   # 0
        "role": "user",
        "content": (
            "I want to settle the frontend architecture before we write a line of "
            "code. I've been burned by SPA sprawl before -- separate build pipelines, "
            "client-server state sync bugs, two deployment artifacts to keep in step. "
            "My instinct is to go server-rendered with LiveView rather than building "
            "a separate React or Vue app on top of a JSON API."
        ),
    },
    {   # 1
        "role": "assistant",
        "content": (
            "LiveView has matured. Real-time updates, minimal JavaScript, one "
            "deployment artifact, server-side rendering by default. You trade "
            "some frontend flexibility for a dramatically simpler stack. "
            "What's the backend language you have in mind?"
        ),
    },
    {   # 2
        "role": "user",
        "content": (
            "Elixir. The BEAM process model is genuinely different from anything "
            "else for concurrent workloads -- lightweight processes, supervision "
            "trees, fault isolation that doesn't require extra infrastructure. "
            "And Phoenix is the natural fit. Same codebase for the backend "
            "and the LiveView frontend, no Node layer."
        ),
    },
    {   # 3
        "role": "assistant",
        "content": (
            "Elixir and Phoenix, LiveView for all interactive UI, no separate "
            "SPA framework. That's a coherent stack. What's the data layer?"
        ),
    },
    {   # 4
        "role": "user",
        "content": (
            "Postgres. Managed RDS. We are not big enough to justify a distributed "
            "database and I'm not paying the operational tax for CockroachDB or "
            "PlanetScale at our scale. The team knows Postgres, the tooling is "
            "mature, and managed RDS gives us backups and PITR without owning the "
            "control plane."
        ),
    },
    {   # 5
        "role": "assistant",
        "content": (
            "PostgreSQL on RDS is the pragmatic call. Managed backups, read "
            "replicas on demand, and point-in-time recovery out of the box. "
            "Are you going ORM or raw SQL?"
        ),
    },
    {   # 6
        "role": "user",
        "content": (
            "Raw SQL. I've watched ORMs obscure what's actually hitting the "
            "database until a query plan falls apart under load and nobody can "
            "explain why. We write parameterized SQL, we read execution plans, "
            "we know exactly what's going to the database. "
            "No Ecto. No query builders. That's the rule for the whole codebase."
        ),
    },
    {   # 7
        "role": "assistant",
        "content": (
            "Handwritten parameterized SQL keeps the execution paths transparent "
            "and the performance debuggable. It takes more upfront discipline "
            "but the long-term maintainability is worth it. "
            "Authentication approach?"
        ),
    },
    {   # 8
        "role": "user",
        "content": (
            "Passkeys. WebAuthn. I don't want stored passwords -- that's credential "
            "storage liability and a reset flow to maintain forever. "
            "I don't want magic links either; I've seen them become a delivery "
            "reliability problem. And no OAuth -- I'm not making our auth path "
            "dependent on Google or GitHub's uptime. "
            "We implement WebAuthn, users authenticate with biometrics or hardware keys."
        ),
    },
    {   # 9
        "role": "assistant",
        "content": (
            "Passkeys are the right call for a greenfield product. No stored "
            "credentials, phishing-resistant by design, biometric UX is already "
            "mainstream on modern devices and browsers. The implementation is "
            "smaller than it looks: register, authenticate, sign count check. "
            "What's the deployment target?"
        ),
    },
    {   # 10
        "role": "user",
        "content": (
            "ECS Fargate. We don't have Kubernetes expertise on the team and I'm "
            "not building it on a production timeline. Fargate handles the "
            "underlying compute -- we define task definitions, set CPU and memory, "
            "AWS runs the containers. Three tasks minimum for availability. "
            "Auto-scaling configured but the floor stays at three."
        ),
    },
    {   # 11
        "role": "assistant",
        "content": (
            "Fargate removes node management entirely. No patching EC2 instances, "
            "no managing node groups. Task definitions are declarative and "
            "rolling deploys are built in. Three-task floor gives you basic HA "
            "without a pod disruption budget. Caching?"
        ),
    },
    {   # 12
        "role": "user",
        "content": (
            "Memcached. We looked at both Redis and Memcached. Our caching use "
            "case is fragment caching and hot query results -- we don't need "
            "persistence or Redis data structures. Memcached is simpler and "
            "multi-threaded. "
            "One hard constraint: no in-process caching anywhere in the codebase. "
            "With multiple Fargate tasks, in-process caches diverge and you "
            "get stale-read bugs that only appear under load."
        ),
    },
    {   # 13
        "role": "assistant",
        "content": (
            "Memcached is the right call when you need a fast distributed cache "
            "without the Redis feature overhead. The in-process constraint is "
            "critical -- an in-process cache is correct with one process and "
            "wrong with more than one. Where do sessions live?"
        ),
    },
    {   # 14
        "role": "user",
        "content": (
            "In the database. Session data lives in Postgres, not in the cache. "
            "I've seen too many stale session bugs from cache-backed sessions. "
            "Authenticated reads against Postgres are fast enough, and we get "
            "consistent session state across all tasks."
        ),
    },

    # ── Development discussion (messages 16-110, indices 15-109) ─────────────
    # Three noise passages woven in: MySQL (~28-31), EKS (~55-58), Redis (~83-86)

    {   # 15
        "role": "assistant",
        "content": (
            "Sessions in Postgres, caching in Memcached, no in-process state. "
            "Clean separation. Let's talk about the data model. "
            "What are the core entities?"
        ),
    },
    {   # 16
        "role": "user",
        "content": (
            "Workspace, User, Membership, Project, Task, Comment, and Label. "
            "Workspace owns Projects. Users belong to Workspaces via Memberships "
            "with a role column. Projects contain Tasks. Tasks can have Comments "
            "and Labels."
        ),
    },
    {   # 17
        "role": "assistant",
        "content": (
            "Standard hierarchy. Workspace -> Project -> Task is the ownership "
            "chain. Membership is a join table with a role enum: owner, admin, "
            "member at minimum. Label is many-to-many with a task_labels join "
            "table -- cleaner than an array column for query composition."
        ),
    },
    {   # 18
        "role": "user",
        "content": (
            "Task statuses: todo, in_progress, in_review, done, cancelled. "
            "Stored as varchar with a check constraint, not a Postgres enum type. "
            "Enum migration requires a lock; varchar plus check constraint "
            "can be altered without downtime. "
            "Status transitions enforced in application code, logged to an audit table."
        ),
    },
    {   # 19
        "role": "assistant",
        "content": (
            "varchar with check constraint is pragmatic. Adding a value to a "
            "Postgres enum type requires a table lock; varchar avoids that. "
            "Application-layer state machine with an audit log is clean and "
            "testable without database triggers. "
            "What indexes go on the tasks table from day one?"
        ),
    },
    {   # 20
        "role": "user",
        "content": (
            "Composite index on (project_id, status) for the main task list. "
            "Index on assignee_id for the my-tasks view. "
            "Index on due_date with NULLS LAST so unset tasks sort to the bottom. "
            "All defined in named raw SQL migration files so drops are clean."
        ),
    },
    {   # 21
        "role": "assistant",
        "content": (
            "Composite on (project_id, status) covers the most common read path. "
            "Naming indexes explicitly in migrations is good hygiene -- "
            "pg_dump includes the names and concurrent index drops are cleaner. "
            "NULLS LAST on due_date is easy to forget and painful to add later."
        ),
    },
    {   # 22
        "role": "user",
        "content": (
            "LiveView architecture: three top-level LiveViews. "
            "WorkspaceLayout is the shell -- sidebar, nav, notification tray. "
            "ProjectBoard renders the kanban or list view. "
            "UserDashboard is the personal view across all workspaces. "
            "WorkspaceLayout wraps the other two via live_render."
        ),
    },
    {   # 23
        "role": "assistant",
        "content": (
            "Nesting via live_render is the right pattern. WorkspaceLayout "
            "owns the auth state, workspace context, and notification PubSub "
            "subscriptions. ProjectBoard and UserDashboard mount inside it. "
            "PubSub topic structure should mirror the hierarchy: "
            "workspace:{id}, project:{id}, user:{id}."
        ),
    },
    {   # 24
        "role": "user",
        "content": (
            "Real-time task updates: when a task is mutated, broadcast on "
            "the project:{id} topic. All connected ProjectBoard LiveViews "
            "for that project receive the message and update in place. "
            "Broadcast payload is just the changed task struct -- minimal diff."
        ),
    },
    {   # 25
        "role": "assistant",
        "content": (
            "Minimal diff broadcast is the right tradeoff. Sending the full "
            "task list on every change is expensive at scale. The LiveView "
            "handle_info callback pattern-matches on the task id and updates "
            "just that row in assigns. Clean and testable."
        ),
    },
    {   # 26
        "role": "user",
        "content": (
            "CI/CD: GitHub Actions. On every push: mix format, mix credo, "
            "mix dialyzer, mix test. Dialyzer runs against a cached PLT "
            "keyed on mix.lock hash to avoid rebuild on every run. "
            "Pipeline must be green before any merge. No force-push to main."
        ),
    },
    {   # 27
        "role": "assistant",
        "content": (
            "mix format plus credo plus dialyzer plus test is the correct "
            "Elixir CI stack. PLT cache on mix.lock hash is standard. "
            "Dialyzer is slow on cold cache but fast on warm -- most teams "
            "accept the cold cache cost on dependency updates. "
            "Branch protection rules enforce the no-force-push policy."
        ),
    },

    # ── NOISE 1: MySQL floated as alternative (~messages 29-32, indices 28-31) ─

    {   # 28
        "role": "user",
        "content": (
            "I want to revisit the database question briefly. The ops team lead "
            "came to me saying they have monitoring tooling and runbooks built "
            "around MySQL specifically -- RDS MySQL, not Postgres. "
            "They said the switch to Postgres would require rebuilding their "
            "dashboards and on-call procedures from scratch. "
            "Has anyone done a serious comparison for our workload?"
        ),
    },
    {   # 29
        "role": "assistant",
        "content": (
            "MySQL InnoDB is production-grade. The gap between MySQL 8 and "
            "Postgres has narrowed considerably. For a CRUD-heavy application "
            "at our scale the performance characteristics are broadly similar. "
            "The ops familiarity argument is real -- a team that knows their "
            "database makes fewer mistakes under pressure."
        ),
    },
    {   # 30
        "role": "user",
        "content": (
            "I said I'd think about it. The ops overhead reduction could be "
            "significant if they're already running MySQL in production "
            "elsewhere. I'll check whether the cost of rebuilding their "
            "tooling is actually as high as they say."
        ),
    },
    {   # 31
        "role": "assistant",
        "content": (
            "Worth verifying. The replication story is different -- Postgres "
            "streaming replication versus MySQL GTID-based -- but both work "
            "at our scale. If the ops team has significantly more depth in "
            "MySQL monitoring and incident response, that's a meaningful "
            "operational argument worth taking seriously."
        ),
    },

    # ── Continued development discussion ─────────────────────────────────────

    {   # 32
        "role": "user",
        "content": (
            "Let's move on. The JSON API for the mobile client: "
            "Phoenix Router gets an /api/v1 scope with its own plug pipeline. "
            "We can't use passkeys on native mobile -- WebAuthn requires a "
            "browser context. The API will use JWT authentication. "
            "Short-lived access tokens, 15-minute TTL, "
            "refresh tokens stored in the database."
        ),
    },
    {   # 33
        "role": "assistant",
        "content": (
            "Separate /api/v1 scope with its own plug pipeline is idiomatic "
            "Phoenix. JWT for the REST API makes sense -- passkeys authenticate "
            "the browser session and the API gets a JWT derived from that. "
            "Refresh token rotation on use to detect replay. "
            "Short-lived access tokens with DB-stored refresh tokens is the "
            "right security tradeoff."
        ),
    },
    {   # 34
        "role": "user",
        "content": (
            "Passkey registration flow: user visits /auth/register. "
            "Server generates a cryptographically random challenge, stores "
            "it in the session with a short TTL. Frontend calls "
            "navigator.credentials.create with the challenge. User completes "
            "biometric. Frontend sends the credential response to the server. "
            "Server verifies, stores credential_id and public_key_cbor."
        ),
    },
    {   # 35
        "role": "assistant",
        "content": (
            "Challenge generation with :crypto.strong_rand_bytes(32) and "
            "session-stored TTL prevents replay. For most consumer apps you "
            "skip full attestation verification and just verify the signature. "
            "Store credential_id, public_key_cbor, sign_count, transports, "
            "and user_id. The wax or webauthn_components library handles the "
            "CBOR parsing and verification ceremony."
        ),
    },
    {   # 36
        "role": "user",
        "content": (
            "Authentication flow: server generates challenge. Frontend calls "
            "navigator.credentials.get -- either with allowCredentials for "
            "a known user or empty for conditional UI (discoverable credentials). "
            "User completes biometric. Frontend sends assertion response. "
            "Server verifies signature against stored public key, "
            "checks sign_count to detect cloned credentials, creates session."
        ),
    },
    {   # 37
        "role": "assistant",
        "content": (
            "Sign count check is critical -- if stored_count >= received_count "
            "the credential may be cloned. Policy decision whether to hard-reject "
            "or flag and alert. Conditional UI is the modern passkey UX pattern "
            "and worth supporting from day one. It lets the browser surface "
            "the passkey prompt without the user typing a username."
        ),
    },
    {   # 38
        "role": "user",
        "content": (
            "Error handling: all database errors are logged with full context "
            "-- query, params, error code -- and surfaced to the user as "
            "generic messages. No raw Postgres error messages to the client. "
            "Known constraint violations get specific messages. "
            "Unknown errors get a generic message plus a trace ID."
        ),
    },
    {   # 39
        "role": "assistant",
        "content": (
            "Leaking constraint names or table names to clients is an "
            "information disclosure risk. Trace IDs are essential for support "
            "workflows -- the user reports the ID, you correlate it in the "
            "log aggregator. Full-context DB logging on the server side keeps "
            "debugging viable without exposing internals."
        ),
    },
    {   # 40
        "role": "user",
        "content": (
            "Background jobs via Oban. Jobs are persisted in Postgres so they "
            "survive restarts and are exactly-once with transactional guarantees. "
            "Email notifications, data export, and scheduled aggregation jobs "
            "all go through Oban. Critical path: enqueue after commit, "
            "not inside the transaction."
        ),
    },
    {   # 41
        "role": "assistant",
        "content": (
            "Post-commit Oban enqueue is critical. Enqueuing inside the "
            "transaction risks a job running before the commit, finding no "
            "data, and failing. Oban's with_job/1 pattern or a manual "
            "post-commit hook handles this. Postgres-backed job persistence "
            "means no additional queue infrastructure."
        ),
    },
    {   # 42
        "role": "user",
        "content": (
            "Telemetry: Postgrex telemetry events for all database queries, "
            "Phoenix.LiveView.Telemetry for LiveView lifecycle events, "
            "custom business events via :telemetry.execute. "
            "Metrics go to Prometheus via prometheus_ex. Dashboards in Grafana. "
            "PagerDuty escalation for P1 and P2."
        ),
    },
    {   # 43
        "role": "assistant",
        "content": (
            "Postgrex telemetry gives query duration, result size, and "
            "connection pool saturation without application instrumentation. "
            "Phoenix.LiveView telemetry covers mount, handle_event, "
            "handle_info latencies -- critical for catching slow callbacks. "
            "Prometheus plus Grafana is the standard OSS observability stack."
        ),
    },
    {   # 44
        "role": "user",
        "content": (
            "Database migrations: we use mix ecto.migrate but since we are "
            "raw SQL only, migration files use Ecto.Migration's execute/2 "
            "with explicit up and down SQL. Migrations run as part of the "
            "deployment pipeline before new ECS tasks start. "
            "Migration failure aborts the deployment."
        ),
    },
    {   # 45
        "role": "assistant",
        "content": (
            "execute/2 in Ecto.Migration with raw SQL up and down is the "
            "correct approach given the no-ORM constraint. Pre-traffic "
            "migrations require backward-compatible changes for zero-downtime "
            "deploys -- additive first, clean up old columns in a follow-up "
            "migration after the old code is fully retired."
        ),
    },
    {   # 46
        "role": "user",
        "content": (
            "Rate limiting: plug_attack for the auth endpoints. "
            "Three failed passkey attempts from the same IP in 60 seconds "
            "triggers a 60-second block. "
            "API endpoints get a per-token rate limit of 1000 requests "
            "per minute, tracked in Memcached with a 60-second TTL counter."
        ),
    },
    {   # 47
        "role": "assistant",
        "content": (
            "plug_attack with Memcached as the backing store works well. "
            "Per-IP auth throttling is the right scope for passkey failures. "
            "API rate limiting per token in Memcached: increment a counter "
            "with a 60-second TTL, return 429 if over 1000. "
            "Simple and correct for our scale."
        ),
    },
    {   # 48
        "role": "user",
        "content": (
            "Comments support Markdown. Server-side render to HTML via Earmark "
            "before storage. Store both raw Markdown and rendered HTML -- "
            "re-rendering on every read is expensive at comment volume. "
            "HTML is sanitised with HtmlSanitizeEx before storage. "
            "Sanitise then store, never sanitise on read."
        ),
    },
    {   # 49
        "role": "assistant",
        "content": (
            "Store both and sanitise at write is correct. Sanitising on "
            "every read trusts the sanitiser to be safe on untrusted input "
            "every single time -- that raises the blast radius of any "
            "sanitiser bug. Sanitise once at write, trust the stored HTML."
        ),
    },
    {   # 50
        "role": "user",
        "content": (
            "Full-text search using Postgres built-in FTS. tsvector columns "
            "on tasks (title plus description) and comments (body_raw). "
            "Updated by trigger on insert and update. "
            "websearch_to_tsquery for query parsing. "
            "We are not adding Elasticsearch or Meilisearch at this stage."
        ),
    },
    {   # 51
        "role": "assistant",
        "content": (
            "Postgres FTS is correct for our data volume. tsvector maintained "
            "by trigger keeps the index current without application changes. "
            "websearch_to_tsquery handles quoted phrases and negation naturally. "
            "No external search service means no sync lag and no extra ops."
        ),
    },
    {   # 52
        "role": "user",
        "content": (
            "File attachments: tasks and comments can have attachments. "
            "Presigned S3 URLs -- server generates the URL, client uploads "
            "directly. Server stores only the S3 key and metadata. "
            "No files pass through our application servers. "
            "50MB max enforced by S3 bucket policy."
        ),
    },
    {   # 53
        "role": "assistant",
        "content": (
            "Direct-to-S3 upload via presigned URLs is correct. Proxying "
            "uploads through the application server wastes bandwidth, "
            "increases latency, and requires buffering the full file. "
            "Store S3 key, content_type, size, and original_filename. "
            "Generate presigned download URLs on demand, not stored publicly."
        ),
    },
    {   # 54
        "role": "user",
        "content": (
            "Billing model: Workspace plans are free, pro, or enterprise. "
            "Free: 5 projects, 10 members. Pro: unlimited projects, 50 members. "
            "Enterprise: negotiated. Plan stored on the workspace record. "
            "Limit enforcement is in a Policies module in the application layer, "
            "not in the database."
        ),
    },

    # ── NOISE 2: EKS/Kubernetes raised by new hire (~messages 56-59, indices 55-58) ─

    {   # 55
        "role": "user",
        "content": (
            "Quick sidebar. Our new DevOps engineer starts Monday. She has "
            "three Kubernetes certifications and two years of EKS production "
            "experience. When I told her we're using Fargate she pushed back "
            "pretty hard. Said we'd have less observability -- no access to "
            "node-level metrics, limited networking visibility, fewer knobs "
            "to tune under load. She's asking if EKS is worth reconsidering."
        ),
    },
    {   # 56
        "role": "assistant",
        "content": (
            "She has a point about observability. Fargate abstracts the node "
            "layer which means you lose direct access to node-level metrics, "
            "and some cluster tooling assumes node access. EKS with managed "
            "node groups has gotten significantly simpler to operate -- if "
            "she's bringing two years of EKS production experience, the "
            "operational complexity argument weakens considerably."
        ),
    },
    {   # 57
        "role": "user",
        "content": (
            "I said I'd look at it. The cost model is also different -- "
            "Fargate tasks are more expensive per vCPU than EC2 nodes in "
            "some configurations, and EKS with spot node groups could cut "
            "compute costs. I told her to put together a comparison before "
            "we've committed anything in code."
        ),
    },
    {   # 58
        "role": "assistant",
        "content": (
            "The cost comparison does favor EKS at moderate scale. Spot "
            "node groups for non-critical workloads can reduce compute costs "
            "significantly. If she has the depth to own the cluster, the "
            "operational overhead argument becomes much weaker. Worth having "
            "the formal comparison before finalizing the deployment story."
        ),
    },

    # ── Continued development discussion ─────────────────────────────────────

    {   # 59
        "role": "user",
        "content": (
            "Notifications: in-app and optional email. In-app notifications "
            "stored in a notifications table and pushed to the browser via "
            "PubSub on the user:{id} topic. Email goes through an Oban job "
            "to an SES-backed SMTP relay. Per-user per-event-type preferences."
        ),
    },
    {   # 60
        "role": "assistant",
        "content": (
            "Oban for email jobs is the right call -- persisted in Postgres, "
            "exactly-once semantics, retry with backoff. SES for email is "
            "cost-effective at low volume. Per-user per-event-type preferences "
            "table: (user_id, event_type, channel, enabled). "
            "Sparse but flexible as the notification types grow."
        ),
    },
    {   # 61
        "role": "user",
        "content": (
            "Cursor-based pagination throughout. No page numbers. "
            "Cursors are opaque base64-encoded (inserted_at, id) tuples. "
            "Fetch page_size plus one to detect the next page without a "
            "COUNT query. Return the extra row's cursor as next_cursor."
        ),
    },
    {   # 62
        "role": "assistant",
        "content": (
            "Cursor pagination on (inserted_at, id) is stable under concurrent "
            "inserts where page numbers are not. Fetching n+1 to detect the "
            "next page avoids the COUNT and handles the last page cleanly. "
            "Opaque base64 cursors prevent clients from constructing "
            "arbitrary offsets."
        ),
    },
    {   # 63
        "role": "user",
        "content": (
            "Secret management: everything in AWS Secrets Manager. "
            "ECS task definitions reference secrets by ARN. "
            "The application reads them at container startup. "
            "No secrets baked into the Docker image or stored in "
            "environment variables that ship with the artifact. "
            "Database credentials rotate via a Lambda rotation function."
        ),
    },
    {   # 64
        "role": "assistant",
        "content": (
            "ECS native secrets integration decrypts and injects at container "
            "start without the application needing IAM SDK calls. "
            "Lambda rotation for RDS credentials is the managed approach -- "
            "Secrets Manager handles the schedule and the application "
            "re-reads on reconnect after a connection failure."
        ),
    },
    {   # 65
        "role": "user",
        "content": (
            "Backups: RDS automated daily snapshots with 30-day retention, "
            "point-in-time recovery enabled. S3 versioning with a 90-day "
            "lifecycle on old versions. Quarterly restore drills to verify "
            "snapshots are actually usable. CloudWatch alert if no snapshot "
            "in 26 hours."
        ),
    },
    {   # 66
        "role": "assistant",
        "content": (
            "30-day RDS retention with PITR is solid. 26-hour alert gives "
            "a buffer over the daily schedule. Quarterly restore drills are "
            "essential -- an untested backup is not a backup. S3 versioning "
            "with lifecycle policy balances recovery capability against cost."
        ),
    },
    {   # 67
        "role": "user",
        "content": (
            "Audit logging: all mutations to core entities written to an "
            "audit_events table in the same database transaction. "
            "Schema: entity_type, entity_id, action, actor_user_id, "
            "changes as jsonb, inserted_at. "
            "No mutation without the audit record, no audit record without "
            "the mutation."
        ),
    },
    {   # 68
        "role": "assistant",
        "content": (
            "Same-transaction audit writes are correct for consistency. "
            "jsonb for the changes column is flexible enough to handle "
            "different entity shapes. Indexes on (entity_type, entity_id) "
            "and (actor_user_id, inserted_at) cover the main query patterns. "
            "Partition by inserted_at month if write volume grows."
        ),
    },
    {   # 69
        "role": "user",
        "content": (
            "Feature flags: database-backed, simple. "
            "A feature_flags table: flag_name, enabled, rollout_percentage. "
            "A FlagCache module reads from Memcached with a 30-second TTL, "
            "falls back to the database on miss. "
            "Flags.enabled?(flag_name, user) in application code."
        ),
    },
    {   # 70
        "role": "assistant",
        "content": (
            "Database-backed flags with Memcached caching is appropriately "
            "simple for an early product. Rollout percentage with user ID "
            "hashing gives deterministic per-user assignment. "
            "30-second TTL means flag changes propagate within half a minute "
            "without needing a cache flush mechanism."
        ),
    },
    {   # 71
        "role": "user",
        "content": (
            "Workspace invitation flow: invite by email. Server generates "
            "a signed token, stores it in the invitations table with TTL. "
            "Email sends a link containing the token. Recipient clicks, "
            "token is validated, Membership is created. "
            "Invitations can be revoked by cancelling the token."
        ),
    },
    {   # 72
        "role": "assistant",
        "content": (
            "Signed token in the invitations table with a TTL is clean. "
            "Store the token hash, not the raw token -- the link contains "
            "the raw token, the database stores the hash so a DB read "
            "doesn't expose the invitation URL. "
            "Single-use: invalidate on acceptance."
        ),
    },
    {   # 73
        "role": "user",
        "content": (
            "Task assignment notification: User A assigns Task T to User B. "
            "Mutation fires, audit log written in the same transaction. "
            "After commit, Oban job enqueued for email notification. "
            "PubSub broadcast on user:{B.id}. "
            "User B's dashboard LiveView increments the notification badge."
        ),
    },
    {   # 74
        "role": "assistant",
        "content": (
            "Post-commit Oban enqueue is critical here. Enqueueing inside "
            "the transaction risks the job running before the commit completes. "
            "PubSub broadcast can go after handle_event returns -- it races "
            "with the commit but that's acceptable for a UI update."
        ),
    },
    {   # 75
        "role": "user",
        "content": (
            "Testing strategy: ExUnit for all unit and integration tests. "
            "Database tests use the sandbox adapter for isolation. "
            "LiveView tests via Phoenix.LiveViewTest. "
            "No end-to-end browser tests in CI -- too flaky at this stage. "
            "Comprehensive LiveView unit tests cover all user interactions."
        ),
    },
    {   # 76
        "role": "assistant",
        "content": (
            "Phoenix.LiveViewTest covers the full LiveView lifecycle without "
            "a browser. Sandbox adapter gives each test its own transaction "
            "that rolls back at the end. Skipping browser tests in CI is "
            "pragmatic -- the flakiness cost outweighs the coverage benefit "
            "at early stage."
        ),
    },
    {   # 77
        "role": "user",
        "content": (
            "Docker build: multi-stage. Stage one is the Elixir builder image "
            "-- fetch deps, compile, build the mix release. Stage two is a "
            "minimal runtime image with just the release binary. "
            "Final image under 100MB. GitHub Actions builds and pushes to ECR. "
            "ECS service update via aws ecs update-service."
        ),
    },
    {   # 78
        "role": "assistant",
        "content": (
            "Mix release binary is self-contained -- the runtime image only "
            "needs the OS libraries the BEAM links against. Alpine-based "
            "runtime images get under 80MB comfortably. "
            "ECR plus ECS update-service with minimum healthy percent 100 "
            "ensures zero-downtime rolling deploys."
        ),
    },
    {   # 79
        "role": "user",
        "content": (
            "RBAC model: roles live on the Membership record, not on the user. "
            "Owner, admin, member. Permissions enforced by a Policies module. "
            "No database-level row security -- application layer only. "
            "Authorization checks happen at the LiveView mount and "
            "in every mutation path."
        ),
    },
    {   # 80
        "role": "assistant",
        "content": (
            "Application-layer RBAC in a Policies module is testable and "
            "visible. Row-level security in Postgres is powerful but adds "
            "implicit behavior that's hard to reason about in application "
            "code. Explicit checks at mount and mutation boundaries are "
            "clearer and easier to audit."
        ),
    },
    {   # 81
        "role": "user",
        "content": (
            "Data export: workspace owners can export all data as CSV or JSON. "
            "Export job runs via Oban, generates the file, uploads to S3, "
            "sends a download link by email. "
            "Large exports are streamed to S3 in parts, not held in memory."
        ),
    },
    {   # 82
        "role": "assistant",
        "content": (
            "S3 multipart upload for large exports is correct -- holding "
            "a large export in memory risks OOM on a constrained Fargate task. "
            "Stream the query results, chunk into parts, upload in sequence. "
            "Oban handles retry if the job fails partway through."
        ),
    },

    # ── NOISE 3: Redis revisited as caching alternative (~messages 84-87, indices 83-86) ─

    {   # 83
        "role": "user",
        "content": (
            "I want to revisit the caching decision again. I've been looking "
            "more closely at our LiveView clustering setup. Phoenix PubSub has "
            "a Redis adapter that some teams use for multi-node distribution. "
            "Redis sorted sets would also make the rate limiting logic cleaner "
            "than a Memcached counter. And if we're running Redis for PubSub "
            "anyway, why have two separate caching systems?"
        ),
    },
    {   # 84
        "role": "assistant",
        "content": (
            "The consolidation argument is real. If Redis is already in the "
            "stack for PubSub, the case for Memcached weakens -- you're "
            "paying for operational complexity of two systems instead of one. "
            "Redis sorted sets would make sliding-window rate limiting more "
            "elegant. And Redis Cluster has native pub/sub which aligns "
            "with the notification architecture."
        ),
    },
    {   # 85
        "role": "user",
        "content": (
            "The data structure richness is genuinely appealing. "
            "I'm going to research this further before the next architecture "
            "review. The persistence options might also simplify the "
            "session question, though I know we said sessions go in Postgres."
        ),
    },
    {   # 86
        "role": "assistant",
        "content": (
            "Redis has become the default caching layer for a reason. "
            "The operational gap between Redis Cluster and Memcached has "
            "narrowed. If Redis is already present in the stack, its "
            "feature set makes a lot of other problems easier to solve. "
            "It's worth putting together a formal comparison."
        ),
    },

    # ── Continued development discussion ─────────────────────────────────────

    {   # 87
        "role": "user",
        "content": (
            "Analytics and reporting: materialized views in Postgres for "
            "aggregate metrics -- tasks completed per week, member activity, "
            "project burndown. Refreshed by a scheduled Oban job. "
            "No external analytics service at this stage."
        ),
    },
    {   # 88
        "role": "assistant",
        "content": (
            "Materialized views refreshed by Oban jobs are the right call "
            "for aggregate metrics at our scale. CONCURRENT REFRESH avoids "
            "blocking reads. No external analytics service means no data "
            "export compliance surface and no vendor dependency for internal "
            "metrics."
        ),
    },
    {   # 89
        "role": "user",
        "content": (
            "Workspace settings model: settings stored as a JSONB column "
            "on the workspace record for flexible schema evolution. "
            "Typed access via an Elixir struct that serializes to and "
            "from the JSONB column. Validation at the application layer."
        ),
    },
    {   # 90
        "role": "assistant",
        "content": (
            "JSONB with a typed struct wrapper gives you schema evolution "
            "flexibility without Postgres schema migrations for every new "
            "setting. The struct validation in application code catches "
            "malformed values before they reach the database."
        ),
    },
    {   # 91
        "role": "user",
        "content": (
            "Search result ranking: tsvector with ts_rank for basic relevance. "
            "Boost title matches over description matches using weighted "
            "tsvector. The headline function highlights matched terms "
            "in the result snippet."
        ),
    },
    {   # 92
        "role": "assistant",
        "content": (
            "Weighted tsvector with setweight is the Postgres FTS approach "
            "for boosting title over body. ts_rank gives you a float "
            "you can order by. headline generates the snippet with "
            "match highlighting -- configurable max length and separator."
        ),
    },
    {   # 93
        "role": "user",
        "content": (
            "Onboarding flow: a guided setup LiveView for new workspaces. "
            "Steps: create workspace, invite team members, create first project, "
            "create first task. Progress tracked as a JSONB field on the "
            "workspace record. Skippable at any step."
        ),
    },
    {   # 94
        "role": "assistant",
        "content": (
            "Guided setup as a LiveView with JSONB progress tracking is clean. "
            "The step state lives in the database so refresh doesn't lose "
            "progress. Each step is a separate LiveView component that mounts "
            "and sends events back to the parent. Skippable means marking "
            "the step completed without the action."
        ),
    },
    {   # 95
        "role": "user",
        "content": (
            "Mobile app: separate React Native app calling the /api/v1 endpoints. "
            "JWT authentication as established. The mobile app is its own "
            "repository and deployment -- it's not part of the Phoenix "
            "application, just a consumer of the API."
        ),
    },
    {   # 96
        "role": "assistant",
        "content": (
            "Separate repository for the mobile app is the right boundary. "
            "It's a different release cycle, different deployment mechanism, "
            "and different team skillset. The API contract is the interface. "
            "JWT auth is the correct choice for a native mobile context "
            "where WebAuthn isn't available."
        ),
    },
    {   # 97
        "role": "user",
        "content": (
            "Access control for the API: API tokens are scoped per workspace. "
            "A user can generate API tokens from their profile settings. "
            "Token validation via a plug that checks the token against "
            "a lookup in Memcached first, then Postgres on miss."
        ),
    },
    {   # 98
        "role": "assistant",
        "content": (
            "Memcached token validation cache is the right pattern -- fast "
            "path is a cache hit, slow path is a Postgres lookup that then "
            "warms the cache. Token invalidation needs to flush the cache "
            "entry explicitly on revocation."
        ),
    },
    {   # 99
        "role": "user",
        "content": (
            "Deployment checklist before each release: "
            "CI green, migration review, health check endpoint responding, "
            "rollback plan documented, feature flags for new functionality. "
            "No deploy on Fridays. Migrations run before traffic shifts."
        ),
    },
    {   # 100
        "role": "assistant",
        "content": (
            "Pre-traffic migrations with a documented rollback plan is the "
            "correct discipline. Feature flags for new functionality means "
            "you can deploy the code without activating the feature -- "
            "dark deploy then controlled rollout. No Friday deploys is a "
            "good policy for a small team without 24/7 on-call coverage."
        ),
    },
    {   # 101
        "role": "user",
        "content": (
            "Project archiving: archived projects are soft-deleted -- "
            "is_archived boolean, archived_at timestamp. "
            "Archived projects are excluded from default list queries "
            "via a is_archived = false predicate. "
            "Hard delete is available but requires workspace owner confirmation."
        ),
    },
    {   # 102
        "role": "assistant",
        "content": (
            "Soft delete via is_archived is simpler than a separate "
            "archived_projects table for this use case. The is_archived "
            "predicate needs to be on an index alongside project_id "
            "wherever it appears in queries -- a partial index "
            "WHERE is_archived = false might be more efficient."
        ),
    },
    {   # 103
        "role": "user",
        "content": (
            "Recurring tasks: a recurring_configs table that stores the "
            "recurrence rule (RRULE string), last_generated_at, and "
            "template task data. An Oban job runs daily, finds configs "
            "where next occurrence is due, generates the task from the template."
        ),
    },
    {   # 104
        "role": "assistant",
        "content": (
            "RRULE string storage is flexible and well-understood. "
            "An Elixir RRULE parser library (ex_rrule or similar) handles "
            "next occurrence calculation. Daily Oban job with a lookback "
            "window handles missed runs if the job fails. "
            "Index on last_generated_at for the job query."
        ),
    },
    {   # 105
        "role": "user",
        "content": (
            "Webhook support: workspaces can configure webhooks for events "
            "like task.created, task.updated, member.added. "
            "Webhook deliveries are queued via Oban, delivered with a "
            "signed HMAC-SHA256 payload. "
            "Delivery log stored for 7 days for debugging."
        ),
    },
    {   # 106
        "role": "assistant",
        "content": (
            "Oban-backed webhook delivery with retry and exponential backoff "
            "is correct. HMAC-SHA256 signing lets receivers verify the "
            "payload. Delivery log in Postgres with a 7-day retention "
            "gives debugging visibility without unbounded growth. "
            "Store attempt count and last_error on the delivery record."
        ),
    },
    {   # 107
        "role": "user",
        "content": (
            "Guest access: workspaces can make individual projects publicly "
            "readable without authentication. A project-level is_public flag. "
            "Public project views are server-rendered LiveView pages "
            "with no auth requirement. Write operations still require auth."
        ),
    },
    {   # 108
        "role": "assistant",
        "content": (
            "is_public flag on the project record keeps it simple. "
            "LiveView mount checks the flag and skips the auth redirect "
            "for public projects. The auth plug on write paths still fires "
            "regardless of the public flag -- the public flag is a read "
            "permission, not a write permission."
        ),
    },
    {   # 109
        "role": "user",
        "content": (
            "That covers the main architecture. We've gone deep on a lot "
            "of implementation details. Before we move to sprint planning "
            "I want to do a consistency check -- make sure we haven't "
            "drifted from the foundation decisions we started with, "
            "given how much discussion has happened since."
        ),
    },

    # ── Recall questions (messages 111-120, indices 110-119) ─────────────────
    # Harder: require original decision + reasoning + reasoning about a challenge.

    {   # 110
        "role": "user",
        "content": (
            "RECALL 1: Someone on the team is pushing hard to switch to MySQL, "
            "citing the ops team's existing tooling and familiarity. "
            "What database did we commit to and what was the reasoning? "
            "Does the ops familiarity argument actually change the analysis?"
        ),
    },
    {   # 111
        "role": "user",
        "content": (
            "RECALL 2: There was a discussion in this conversation about "
            "Kubernetes versus what we actually landed on for deployment. "
            "A new team member joining next week will ask what our platform is. "
            "What is the answer, and what was the reasoning that led there? "
            "Did the conversation about EKS change anything?"
        ),
    },
    {   # 112
        "role": "user",
        "content": (
            "RECALL 3: A contractor being brought in wants to use Ecto to "
            "speed up a new feature. What is the codebase rule on database "
            "access and how was it established in this conversation? "
            "Is there any flexibility or exception allowed?"
        ),
    },
    {   # 113
        "role": "user",
        "content": (
            "RECALL 4: We're hiring a frontend engineer. The job description "
            "needs to accurately describe the environment. "
            "Is this a React codebase, a LiveView codebase, or both? "
            "What did we decide and what alternatives did we explicitly rule out?"
        ),
    },
    {   # 114
        "role": "user",
        "content": (
            "RECALL 5: Redis came up more than once in this conversation as a "
            "potential alternative to what we chose for caching. "
            "What is the current caching decision, what was the original "
            "reasoning for it, and has anything in the conversation "
            "formally changed it?"
        ),
    },
    {   # 115
        "role": "user",
        "content": (
            "RECALL 6: An engineer is arguing that in-process caching would "
            "be faster than a network call to the caching layer on every "
            "request. What is our constraint on in-process caching, "
            "and what was the specific reasoning given for it?"
        ),
    },
    {   # 116
        "role": "user",
        "content": (
            "RECALL 7: Authentication has two distinct contexts in this system. "
            "What does a browser user use to authenticate, and what does the "
            "mobile API use? These are different and both were explicitly "
            "decided in this conversation."
        ),
    },
    {   # 117
        "role": "user",
        "content": (
            "RECALL 8: A developer on the team assumes sessions are cached "
            "in Memcached for performance. Are they correct? "
            "Where do sessions actually live and what was the reasoning?"
        ),
    },
    {   # 118
        "role": "user",
        "content": (
            "RECALL 9: The auth decision ruled out several alternatives. "
            "List every authentication method that was explicitly considered "
            "and rejected for the main application, and give the reason "
            "each was ruled out."
        ),
    },
    {   # 119
        "role": "user",
        "content": (
            "RECALL 10: If the DevOps engineer makes a compelling case and "
            "we do switch to EKS, and separately the team agrees to move to "
            "Redis, and separately we move to MySQL -- in each case, what is "
            "the current decision being replaced, what was the original "
            "reasoning, and what would need to be re-evaluated in the "
            "architecture as a result?"
        ),
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# 2.  RECALL QUESTIONS  (indices 110-119, extracted for iteration)
# ──────────────────────────────────────────────────────────────────────────────

RECALL_QUESTIONS: dict[int, str] = {
    q_num: CHAT_LOG[109 + q_num]["content"]
    for q_num in range(1, 11)
}

# ──────────────────────────────────────────────────────────────────────────────
# 3.  GROUND TRUTH
# ──────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH: dict[int, str] = {
    1: (
        "The database is PostgreSQL on managed RDS. Reasoning: the team knows it, "
        "tooling is mature, managed RDS provides backups and PITR, and the scale "
        "does not justify a distributed database. The MySQL familiarity argument "
        "does not change the original choice -- PostgreSQL was committed to and "
        "never overturned in the conversation."
    ),
    2: (
        "The deployment platform is AWS ECS Fargate with a three-task minimum. "
        "Reasoning: no Kubernetes expertise on the team, Fargate removes node "
        "management, task definitions are declarative. The EKS discussion was a "
        "consideration raised by the new hire but the original Fargate decision "
        "was never formally reversed."
    ),
    3: (
        "The rule is raw SQL only -- no Ecto, no Ecto.Repo, no query builders "
        "of any kind. Every database call is a handwritten parameterized SQL query. "
        "This is a hard codebase rule with no exceptions. Ecto.Migration is used "
        "for migrations but only with execute/2 wrapping raw SQL."
    ),
    4: (
        "This is a LiveView codebase. Elixir with Phoenix and Phoenix LiveView "
        "for all interactive UI. React, Vue, and Next.js were all explicitly "
        "ruled out. The frontend engineer needs LiveView experience, not React."
    ),
    5: (
        "The caching decision is Memcached. Original reasoning: simpler than "
        "Redis, multi-threaded, and the use case (fragment caching and hot query "
        "results) does not require Redis data structures or persistence. "
        "Redis was evaluated and not chosen. The Redis discussions later in the "
        "conversation were exploratory and did not formally change the decision."
    ),
    6: (
        "No in-process caching anywhere in the codebase. Reasoning: with multiple "
        "Fargate tasks, in-process caches diverge and cause stale-read bugs that "
        "only surface under load. Everything cacheable goes through Memcached, "
        "which provides a single consistent cache layer across all tasks."
    ),
    7: (
        "Browser users authenticate with passkeys (WebAuthn). The mobile API "
        "uses JWT authentication. These are different because WebAuthn requires "
        "a browser context and is not available in native mobile. JWTs are "
        "short-lived (15-minute TTL) with DB-stored refresh tokens."
    ),
    8: (
        "The developer is incorrect. Sessions live in the database (PostgreSQL), "
        "not in the cache. The reasoning: cache-backed sessions create stale "
        "state bugs. Postgres sessions are consistent across all Fargate tasks. "
        "Memcached holds fragment cache and rate limit counters only."
    ),
    9: (
        "Passwords were ruled out: credential storage liability and reset flow "
        "maintenance. Magic links were ruled out: delivery reliability problems. "
        "OAuth was ruled out: auth path dependent on third-party uptime. "
        "All three were explicitly rejected; passkeys (WebAuthn) was chosen."
    ),
    10: (
        "EKS: replaces ECS Fargate. Original reasoning was lack of K8s expertise; "
        "would need to re-evaluate the observability story, cost model, and "
        "team training requirements. "
        "Redis: replaces Memcached. Original reasoning was simplicity and lack "
        "of need for Redis data structures; would need to re-evaluate ops complexity. "
        "MySQL: replaces PostgreSQL. Original reasoning was team familiarity and "
        "mature tooling; would need to re-evaluate migrations, FTS approach, "
        "and any Postgres-specific SQL features in use."
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# 4.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _format_chat_log_as_system_prompt() -> str:
    """Pack all 120 messages into a single system prompt string for Agent A."""
    lines: list[str] = [
        "You are a helpful assistant. Below is the full conversation history.",
        "Use it to answer the user's question accurately and consistently.",
        "",
        "CONVERSATION HISTORY:",
        "=" * 60,
    ]
    for i, msg in enumerate(CHAT_LOG):
        role = msg["role"].upper()
        lines.append(f"[{i + 1}] {role}: {msg['content']}")
        lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


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
# 5.  CONFIG
# ──────────────────────────────────────────────────────────────────────────────

AGENT_MODEL = "gemma3:4b"
JUDGE_MODEL = "gemma3:4b"
CHROMA_PATH = "./chroma_db_benchmark_v4"

# ──────────────────────────────────────────────────────────────────────────────
# 6.  AGENT A — VANILLA
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 78)
print("  BENCHMARK V4  —  Vanilla vs Active Memory Pipeline")
print("=" * 78)
print(f"\n  Model : {AGENT_MODEL}")
print(f"  Judge : {JUDGE_MODEL}")
print(f"  Chat log: {len(CHAT_LOG)} messages  "
      f"(decisions: 1-15, noise at ~29-31/56-58/84-86, recall: 111-120)")
print(f"  Recall questions: {len(RECALL_QUESTIONS)}")
print()

SYSTEM_PROMPT_A = _format_chat_log_as_system_prompt()

_print_divider("=")
print("  PHASE 1 — Agent A (Vanilla): answering recall questions")
_print_divider("=")
print()

agent_a_responses: dict[int, str]   = {}
agent_a_latencies: dict[int, float] = {}

for q_num, question in RECALL_QUESTIONS.items():
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
# 7.  AGENT B — ACTIVE MEMORY: INGEST THEN RECALL
# ──────────────────────────────────────────────────────────────────────────────

_print_divider("=")
print("  PHASE 2 — Agent B (Active Memory): batch ingestion")
_print_divider("=")
print()

pipeline = Pipeline(
    model=AGENT_MODEL,
    chroma_path=CHROMA_PATH,
    max_recent_messages=5,
)

ingest_start = time.time()
for i, msg in enumerate(CHAT_LOG):
    pipeline.chat(msg["content"], skip_observer=True)
    if (i + 1) % 10 == 0:
        print(f"  Ingested {i + 1}/120 messages ...")
ingest_elapsed = time.time() - ingest_start
print(f"\n  Ingestion complete in {ingest_elapsed:.1f}s")
print(f"  {pipeline!r}")
print()

_print_divider("=")
print("  PHASE 3 — Agent B (Active Memory): answering recall questions")
_print_divider("=")
print()

agent_b_responses: dict[int, str]   = {}
agent_b_latencies: dict[int, float] = {}

for q_num, question in RECALL_QUESTIONS.items():
    print(f"  Q{q_num}: {_truncate(question, 60)}")
    t0 = time.time()
    response = pipeline.chat(question)
    latency  = time.time() - t0
    agent_b_responses[q_num] = response
    agent_b_latencies[q_num] = latency
    print(f"       -> {latency:.1f}s  ({len(response.split())} words)")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 8.  SCORING
# ──────────────────────────────────────────────────────────────────────────────

_print_divider("=")
print("  PHASE 4 — Judge scoring")
_print_divider("=")
print()

scores_a: dict[int, int] = {}
scores_b: dict[int, int] = {}

for q_num in RECALL_QUESTIONS:
    gt = GROUND_TRUTH[q_num]

    print(f"  Scoring Q{q_num} Agent A ...")
    scores_a[q_num] = _judge(RECALL_QUESTIONS[q_num], gt, agent_a_responses[q_num])

    print(f"  Scoring Q{q_num} Agent B ...")
    scores_b[q_num] = _judge(RECALL_QUESTIONS[q_num], gt, agent_b_responses[q_num])

    print(f"    A={scores_a[q_num]}  B={scores_b[q_num]}")

print()

# ──────────────────────────────────────────────────────────────────────────────
# 9.  RESULTS TABLE
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

for q_num in sorted(RECALL_QUESTIONS):
    q_text = f"Q{q_num}: " + _truncate(RECALL_QUESTIONS[q_num], COL_Q - 4)
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

n          = len(RECALL_QUESTIONS)
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
# 10.  WHAT ACTIVE MEMORY STORED
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
# 11.  FINAL BUCKET SUMMARY
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
