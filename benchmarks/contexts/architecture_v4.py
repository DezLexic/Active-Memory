# -*- coding: utf-8 -*-
"""
architecture_v4.py

Architecture decision conversation fixture for benchmarking recall accuracy.

Six key technology decisions are established in the first 7-8 pairs, then
buried under an extended implementation discussion that contains six noise
passages (fake reversals) designed to confuse a model that has lost track
of the original commitments.

Usage
-----
    from benchmarks.contexts.architecture_v4 import ArchitectureContext

    ctx = ArchitectureContext()           # full context (~125 pairs, ~18K tokens)
    ctx = ArchitectureContext(n_pairs=55) # original baseline (~55 pairs, ~7.5K tokens)
    ctx = ArchitectureContext(n_pairs=20) # light smoke test (~2K tokens)

    ctx.messages          -> list[dict]  # conversation pairs to ingest
    ctx.recall_questions  -> dict[int, str]
    ctx.ground_truth      -> dict[int, str]
    ctx.pair_count        -> int
    ctx.token_estimate    -> int         # rough (~4 chars/token)
    ctx.as_system_prompt()               # formats messages for vanilla agent
"""

from __future__ import annotations


class ArchitectureContext:
    """
    Architecture decision conversation fixture.

    Parameters
    ----------
    n_pairs : int | None
        Number of conversation pairs (user + assistant) to include.
        Decisions are always in the first 7-8 pairs.
        Larger values bury the decisions deeper, stressing vanilla agents.
        None (default) uses the full conversation.
    """

    # ── Conversation data ───────────────────────────────────────────────────────
    # Six decisions established in pairs 0-7 (messages 0-14), then buried under
    # ~117 pairs of implementation discussion containing 6 noise passages.
    #
    # Noise passages (fake reversals — no decision is formally changed):
    #   Noise 1: MySQL as Postgres alternative          (~pair 14)
    #   Noise 2: EKS/Kubernetes as Fargate alternative  (~pair 27)
    #   Noise 3: Redis as Memcached alternative         (~pair 41)
    #   Noise 4: Redis for session storage              (~pair 61)
    #   Noise 5: Ecto.Repo exception for ORM rule       (~pair 71)
    #   Noise 6: OAuth as passkey fallback              (~pair 78)
    # ──────────────────────────────────────────────────────────────────────────

    _CONVERSATION: list[dict] = [

        # ── Six decisions emerge naturally (pairs 0-7, messages 0-14) ──────────

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

        # ── Development discussion (pairs 8-13, messages 15-27) ─────────────────

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

        # ── NOISE 1: MySQL floated as alternative (~pair 14) ────────────────────

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
        {   # 29  ← SHARPENED: adds "compelling enough... decision backlog"
            "role": "assistant",
            "content": (
                "MySQL InnoDB is production-grade. The gap between MySQL 8 and "
                "Postgres has narrowed considerably. For a CRUD-heavy application "
                "at our scale the performance characteristics are broadly similar. "
                "The ops familiarity argument is real -- compelling enough that it "
                "deserves a formal evaluation. Let's put the database choice on the "
                "decision backlog for next sprint."
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
        {   # 31  ← SHARPENED: ends "decision is still open"
            "role": "assistant",
            "content": (
                "Worth verifying. The replication story is different -- Postgres "
                "streaming replication versus MySQL GTID-based -- but both work "
                "at our scale. If the ops team has significantly more depth in "
                "MySQL monitoring and incident response, that's a meaningful "
                "operational argument. The database decision is still open pending "
                "that evaluation."
            ),
        },

        # ── Development discussion (pairs 17-26, messages 32-53) ────────────────

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

        # ── NOISE 2: EKS/Kubernetes raised by new hire (~pair 27) ───────────────

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
        {   # 56  ← SHARPENED: adds "worth a formal comparison before we commit"
            "role": "assistant",
            "content": (
                "She has a point about observability. Fargate abstracts the node "
                "layer which means you lose direct access to node-level metrics, "
                "and some cluster tooling assumes node access. EKS with managed "
                "node groups has gotten significantly simpler to operate -- if "
                "she's bringing two years of EKS production experience, the "
                "operational complexity argument weakens considerably. "
                "This is worth a formal comparison before we commit anything in code."
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
        {   # 58  ← SHARPENED: ends "deployment platform should be treated as open"
            "role": "assistant",
            "content": (
                "The cost comparison does favor EKS at moderate scale. Spot "
                "node groups for non-critical workloads can reduce compute costs "
                "significantly. If she has the depth to own the cluster, the "
                "operational overhead argument becomes much weaker. "
                "The deployment platform decision should be treated as open "
                "until she delivers the formal comparison."
            ),
        },

        # ── Development discussion (pairs 30-41, messages 59-83) ────────────────

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

        # ── NOISE 3: Redis as caching alternative (~pair 41) ────────────────────

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
        {   # 84  ← SHARPENED: adds "worth revisiting... put on the backlog"
            "role": "assistant",
            "content": (
                "The consolidation argument is real. If Redis is already in the "
                "stack for PubSub, the case for Memcached weakens -- you're "
                "paying for operational complexity of two systems instead of one. "
                "Redis sorted sets would make sliding-window rate limiting more "
                "elegant. The caching choice is worth revisiting before we "
                "finalize the infrastructure. Put it on the backlog."
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
        {   # 86  ← SHARPENED: ends "caching choice should be treated as under evaluation"
            "role": "assistant",
            "content": (
                "Redis has become the default caching layer for a reason. "
                "The operational gap between Redis Cluster and Memcached has "
                "narrowed. If Redis is already present in the stack, its "
                "feature set makes a lot of other problems easier to solve. "
                "The caching choice should be treated as under evaluation -- "
                "we haven't formally committed to Memcached long-term."
            ),
        },

        # ── Development discussion (pairs 44-54, messages 87-109) ───────────────

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
                "you can deploy the code without activating the feature -- dark "
                "deploy then controlled rollout. No Friday deploys is a good "
                "policy for a small team without 24/7 on-call coverage."
            ),
        },
        {   # 101
            "role": "user",
            "content": (
                "Project archiving: archived projects are soft-deleted -- "
                "is_archived boolean, archived_at timestamp. Archived projects "
                "are excluded from default list queries via a is_archived = false "
                "predicate. Hard delete is available but requires workspace owner "
                "confirmation."
            ),
        },
        {   # 102
            "role": "assistant",
            "content": (
                "Soft delete via is_archived is simpler than a separate "
                "archived_projects table for this use case. The is_archived "
                "predicate needs to be on an index alongside project_id wherever "
                "it appears in queries -- a partial index WHERE is_archived = false "
                "might be more efficient."
            ),
        },
        {   # 103
            "role": "user",
            "content": (
                "Recurring tasks: a recurring_configs table that stores the "
                "recurrence rule (RRULE string), last_generated_at, and template "
                "task data. An Oban job runs daily, finds configs where next "
                "occurrence is due, generates the task from the template."
            ),
        },
        {   # 104
            "role": "assistant",
            "content": (
                "RRULE string storage is flexible and well-understood. An Elixir "
                "RRULE parser library (ex_rrule or similar) handles next occurrence "
                "calculation. Daily Oban job with a lookback window handles missed "
                "runs if the job fails. Index on last_generated_at for the job query."
            ),
        },
        {   # 105
            "role": "user",
            "content": (
                "Webhook support: workspaces can configure webhooks for events "
                "like task.created, task.updated, member.added. Webhook deliveries "
                "are queued via Oban, delivered with a signed HMAC-SHA256 payload. "
                "Delivery log stored for 7 days for debugging."
            ),
        },
        {   # 106
            "role": "assistant",
            "content": (
                "Oban-backed webhook delivery with retry and exponential backoff "
                "is correct. HMAC-SHA256 signing lets receivers verify the payload. "
                "Delivery log in Postgres with a 7-day retention gives debugging "
                "visibility without unbounded growth. Store attempt count and "
                "last_error on the delivery record."
            ),
        },
        {   # 107
            "role": "user",
            "content": (
                "Guest access: workspaces can make individual projects publicly "
                "readable without authentication. A project-level is_public flag. "
                "Public project views are server-rendered LiveView pages with no "
                "auth requirement. Write operations still require auth."
            ),
        },
        {   # 108
            "role": "assistant",
            "content": (
                "is_public flag on the project record keeps it simple. LiveView "
                "mount checks the flag and skips the auth redirect for public "
                "projects. The auth plug on write paths still fires regardless "
                "of the public flag -- the public flag is a read permission, "
                "not a write permission."
            ),
        },
        {   # 109
            "role": "user",
            "content": (
                "That covers the main architecture. We've gone deep on a lot of "
                "implementation details. Before we move to sprint planning I want "
                "to do a consistency check -- make sure we haven't drifted from "
                "the foundation decisions we started with, given how much "
                "discussion has happened since."
            ),
        },

        # ═══════════════════════════════════════════════════════════════════════
        # EXTENDED SECTION  (pairs 55-124, messages 110-249)
        # Adds ~140 messages of realistic implementation discussion.
        # Buries the original 6 decisions further in the context.
        # Contains 3 more noise passages (Noises 4-6).
        # ═══════════════════════════════════════════════════════════════════════

        {   # 110
            "role": "assistant",
            "content": (
                "Good checkpoint. The core decisions -- Phoenix LiveView, Elixir, "
                "Postgres on RDS, raw SQL, passkeys, Fargate, Memcached, "
                "sessions in Postgres -- were all confirmed in the first session. "
                "Several are marked for formal evaluation. "
                "What's the first sprint scope?"
            ),
        },
        {   # 111
            "role": "user",
            "content": (
                "Sprint one: authentication (passkey register and authenticate), "
                "workspace and project CRUD, and the basic task list LiveView. "
                "No notifications, no file uploads, no billing in sprint one. "
                "Get the core data model and auth working end to end."
            ),
        },

        # ── PR process and branching ─────────────────────────────────────────

        {   # 112
            "role": "assistant",
            "content": (
                "Scope is right for sprint one. Authentication is the highest-risk "
                "piece and should unblock everything else. Basic task list first "
                "means the data model is stress-tested early. "
                "What's the PR review policy?"
            ),
        },
        {   # 113
            "role": "user",
            "content": (
                "One approval required before merge. The author picks a reviewer. "
                "For changes touching auth, database schema, or billing: "
                "two approvals required. No self-merge. "
                "Draft PRs are encouraged for early feedback on approach."
            ),
        },
        {   # 114
            "role": "assistant",
            "content": (
                "Two-approval gate on auth and schema changes is the right "
                "elevated threshold. Self-merge ban is straightforward to enforce "
                "with branch protection. Draft PRs for early design feedback "
                "reduces wasted implementation work."
            ),
        },
        {   # 115
            "role": "user",
            "content": (
                "Branching: feature branches off main, named feature/short-description. "
                "Hotfix branches named hotfix/ticket-id. "
                "Squash merge to main to keep history linear. "
                "Tags on main for every release, format v1.2.3."
            ),
        },

        # ── Staging and local dev ────────────────────────────────────────────

        {   # 116
            "role": "assistant",
            "content": (
                "Squash merge keeps the main branch history readable. "
                "Semantic versioning tags on main make the release log trivial. "
                "Hotfix branches off main with cherry-pick back if needed. "
                "What does the staging environment look like?"
            ),
        },
        {   # 117
            "role": "user",
            "content": (
                "Staging is a single ECS task pointing at an RDS instance "
                "with sanitised production data loaded monthly. "
                "Same Docker image as production, different environment variables. "
                "Staging deploys happen automatically on merge to main. "
                "Production deploys are manual trigger."
            ),
        },
        {   # 118
            "role": "assistant",
            "content": (
                "Automatic staging deploy on merge keeps staging fresh. "
                "Manual production trigger is correct -- staging is the canary, "
                "production promote is a deliberate act. Sanitised production "
                "data monthly gives realistic test conditions without PII exposure."
            ),
        },
        {   # 119
            "role": "user",
            "content": (
                "Local dev: docker compose for Postgres and Memcached. "
                "Developers run the Phoenix app natively on their machine "
                "against local services. mix ecto.setup seeds the database "
                "from a priv/repo/seeds.exs file. Seed data covers all "
                "workspace roles and task states."
            ),
        },

        # ── NOISE 4: Redis session store (~pair 61) ──────────────────────────

        {   # 120
            "role": "assistant",
            "content": (
                "Local compose for dependencies is the right choice -- native "
                "Phoenix with local services is faster to iterate than running "
                "everything in containers locally. "
                "Seed data covering all roles and states catches permission "
                "bugs early."
            ),
        },
        {   # 121
            "role": "user",
            "content": (
                "One more thing on sessions. An engineer is pointing out that "
                "Redis has native TTL management and could simplify the session "
                "cleanup logic. Moving sessions from Postgres to a Redis store "
                "would reduce read load on the primary and give us native expiry "
                "without a separate cleanup job. The ops cost would be minimal "
                "if Redis is already in the stack."
            ),
        },
        {   # 122  ← NOISE 4 response
            "role": "assistant",
            "content": (
                "The operational simplicity argument is real. Redis TTL is native "
                "and doesn't require a cleanup job. Session read performance "
                "would improve significantly with in-memory storage. "
                "If Redis is already in the stack for PubSub and caching, adding "
                "sessions doesn't materially increase the operational surface. "
                "The session storage decision could reasonably go either way -- "
                "worth adding to the review backlog."
            ),
        },
        {   # 123
            "role": "user",
            "content": (
                "I'll add it to the list. We have a few open evaluations now. "
                "I want to make sure we actually resolve them before too much "
                "code is written against the provisional choices."
            ),
        },

        # ── API error shape and versioning ───────────────────────────────────

        {   # 124
            "role": "assistant",
            "content": (
                "The evaluation backlog is worth tracking explicitly. "
                "None of the open items block sprint one -- auth and data model "
                "are not affected. What's the API error response shape?"
            ),
        },
        {   # 125
            "role": "user",
            "content": (
                "JSON error envelope: { error: { code: string, message: string, "
                "trace_id: string, details: object | null } }. "
                "The code field is a machine-readable string like "
                "VALIDATION_FAILED or RESOURCE_NOT_FOUND. "
                "HTTP status maps to the error category, not to each specific code."
            ),
        },
        {   # 126
            "role": "assistant",
            "content": (
                "Machine-readable error codes let API clients branch on specific "
                "conditions without parsing human messages. trace_id ties the "
                "client-visible error to the server log. details object carries "
                "field-level validation errors without polluting the top-level shape."
            ),
        },
        {   # 127
            "role": "user",
            "content": (
                "HTTP status conventions: 400 for validation failures, "
                "401 for unauthenticated, 403 for unauthorized, "
                "404 for not found, 409 for conflicts, 422 for business logic "
                "rejections, 429 for rate limit, 500 for unexpected server errors. "
                "No 200 with an error body."
            ),
        },
        {   # 128
            "role": "assistant",
            "content": (
                "422 for business logic rejections is a good separation from "
                "400 validation failures. No 200 with an error body is a "
                "hard rule worth enforcing in a shared response helper -- "
                "it prevents the anti-pattern from creeping in under deadline."
            ),
        },
        {   # 129
            "role": "user",
            "content": (
                "API versioning: /api/v1 is the first version. Breaking changes "
                "require a new version prefix. Non-breaking additions are "
                "backward-compatible and can land in the current version. "
                "Deprecation policy: six months notice in response headers "
                "before removing a version."
            ),
        },

        # ── Health check and load testing ────────────────────────────────────

        {   # 130
            "role": "assistant",
            "content": (
                "Six months deprecation notice with response headers is a "
                "reasonable consumer-friendly policy. Sunset header (RFC 8594) "
                "is the standard mechanism. v1 should be stable for at least "
                "two years before any sunset consideration."
            ),
        },
        {   # 131
            "role": "user",
            "content": (
                "Health check endpoint at /health. Returns 200 with "
                "{ status: ok, db: ok|degraded, cache: ok|degraded }. "
                "Checks Postgres connectivity with a lightweight query and "
                "Memcached with a get/set roundtrip. "
                "ALB health check polls /health every 30 seconds."
            ),
        },
        {   # 132
            "role": "assistant",
            "content": (
                "Shallow health check for the ALB is correct -- a deep check "
                "that makes real queries can fail under load and take the task "
                "out of rotation when it's actually fine. "
                "Degraded status in the response body lets you distinguish "
                "a healthy task with a sick dependency from a sick task."
            ),
        },
        {   # 133
            "role": "user",
            "content": (
                "Load testing with k6. Baseline test: 100 concurrent users, "
                "10 minutes, mixed read/write workload. Performance targets: "
                "p95 response time under 500ms for task list, "
                "p95 under 200ms for task updates. "
                "Run against staging before each production deploy."
            ),
        },
        {   # 134
            "role": "assistant",
            "content": (
                "k6 is the right tool -- scriptable, CI-friendly, produces "
                "structured output. 100 concurrent users for 10 minutes is a "
                "realistic synthetic load for an early product. "
                "500ms p95 for list views is conservative enough to catch "
                "query regressions before they reach production."
            ),
        },
        {   # 135
            "role": "user",
            "content": (
                "Connection pool sizing: Postgrex pool size 10 per task, "
                "three tasks minimum, so 30 max connections to Postgres. "
                "RDS instance sized to handle 100 max connections with headroom. "
                "Pool timeout 5 seconds -- fail fast rather than pile up waiters."
            ),
        },

        # ── NOISE 5: Ecto.Repo exception proposal (~pair 71) ─────────────────

        {   # 136
            "role": "assistant",
            "content": (
                "30 connections across three tasks is conservative and correct. "
                "Pool timeout of 5 seconds under load surfaces connection "
                "exhaustion quickly rather than letting requests queue silently. "
                "Monitor pool checkout time in telemetry."
            ),
        },
        {   # 137
            "role": "user",
            "content": (
                "The contractor we're onboarding for the analytics sprint wants "
                "an exception to the no-ORM rule. He wants to use Ecto.Repo for "
                "a complex aggregation query -- says it reduces 80 lines of raw "
                "SQL to 15 lines with Ecto's composability. He has five years of "
                "Ecto experience and argues the query is actually more readable."
            ),
        },
        {   # 138  ← NOISE 5 response
            "role": "assistant",
            "content": (
                "The composability argument is strongest for complex multi-join "
                "aggregations. 80 lines of raw SQL for a reporting query is "
                "legitimately harder to maintain than a well-structured Ecto "
                "pipeline. The no-ORM rule was primarily about avoiding obscured "
                "execution paths -- a well-understood Ecto query by someone fluent "
                "in it isn't necessarily more opaque. "
                "Worth a formal exception policy discussion. Put it on the backlog."
            ),
        },
        {   # 139
            "role": "user",
            "content": (
                "Noted. I'll flag it for the next architecture sync. "
                "For now he'll write it in raw SQL and we can revisit "
                "once we see the actual query."
            ),
        },

        # ── Security scanning and incident response ──────────────────────────

        {   # 140
            "role": "assistant",
            "content": (
                "Seeing the actual query first is the right approach. "
                "If the SQL is genuinely unmaintainable, the exception case "
                "is stronger. If it's readable, the exception isn't needed. "
                "What about security scanning in CI?"
            ),
        },
        {   # 141
            "role": "user",
            "content": (
                "mix audit in CI to check Hex dependencies against the "
                "vulnerability database. Trivy for Docker image scanning "
                "on every build -- base image CVEs, OS packages, bundled deps. "
                "Fail the build on critical severity. Warn on high."
            ),
        },
        {   # 142
            "role": "assistant",
            "content": (
                "mix audit and Trivy together cover the two main surfaces. "
                "Failing on critical is the right default -- a known critical "
                "vulnerability in a base image shouldn't ship. "
                "Weekly Dependabot PRs keep the dependency drift manageable."
            ),
        },
        {   # 143
            "role": "user",
            "content": (
                "OWASP Top 10 review before launch. Specifically: injection "
                "(parameterized SQL covers this), broken auth (passkeys help), "
                "XSS (Earmark + HtmlSanitizeEx), CSRF (Phoenix built-in). "
                "Security review checklist in the pre-launch runbook."
            ),
        },
        {   # 144
            "role": "assistant",
            "content": (
                "Parameterized SQL eliminates injection at the ORM layer. "
                "Phoenix LiveView has CSRF protection built in. "
                "Earmark plus sanitizer covers stored XSS. "
                "The main residual risk is business logic access control -- "
                "that's where the Policies module review should focus."
            ),
        },
        {   # 145
            "role": "user",
            "content": (
                "Incident response: P1 is full outage or data loss risk. "
                "P2 is degraded service affecting majority of users. "
                "P3 is isolated degradation. "
                "P1 and P2 trigger PagerDuty immediately. "
                "Incident commander is the on-call engineer. "
                "Post-mortem required for every P1 and P2 within 48 hours."
            ),
        },
        {   # 146
            "role": "assistant",
            "content": (
                "48-hour post-mortem window is tight enough to stay fresh "
                "but not so immediate it blocks recovery. Blameless post-mortems "
                "with a 5-why root cause is standard. "
                "On-call rotation: two-week rotations, one primary and one secondary."
            ),
        },

        # ── NOISE 6: OAuth fallback for enterprise (~pair 78) ────────────────

        {   # 147
            "role": "user",
            "content": (
                "We have a potential enterprise customer with 500 seats who won't "
                "sign unless we support OAuth with Google Workspace. Their IT "
                "security team requires federated identity and won't allow passkeys "
                "as the sole auth method. Adding OAuth as a configurable enterprise "
                "auth option -- is that viable alongside the passkey flow?"
            ),
        },
        {   # 148  ← NOISE 6 response
            "role": "assistant",
            "content": (
                "For enterprise deals of that size, OAuth or SAML is a standard "
                "ask. The technical path is adding an OAuth provider plug to the "
                "Phoenix pipeline alongside the passkey flow -- they're not "
                "mutually exclusive at the route level. The original passkey "
                "decision was about avoiding OAuth dependency for the core product, "
                "but a configurable enterprise auth path is a different scope. "
                "This is worth serious evaluation. Let's schedule a decision meeting."
            ),
        },
        {   # 149
            "role": "user",
            "content": (
                "That could be a significant revenue decision. I'll flag it "
                "for the founders. For now we proceed with passkeys only "
                "and treat the enterprise auth question as an open item."
            ),
        },

        # ── MySQL follow-up: still unresolved (~pair 80) ─────────────────────

        {   # 150
            "role": "assistant",
            "content": (
                "Proceeding with passkeys while keeping the enterprise auth "
                "question open is the right call. Nothing in sprint one is "
                "affected by the eventual OAuth decision. "
                "Any update on the MySQL evaluation from the ops team?"
            ),
        },
        {   # 151
            "role": "user",
            "content": (
                "Still on the backlog. The ops team lead hasn't come back "
                "with the formal cost estimate for their tooling rebuild. "
                "We haven't formally resolved it. The Postgres work "
                "is proceeding in the meantime."
            ),
        },

        # ── Billing integration ───────────────────────────────────────────────

        {   # 152
            "role": "assistant",
            "content": (
                "Postgres proceeding by default is fine. If the evaluation "
                "comes back and changes the decision, migration cost grows "
                "the longer we wait, so the ops team has an incentive "
                "to close it out. Let's cover billing integration."
            ),
        },
        {   # 153
            "role": "user",
            "content": (
                "Stripe for billing. Free tier has no payment details. "
                "Pro upgrade: Stripe Checkout flow, webhook events update "
                "the workspace plan on payment success or failure. "
                "We never store card numbers -- Stripe handles all PCI scope. "
                "Stripe customer ID stored on the workspace record."
            ),
        },
        {   # 154
            "role": "assistant",
            "content": (
                "Stripe Checkout offloads the PCI compliance surface entirely. "
                "Webhook events for plan changes via Oban with idempotency "
                "on the Stripe event ID prevents duplicate plan upgrades "
                "from retried webhooks. Stripe customer ID is the only "
                "Stripe-specific data we own."
            ),
        },
        {   # 155
            "role": "user",
            "content": (
                "Webhook idempotency: stripe_events table with the event ID "
                "as a unique key. Process event, upsert to stripe_events in "
                "the same transaction. On duplicate key error, discard the "
                "event -- it's already been processed."
            ),
        },
        {   # 156
            "role": "assistant",
            "content": (
                "Idempotency via unique constraint on the event ID is clean "
                "and race-condition safe under concurrent webhook deliveries. "
                "The upsert-in-transaction pattern ensures exactly-once processing "
                "without a distributed lock. Index on event ID for fast lookup."
            ),
        },
        {   # 157
            "role": "user",
            "content": (
                "Plan enforcement: a Plug checks the workspace plan on every "
                "request that touches a plan-limited resource. Plan limit "
                "violations return a 402 Payment Required with an upgrade prompt. "
                "Checks are synchronous -- no async lag between payment and access."
            ),
        },

        # ── GDPR and data retention ───────────────────────────────────────────

        {   # 158
            "role": "assistant",
            "content": (
                "Synchronous plan check on every request is correct for a "
                "small scale. At higher scale a cached plan check (Memcached, "
                "short TTL) could reduce Postgres reads, but the correctness "
                "risk of stale plan data isn't worth it at launch."
            ),
        },
        {   # 159
            "role": "user",
            "content": (
                "GDPR: we're EU-accessible so we need the basics covered. "
                "Privacy policy and cookie notice on signup. "
                "Right to access: workspace owners can download their data. "
                "Right to erasure: account deletion anonymizes user records "
                "rather than hard-deleting, to preserve referential integrity "
                "in audit logs."
            ),
        },
        {   # 160
            "role": "assistant",
            "content": (
                "Anonymization over hard-delete is the pragmatic choice when "
                "audit logs reference the user. Replace name and email with "
                "anonymized placeholders, nullify optional fields, keep the "
                "user_id in place so foreign keys don't break. "
                "GDPR erasure requires removing PII, not the record itself."
            ),
        },
        {   # 161
            "role": "user",
            "content": (
                "Data retention policy: application logs purged after 90 days. "
                "Audit events retained for 2 years. Soft-deleted entities "
                "hard-deleted after 30 days via an Oban cleanup job. "
                "Stripe event log retained for 7 years per financial record "
                "requirements."
            ),
        },

        # ── Structured logging ────────────────────────────────────────────────

        {   # 162
            "role": "assistant",
            "content": (
                "90-day log retention is standard. 2-year audit retention is "
                "reasonable for an early product -- revisit as the compliance "
                "requirements clarify. 30-day grace period before hard-delete "
                "is enough for accidental deletion recovery without "
                "unbounded data accumulation."
            ),
        },
        {   # 163
            "role": "user",
            "content": (
                "Structured logging: JSON format for all application logs. "
                "Every log entry includes timestamp, level, request_id, "
                "user_id if authenticated, and the message. "
                "No PII in log messages -- no names, emails, or payload content. "
                "Log level info in staging and production, debug in local dev."
            ),
        },
        {   # 164
            "role": "assistant",
            "content": (
                "JSON logs are machine-parseable from day one, which matters "
                "when you start shipping to a log aggregator. request_id "
                "correlation across a single request is essential for debugging. "
                "No PII in logs is a hard rule -- add it to the code review "
                "checklist."
            ),
        },
        {   # 165
            "role": "user",
            "content": (
                "Log aggregation: CloudWatch Logs. ECS Fargate sends container "
                "stdout to CloudWatch via the awslogs driver. "
                "Log Insights queries for ad-hoc investigation. "
                "CloudWatch Alarms on error rate and p95 latency. "
                "No third-party log aggregator at launch -- cost and complexity."
            ),
        },

        # ── Alerting and observability ────────────────────────────────────────

        {   # 166
            "role": "assistant",
            "content": (
                "CloudWatch Logs via awslogs is zero-config for ECS Fargate. "
                "Log Insights is powerful enough for most investigations. "
                "Adding a third-party aggregator later is straightforward -- "
                "just change the log driver. No Datadog or Splunk at launch "
                "is the right cost decision."
            ),
        },
        {   # 167
            "role": "user",
            "content": (
                "Alerting thresholds: 5xx error rate above 1% over 5 minutes "
                "triggers P2. Database connection pool exhaustion above 80% "
                "for 2 minutes triggers P2. Task queue depth above 1000 "
                "for 5 minutes triggers P3. All thresholds reviewed quarterly."
            ),
        },
        {   # 168
            "role": "assistant",
            "content": (
                "1% 5xx rate over 5 minutes is a tight threshold and correct -- "
                "normal traffic shouldn't produce server errors at that rate. "
                "Pool exhaustion at 80% is an early warning before 100% blocks "
                "requests. Queue depth alert catches a stalled worker before "
                "it becomes a user-visible problem."
            ),
        },
        {   # 169
            "role": "user",
            "content": (
                "EKS comparison: any update from the DevOps engineer on the "
                "Fargate vs EKS formal comparison she was putting together? "
                "The sprint one work doesn't depend on this but I want to "
                "close the decision before we write too much "
                "infrastructure code against Fargate."
            ),
        },
        {   # 170
            "role": "assistant",
            "content": (
                "Not yet -- the comparison hasn't been formally presented. "
                "The Fargate task definitions and ECS configuration are what "
                "we're building against for now. The deployment platform "
                "decision is still open until she delivers the comparison. "
                "Worth setting a deadline on that."
            ),
        },

        # ── Documentation and ADRs ────────────────────────────────────────────

        {   # 171
            "role": "user",
            "content": (
                "Architecture Decision Records: we'll write an ADR for each "
                "major decision in a /docs/adr directory. Format: "
                "title, status, context, decision, consequences. "
                "Status can be proposed, accepted, deprecated, or superseded. "
                "ADR is required for any change to a foundational decision."
            ),
        },
        {   # 172
            "role": "assistant",
            "content": (
                "ADRs in /docs/adr with lightweight format is right. "
                "The status field carries the lifecycle -- a superseded ADR "
                "links to its replacement. Requiring an ADR for foundational "
                "changes creates a natural friction point that catches "
                "accidental drift."
            ),
        },
        {   # 173
            "role": "user",
            "content": (
                "Team conventions doc: a CONVENTIONS.md at the repo root. "
                "Covers module naming, function naming, test naming, error "
                "handling patterns, commit message format, and the "
                "no-raw-error-to-client rule. "
                "New contributors read this before their first PR."
            ),
        },
        {   # 174
            "role": "assistant",
            "content": (
                "A single CONVENTIONS.md is the right scope for an early team. "
                "Commit message format matters for a readable git log -- "
                "conventional commits (feat:, fix:, chore:) are a good "
                "lightweight standard. The error handling pattern should "
                "be explicit about the {:ok, result} | {:error, reason} convention."
            ),
        },
        {   # 175
            "role": "user",
            "content": (
                "Redis evaluation update: where did we land on the caching "
                "question? I know we put Memcached vs Redis on the backlog "
                "after the discussion a while back. Have we made a call?"
            ),
        },
        {   # 176
            "role": "assistant",
            "content": (
                "Still open. The Memcached vs Redis evaluation hasn't been "
                "formally resolved. We've been building against Memcached "
                "as the default, but the caching choice is on the backlog "
                "for a formal decision. The same applies to the session "
                "storage review."
            ),
        },

        # ── Performance and capacity ──────────────────────────────────────────

        {   # 177
            "role": "user",
            "content": (
                "Capacity planning: launch target is 100 active workspaces, "
                "500 concurrent users at peak. RDS instance: db.t4g.medium "
                "for launch, upgrade to db.m6g.large at 500 workspaces. "
                "ECS task size: 0.5 vCPU, 1GB memory per task, "
                "auto-scale up to 10 tasks."
            ),
        },
        {   # 178
            "role": "assistant",
            "content": (
                "db.t4g.medium is cost-effective for 100 workspaces but has "
                "CPU burst limits that will bite at sustained load. "
                "Monitor CPU credit balance in CloudWatch. "
                "0.5 vCPU per task means the BEAM scheduler won't saturate "
                "it at low concurrency -- appropriate for launch."
            ),
        },
        {   # 179
            "role": "user",
            "content": (
                "Database index review schedule: quarterly index analysis "
                "using pg_stat_user_indexes to find unused indexes and "
                "slow queries from pg_stat_statements. "
                "New indexes require an explain analyze in the PR description. "
                "Concurrent index creation only -- no table locks in production."
            ),
        },
        {   # 180
            "role": "assistant",
            "content": (
                "pg_stat_statements is essential for query performance work. "
                "Requiring explain analyze in PRs catches bad query plans "
                "before they reach production. CONCURRENTLY for all index "
                "creates in migration up blocks is a good standing rule "
                "for the migration review checklist."
            ),
        },

        # ── Blue/green and rollback ───────────────────────────────────────────

        {   # 181
            "role": "user",
            "content": (
                "Rollback strategy: migrations must be backward-compatible. "
                "If a migration is not backward-compatible, the deploy is "
                "a two-phase deploy: old code plus new schema, then new "
                "code after the old code is retired. No direct breaking "
                "schema changes in a single deploy."
            ),
        },
        {   # 182
            "role": "assistant",
            "content": (
                "Expand-then-contract is the pattern -- add the new column, "
                "deploy code that works with both old and new schema, "
                "then remove the old column in a follow-up. "
                "This is a discipline requirement, not a technical one, "
                "so it lives in the migration review checklist."
            ),
        },
        {   # 183
            "role": "user",
            "content": (
                "Post-deploy verification: automated smoke test suite in CI "
                "runs against staging after each deploy. Tests: login flow, "
                "workspace creation, task creation, API health check. "
                "If any smoke test fails, the on-call engineer is paged. "
                "Production deploys are gated on staging smoke tests passing."
            ),
        },
        {   # 184
            "role": "assistant",
            "content": (
                "Smoke tests as a deploy gate is the right pattern. "
                "The four scenarios cover all three layers: auth, data, API. "
                "Paging on smoke failure rather than waiting for user reports "
                "catches regressions within minutes of deploy. "
                "Keep smoke tests fast -- under 2 minutes for the full suite."
            ),
        },

        # ── CDN and static assets ─────────────────────────────────────────────

        {   # 185
            "role": "user",
            "content": (
                "Static assets: compiled by the Phoenix asset pipeline "
                "and uploaded to S3 on deploy. CloudFront serves the "
                "S3 bucket with a long cache TTL -- assets are "
                "content-hash-named so they're immutable. "
                "No LiveView JavaScript bloat -- minimal custom JS."
            ),
        },
        {   # 186
            "role": "assistant",
            "content": (
                "Content-hash asset names plus long CloudFront TTL is the "
                "standard pattern. Old assets stay cached in browsers until "
                "the hash changes. Phoenix's asset pipeline generates the "
                "hashed filenames automatically. No purge required on deploy."
            ),
        },
        {   # 187
            "role": "user",
            "content": (
                "Image optimization: user avatars and project covers are "
                "resized on upload via an Oban job. Source stored in S3, "
                "variants generated at 32px, 64px, and 128px. "
                "Served via CloudFront with the appropriate Cache-Control headers. "
                "No third-party image optimization service."
            ),
        },
        {   # 188
            "role": "assistant",
            "content": (
                "On-upload resize via Oban keeps the critical upload path fast -- "
                "the upload succeeds immediately and resize happens asynchronously. "
                "Three fixed variants covers every UI context without on-the-fly "
                "resizing. Mogrify or Image wrapping ImageMagick handles the resize."
            ),
        },

        # ── LiveView presence and WebSocket ──────────────────────────────────

        {   # 189
            "role": "user",
            "content": (
                "Phoenix Presence for collaborative indicators: show which users "
                "are viewing the same project board. Presence tracks connected "
                "users on the project:{id} topic. Avatars appear in the "
                "board header when others are present. "
                "No real-time cursor tracking at launch."
            ),
        },
        {   # 190
            "role": "assistant",
            "content": (
                "Phoenix Presence uses CRDT-based conflict resolution across "
                "the cluster and integrates with PubSub. The presence_diff event "
                "handles join and leave in the LiveView handle_info callback. "
                "Presence state is eventually consistent, which is fine for "
                "avatar indicators."
            ),
        },
        {   # 191
            "role": "user",
            "content": (
                "WebSocket connection management: LiveView connections re-establish "
                "automatically on disconnect. No manual reconnect logic needed. "
                "The LiveView lifecycle handles mount and reconnect state reloading. "
                "Users get a brief reconnecting indicator if the socket drops."
            ),
        },
        {   # 192
            "role": "assistant",
            "content": (
                "LiveView's built-in reconnect is one of its strong points. "
                "The client automatically re-establishes the WebSocket and "
                "triggers a fresh mount. The reconnecting indicator is built "
                "into the Phoenix LiveView JS client. No custom reconnect "
                "logic is needed."
            ),
        },
        {   # 193
            "role": "user",
            "content": (
                "Optimistic UI updates: task status changes apply immediately "
                "in the LiveView assigns before the server confirms. "
                "On error, the assigns are reverted. "
                "Reduces perceived latency on frequent task updates "
                "without complex client-side state management."
            ),
        },
        {   # 194
            "role": "assistant",
            "content": (
                "Optimistic updates in LiveView assigns are straightforward -- "
                "update the local state in handle_event before the DB call, "
                "then revert on error. The server is still the source of truth. "
                "This pattern is appropriate for frequent low-stakes updates "
                "like status changes."
            ),
        },

        # ── Final sprint planning and launch ─────────────────────────────────

        {   # 195
            "role": "user",
            "content": (
                "MVP scope: authentication, workspace + project + task CRUD, "
                "task assignment and status updates, in-app notifications, "
                "basic file attachments, billing integration for Pro tier. "
                "Anything beyond that is post-launch. "
                "Target: internal beta in 8 weeks."
            ),
        },
        {   # 196
            "role": "assistant",
            "content": (
                "Eight weeks to internal beta is tight but achievable with "
                "sprint one scoped to auth and core data model. "
                "Billing in MVP is the right call -- it de-risks revenue "
                "before public launch. "
                "What are the launch criteria?"
            ),
        },
        {   # 197
            "role": "user",
            "content": (
                "Launch criteria: all P1 bugs resolved, load test passing "
                "at target thresholds, security review complete, "
                "GDPR basics in place, Stripe billing working end-to-end, "
                "rollback plan tested at least once in staging. "
                "No feature completeness criteria -- we ship with the MVP scope."
            ),
        },
        {   # 198
            "role": "assistant",
            "content": (
                "Quality-based launch criteria rather than feature-complete "
                "criteria is the right call. Untested rollback is a risk that "
                "bites exactly when you need it most. "
                "GDPR and security review as hard gates prevents a compliance "
                "scramble after users are onboard."
            ),
        },
        {   # 199
            "role": "user",
            "content": (
                "Post-launch monitoring plan: daily active user tracking, "
                "error rate dashboard checked every morning, "
                "weekly load test against production with synthetic traffic, "
                "monthly review of slow query log. "
                "First 30 days: on-call coverage 7 days a week."
            ),
        },
        {   # 200
            "role": "assistant",
            "content": (
                "Morning error rate check is a good lightweight daily ritual. "
                "Weekly synthetic load test catches performance regressions "
                "before real traffic amplifies them. "
                "30-day heightened on-call coverage for launch is prudent -- "
                "the surprise rate is highest in the first month."
            ),
        },
        {   # 201
            "role": "user",
            "content": (
                "Team communication: architecture decisions posted to a "
                "#architecture Slack channel with a link to the ADR. "
                "Breaking changes flagged in the PR description with "
                "BREAKING CHANGE in the title. "
                "Weekly team sync to review the decision backlog."
            ),
        },
        {   # 202
            "role": "assistant",
            "content": (
                "Weekly decision backlog review is the mechanism to actually "
                "close the open evaluations -- MySQL, EKS, Redis, OAuth, "
                "Ecto exception. Without a structured review, open items "
                "accumulate and code gets written against provisional choices "
                "indefinitely."
            ),
        },
        {   # 203
            "role": "user",
            "content": (
                "Database vacuum settings: autovacuum is on with default settings "
                "at launch. Monitor dead tuple accumulation on the tasks and "
                "audit_events tables -- these will have the highest write volume. "
                "Tune autovacuum_vacuum_scale_factor if dead tuple percentage "
                "exceeds 10% in pg_stat_user_tables."
            ),
        },
        {   # 204
            "role": "assistant",
            "content": (
                "Default autovacuum is fine at launch volume. "
                "The scale factor default of 0.2 means vacuum triggers at "
                "20% dead tuples, which is conservative. For high-write "
                "tables like audit_events, a lower scale_factor of 0.05 "
                "keeps bloat tighter. Add to the quarterly index review."
            ),
        },
        {   # 205
            "role": "user",
            "content": (
                "That covers everything we need before sprint planning. "
                "We've made a lot of decisions today and a few are still "
                "marked open. I want to make sure we close the evaluation "
                "backlog in the first two weeks before those decisions "
                "affect any code we write."
            ),
        },
        {   # 206
            "role": "assistant",
            "content": (
                "Agreed. The open items -- MySQL vs Postgres, Fargate vs EKS, "
                "Memcached vs Redis, Redis sessions, Ecto exception policy, "
                "OAuth for enterprise -- should all have a formal resolution "
                "by the end of week two. Until then, we build against the "
                "provisional choices: Postgres, Fargate, Memcached, "
                "Postgres sessions, raw SQL, passkeys only."
            ),
        },
        {   # 207
            "role": "user",
            "content": (
                "Agreed. Everything else looks solid. The stack is coherent, "
                "the constraints are clear, and the team knows what to build "
                "in sprint one. Let's lock the sprint plan and start."
            ),
        },
        {   # 208
            "role": "assistant",
            "content": (
                "Sprint one is locked: passkey auth end to end, workspace "
                "and project CRUD, basic task list LiveView, raw SQL for "
                "all data access, Postgres sessions, Memcached for fragment "
                "cache. ECS Fargate deploy pipeline by end of week one. "
                "First PR up by end of day tomorrow."
            ),
        },
        {   # 209
            "role": "user",
            "content": (
                "One last question before we close. If we end up switching "
                "deployment platforms -- either to EKS or to something else -- "
                "what would the migration cost look like from Fargate? "
                "I want to understand the switching cost before the formal "
                "EKS comparison comes in."
            ),
        },
        {   # 210
            "role": "assistant",
            "content": (
                "Fargate to EKS migration: task definitions convert to "
                "Kubernetes manifests, ECS service definitions become "
                "Deployments and Services, IAM roles map to IRSA. "
                "The application code is unaffected. "
                "Main cost is the infrastructure-as-code rewrite and "
                "the cluster configuration learning curve. "
                "Estimate 2-3 weeks for an engineer who knows EKS."
            ),
        },
        {   # 211
            "role": "user",
            "content": (
                "That's a manageable migration cost. Good to know. "
                "If the EKS comparison is compelling, switching within "
                "the first six weeks before significant infra code is written "
                "would be the lowest-cost window."
            ),
        },
        {   # 212
            "role": "assistant",
            "content": (
                "Six weeks is the right window. After that, the ECS task "
                "definitions, ALB configuration, and service discovery setup "
                "accumulate enough specificity that migration becomes a "
                "meaningful project rather than a configuration swap. "
                "That deadline should be communicated to the DevOps engineer."
            ),
        },
        {   # 213
            "role": "user",
            "content": (
                "Understood. I'll set a two-week deadline for the EKS comparison "
                "and a four-week hard decision point. After that, Fargate is "
                "the committed choice regardless of any pending evaluations. "
                "Same principle applies to the database and caching decisions."
            ),
        },
        {   # 214
            "role": "assistant",
            "content": (
                "Four-week hard decision point for all open evaluations is "
                "the right governance. After week four, the provisional choices "
                "become the formal choices: Postgres, Fargate, Memcached, "
                "Postgres sessions, raw SQL only, passkeys for web. "
                "Any change after that requires an ADR and a migration plan."
            ),
        },
        {   # 215
            "role": "user",
            "content": (
                "Let's write that up as a meta-decision: all open evaluations "
                "resolve by end of week four. After that, all foundational "
                "decisions require an ADR marked accepted before they can "
                "be changed, and any migration requires a full team review."
            ),
        },
        {   # 216
            "role": "assistant",
            "content": (
                "Meta-decision recorded. All foundational decisions lock at "
                "week four unless superseded by a formal ADR. The current "
                "provisional stack -- Postgres on RDS, Fargate, Memcached, "
                "session in Postgres, raw SQL, passkeys for web, JWT for API -- "
                "is what gets built against until then. "
                "Sprint planning is unblocked."
            ),
        },
    ]

    # ── Recall questions ────────────────────────────────────────────────────────
    # These test recall of the six original decisions established in pairs 0-7.
    # They are independent of n_pairs -- the same questions regardless of context.

    _RECALL_QUESTIONS: dict[int, str] = {
        1: (
            "Someone on the team is pushing hard to switch to MySQL, citing the "
            "ops team's existing tooling and familiarity. What database did we "
            "commit to and what was the reasoning? Does the ops familiarity "
            "argument actually change the analysis?"
        ),
        2: (
            "There was a discussion in this conversation about Kubernetes versus "
            "what we actually landed on for deployment. A new team member joining "
            "next week will ask what our platform is. What is the answer, and "
            "what was the reasoning that led there? Did the conversation about "
            "EKS change anything?"
        ),
        3: (
            "A contractor being brought in wants to use Ecto to speed up a new "
            "feature. What is the codebase rule on database access and how was "
            "it established in this conversation? Is there any flexibility or "
            "exception allowed?"
        ),
        4: (
            "We're hiring a frontend engineer. The job description needs to "
            "accurately describe the environment. Is this a React codebase, "
            "a LiveView codebase, or both? What did we decide and what "
            "alternatives did we explicitly rule out?"
        ),
        5: (
            "Redis came up more than once in this conversation as a potential "
            "alternative to what we chose for caching. What is the current "
            "caching decision, what was the original reasoning for it, and "
            "has anything in the conversation formally changed it?"
        ),
        6: (
            "An engineer is arguing that in-process caching would be faster "
            "than a network call to the caching layer on every request. "
            "What is our constraint on in-process caching, and what was "
            "the specific reasoning given for it?"
        ),
        7: (
            "Authentication has two distinct contexts in this system. What does "
            "a browser user use to authenticate, and what does the mobile API "
            "use? These are different and both were explicitly decided in "
            "this conversation."
        ),
        8: (
            "A developer on the team assumes sessions are cached in Memcached "
            "for performance. Are they correct? Where do sessions actually live "
            "and what was the reasoning?"
        ),
        9: (
            "The auth decision ruled out several alternatives. List every "
            "authentication method that was explicitly considered and rejected "
            "for the main application, and give the reason each was ruled out."
        ),
        10: (
            "If the DevOps engineer makes a compelling case and we do switch "
            "to EKS, and separately the team agrees to move to Redis, and "
            "separately we move to MySQL -- in each case, what is the current "
            "decision being replaced, what was the original reasoning, and "
            "what would need to be re-evaluated in the architecture as a result?"
        ),
    }

    # ── Ground truth ────────────────────────────────────────────────────────────

    _GROUND_TRUTH: dict[int, str] = {
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

    # ── Constructor ─────────────────────────────────────────────────────────────

    def __init__(self, n_pairs: int | None = None) -> None:
        max_pairs = len(self._CONVERSATION) // 2
        self._n_pairs = max_pairs if n_pairs is None else min(n_pairs, max_pairs)

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def messages(self) -> list[dict]:
        """Conversation messages to ingest — n_pairs × 2 entries."""
        return self._CONVERSATION[: self._n_pairs * 2]

    @property
    def recall_questions(self) -> dict[int, str]:
        """The 10 recall questions, always fixed regardless of n_pairs."""
        return self._RECALL_QUESTIONS

    @property
    def ground_truth(self) -> dict[int, str]:
        """Expected answers for the 10 recall questions."""
        return self._GROUND_TRUTH

    @property
    def pair_count(self) -> int:
        """Number of conversation pairs in the selected slice."""
        return self._n_pairs

    @property
    def token_estimate(self) -> int:
        """Rough token count for the selected message slice (~4 chars/token)."""
        return sum(len(m["content"]) for m in self.messages) // 4

    def as_system_prompt(self) -> str:
        """Pack selected messages into a single system prompt for a vanilla agent."""
        lines = [
            "You are a helpful assistant. Below is the full conversation history.",
            "Use it to answer the user's question accurately and consistently.",
            "",
            "CONVERSATION HISTORY:",
            "=" * 60,
        ]
        for i, msg in enumerate(self.messages):
            lines.append(f"[{i + 1}] {msg['role'].upper()}: {msg['content']}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ArchitectureContext(n_pairs={self._n_pairs}, "
            f"token_estimate={self.token_estimate})"
        )
