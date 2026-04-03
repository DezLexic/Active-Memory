# -*- coding: utf-8 -*-
import os
os.environ["OLLAMA_KEEP_ALIVE"] = "10m"

"""
benchmark_v4.py

Controlled comparison between a vanilla agent and the Active Memory pipeline.

  Agent A - Vanilla
    For each recall question the entire 80-message chat log is packed into a
    single system prompt and sent to Ollama alongside the question.  No
    compression, no retrieval.  The model sees everything but has to hold it
    all in its context window at once.

  Agent B - Active Memory
    All 80 messages are batch-ingested via pipeline.chat().  This builds the
    summary (Observer) and memory store (Curator -> Retrieval) as if the
    conversation had actually happened.  Each recall question is then answered
    by the Active Agent, which reads from the compressed Bucket context.

Six decisions are locked in during messages 1-15.  Messages 16-65 are normal
development discussion that dilutes the early context.  Messages 66-80 are
the ten recall questions (indices 65-79).

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
# 1.  CHAT LOG  (80 messages)
#
#     Index  0-14  : six decisions, explicit choice + reason
#     Index 15-64  : development discussion, dilutes early context
#     Index 65-79  : ten recall questions (user role only; answered by agents)
# ──────────────────────────────────────────────────────────────────────────────

CHAT_LOG: list[dict] = [

    # ── Decision block (messages 1-15, indices 0-14) ──────────────────────────

    {   # 0
        "role": "user",
        "content": (
            "Decision one is locked: we are building this product with Elixir and the "
            "Phoenix framework. The BEAM VM gives us lightweight process-per-connection "
            "concurrency with no threads, supervision trees mean faults are isolated and "
            "auto-recovered, and Phoenix LiveView lets us ship real-time interactive UI "
            "without a separate JavaScript SPA. No React. No Vue. No Next.js. "
            "Every interactive feature goes through LiveView."
        ),
    },
    {   # 1
        "role": "assistant",
        "content": (
            "Elixir and Phoenix is a strong choice for this. BEAM's actor model "
            "scales horizontally and fault tolerance is structural rather than bolted on. "
            "LiveView eliminates the client-server state sync problem, which means less "
            "code and fewer race conditions. Ruling out SPA frameworks removes an entire "
            "category of complexity. Decision one confirmed: Elixir, Phoenix, LiveView. "
            "No SPA frameworks of any kind."
        ),
    },
    {   # 2
        "role": "user",
        "content": (
            "Decision two: the database is PostgreSQL. We are not using CockroachDB, "
            "PlanetScale, or any other distributed system. Plain PostgreSQL running on "
            "a managed RDS instance. The team knows it, the tooling is mature, and we "
            "do not have the scale to justify a distributed SQL database today. "
            "Decision three: raw SQL only. No Ecto ORM, no query builders, no generated "
            "queries. Every database call is a handwritten parameterised SQL query. "
            "This is a hard codebase rule, not a preference."
        ),
    },
    {   # 3
        "role": "assistant",
        "content": (
            "PostgreSQL on RDS is sensible — managed backups, read replicas, and "
            "point-in-time recovery out of the box without owning the ops. Raw SQL only "
            "is a discipline that pays off in long-running projects; execution plans stay "
            "predictable and there is no ORM magic to debug under load. "
            "Decision two: PostgreSQL on managed RDS. "
            "Decision three: raw SQL, no Ecto, no query builders. "
            "Authentication next."
        ),
    },
    {   # 4
        "role": "user",
        "content": (
            "Decision four: authentication is passkeys only. No passwords. No magic links. "
            "No OAuth. WebAuthn passkeys backed by device biometrics or hardware keys. "
            "We store the public key credential, never a secret. There is no password "
            "reset flow to build or maintain because there is nothing to reset. "
            "Passkeys are phishing-resistant by design and the UX on modern devices is "
            "faster than typing a password. This is the auth story for the product, full stop."
        ),
    },
    {   # 5
        "role": "assistant",
        "content": (
            "Passkeys are the right call for a greenfield product. No credential "
            "storage liability, no phishing surface, and biometric UX is already "
            "mainstream on mobile and desktop. WebAuthn support in modern browsers is "
            "solid. The implementation surface is smaller than it sounds -- register "
            "and authenticate RPCs, public key storage in PostgreSQL, and a CBOR parser. "
            "Decision four confirmed: passkeys only, no passwords, no magic links, no OAuth."
        ),
    },
    {   # 6
        "role": "user",
        "content": (
            "Decision five is the deployment platform: AWS ECS with Fargate. "
            "No Kubernetes. The team does not have Kubernetes expertise and I am not "
            "willing to build that skillset on a production timeline. Fargate handles "
            "the compute layer -- we define tasks, set the CPU and memory, and AWS runs "
            "the containers. We keep three tasks running at minimum at all times for "
            "availability. Auto-scaling is configured but the floor is three."
        ),
    },
    {   # 7
        "role": "assistant",
        "content": (
            "Fargate removes the node management overhead entirely. No patching EC2 "
            "instances, no managing node groups, no etcd to worry about. ECS task "
            "definitions are declarative and deployable via the CLI or CDK. "
            "Three-task minimum for availability aligns with the Phoenix cluster "
            "distribution model -- LiveView PubSub works across nodes via pg module. "
            "Decision five: AWS ECS Fargate, three-task minimum, no Kubernetes."
        ),
    },
    {   # 8
        "role": "user",
        "content": (
            "Decision six: caching layer is Memcached, not Redis. We evaluated both. "
            "Memcached is simpler, multi-threaded, and we do not need Redis data "
            "structures or persistence for our caching use case. We cache rendered "
            "page fragments and query results. No session data in cache -- sessions "
            "live in the database. No in-process caching of any kind; with multiple "
            "tasks in ECS, in-process caches diverge and cause stale-read bugs. "
            "Everything cached goes through Memcached."
        ),
    },
    {   # 9
        "role": "assistant",
        "content": (
            "Memcached is the right choice when you need a fast, durable, distributed "
            "cache without Redis's feature surface. Multi-threaded means better core "
            "utilisation on modern instances. Sessions in PostgreSQL rather than cache "
            "is the safer default -- consistent reads, no cache-miss invalidation logic. "
            "No in-process caching is critical with ECS multi-task: shared Memcached "
            "is the only correct answer. "
            "Decision six: Memcached for caching, no in-process caching, sessions in DB."
        ),
    },
    {   # 10
        "role": "user",
        "content": (
            "Good. The six decisions are now locked. To state them cleanly:\n"
            "(1) Elixir + Phoenix + LiveView -- no SPA frameworks.\n"
            "(2) PostgreSQL on managed RDS -- no distributed SQL.\n"
            "(3) Raw SQL only -- no Ecto, no query builders.\n"
            "(4) Passkeys only -- no passwords, no magic links, no OAuth.\n"
            "(5) AWS ECS Fargate, three-task minimum -- no Kubernetes.\n"
            "(6) Memcached for caching, no in-process cache, sessions in DB.\n"
            "Nothing overrides these without a full team discussion. "
            "Let's move on to the data model."
        ),
    },
    {   # 11
        "role": "assistant",
        "content": (
            "Confirmed. Six decisions locked. The data model -- tell me the core "
            "entities you are modelling and I'll help work through the schema."
        ),
    },
    {   # 12
        "role": "user",
        "content": (
            "The core entities are: Workspace, User, Membership, Project, Task, "
            "Comment, and Label. A Workspace owns Projects. Users belong to Workspaces "
            "via Memberships with a role column. Projects contain Tasks. Tasks can have "
            "Comments and Labels. That is the starting shape."
        ),
    },
    {   # 13
        "role": "assistant",
        "content": (
            "Clean hierarchy. Workspace -> Project -> Task is the ownership chain. "
            "Membership is a join table with a role enum -- at minimum owner, admin, "
            "member. Comment is a polymorphic or task-scoped entity. "
            "Label is likely a many-to-many between Task and a labels table. "
            "I'd suggest a task_labels join table rather than an array column to keep "
            "queries composable. What does the Task status lifecycle look like?"
        ),
    },
    {   # 14
        "role": "user",
        "content": (
            "Task statuses: todo, in_progress, in_review, done, cancelled. "
            "Stored as a varchar with a check constraint. Not an enum type in Postgres "
            "because enum migration is painful. Status transitions are enforced in "
            "application code, not in the database. We will write a raw SQL function "
            "to log status transitions to an audit table."
        ),
    },

    # ── Development discussion (messages 16-65, indices 15-64) ────────────────

    {   # 15
        "role": "assistant",
        "content": (
            "varchar with a check constraint is pragmatic. Adding a value to a Postgres "
            "enum requires a lock; varchar + check constraint can be altered without "
            "downtime. Application-layer state machine with a raw SQL audit log is clean. "
            "What indexes do we put on the tasks table from the start?"
        ),
    },
    {   # 16
        "role": "user",
        "content": (
            "Tasks table indexes: primary key on id, index on (project_id, status) "
            "for the main task list query, index on (assignee_id) for the my-tasks view, "
            "index on (due_date) for the upcoming filter. All defined in raw SQL "
            "migration files with explicit names so drops are clean."
        ),
    },
    {   # 17
        "role": "assistant",
        "content": (
            "Composite index on (project_id, status) covers the most common read path. "
            "Naming indexes explicitly is good hygiene -- pg_dump includes names and "
            "replication is cleaner. due_date index will need NULLS LAST if unset tasks "
            "should sort to the bottom. Noted."
        ),
    },
    {   # 18
        "role": "user",
        "content": (
            "Let's talk about the LiveView architecture. The main app has three "
            "top-level LiveViews: WorkspaceLayout, ProjectBoard, and UserDashboard. "
            "WorkspaceLayout is the shell -- sidebar, nav, notification tray. "
            "ProjectBoard renders the kanban or list view of tasks. "
            "UserDashboard is the personal view across all workspaces."
        ),
    },
    {   # 19
        "role": "assistant",
        "content": (
            "Nesting LiveViews via live_render is the right pattern for this. "
            "WorkspaceLayout handles auth state, workspace context, and global "
            "notification subscriptions. ProjectBoard and UserDashboard mount inside it. "
            "The PubSub topic structure should mirror the hierarchy: "
            "workspace:{id}, project:{id}, user:{id}."
        ),
    },
    {   # 20
        "role": "user",
        "content": (
            "For real-time task updates the pattern is: when a task is mutated, "
            "broadcast on the project:{id} topic. All connected ProjectBoard LiveViews "
            "for that project receive the message and update in place. "
            "No full page reload, no polling. The broadcast payload is the minimum "
            "diff -- just the changed task struct."
        ),
    },
    {   # 21
        "role": "assistant",
        "content": (
            "Minimal diff broadcast is the right tradeoff. Sending the full task list "
            "on every change is expensive at scale. With the task struct as the payload "
            "the LiveView can pattern-match on the id and update just that row in the "
            "assigns. The handle_info callback is clean and testable."
        ),
    },
    {   # 22
        "role": "user",
        "content": (
            "The API surface: we need a JSON API in addition to LiveView for the "
            "mobile client that will come later. Phoenix Router gets a /api/v1 scope "
            "with its own pipeline. JWT authentication on the API routes -- not "
            "passkeys, those are browser-only. JWTs are generated at login and "
            "short-lived, 15 minutes, refreshed via a refresh token stored in the DB."
        ),
    },
    {   # 23
        "role": "assistant",
        "content": (
            "Separate /api/v1 scope with its own plug pipeline is the Phoenix idiomatic "
            "approach. JWT for the REST API is fine -- passkeys authenticate the browser "
            "session and the API gets a JWT derived from that session. "
            "Short-lived access tokens with DB-stored refresh tokens is the right "
            "security tradeoff. Refresh token rotation on use to detect reuse."
        ),
    },
    {   # 24
        "role": "user",
        "content": (
            "Error handling policy: all database errors are logged with full context "
            "-- query, params, error code -- and surfaced to the user as generic "
            "messages. No raw Postgres error messages reach the client. "
            "Constraint violations on known paths get specific user-facing messages. "
            "Unknown errors get a generic 'something went wrong' plus a trace ID."
        ),
    },
    {   # 25
        "role": "assistant",
        "content": (
            "Full-context DB error logging with generic client messages is correct "
            "from a security and UX perspective. Leaking constraint names or table "
            "names to clients is an information disclosure risk. Trace IDs are "
            "essential for support workflows -- the user can report the ID and "
            "you can correlate it in the log aggregator."
        ),
    },
    {   # 26
        "role": "user",
        "content": (
            "CI/CD: GitHub Actions for the pipeline. On every push: mix format, "
            "mix credo, mix dialyzer, mix test. Dialyzer runs on a cache of the PLT "
            "so it does not rebuild from scratch on every run. The pipeline must be "
            "green before any merge. No force-push to main."
        ),
    },
    {   # 27
        "role": "assistant",
        "content": (
            "mix format + credo + dialyzer + test is the right Elixir CI stack. "
            "PLT caching in GitHub Actions via cache key on the mix.lock hash is "
            "standard. Dialyzer is slow on cold cache, fast on warm -- most teams "
            "accept the cold cache cost on dependency changes. No force-push to "
            "main is enforced via branch protection rules."
        ),
    },
    {   # 28
        "role": "user",
        "content": (
            "Telemetry: we instrument all database queries via Postgrex telemetry "
            "events, all LiveView lifecycle events via Phoenix.LiveView.Telemetry, "
            "and custom business events via :telemetry.execute. "
            "Metrics go to Prometheus via prometheus_ex. Dashboards in Grafana. "
            "Alerting via PagerDuty for P1 and P2."
        ),
    },
    {   # 29
        "role": "assistant",
        "content": (
            "Postgrex telemetry gives query duration, result size, and connection "
            "pool stats without instrumentation in application code. Phoenix.LiveView "
            "telemetry covers mount, handle_event, handle_info latencies -- critical "
            "for catching slow callbacks. Prometheus + Grafana is the standard OSS "
            "stack. PagerDuty escalation policies need to be set up before launch."
        ),
    },
    {   # 30
        "role": "user",
        "content": (
            "The Comment entity supports markdown. We render it server-side to HTML "
            "using Earmark before storing the rendered HTML alongside the raw markdown. "
            "We store both because re-rendering on read is expensive at comment volume. "
            "HTML is sanitised before storage with HtmlSanitizeEx. "
            "Sanitise first, store, render as raw HTML in LiveView."
        ),
    },
    {   # 31
        "role": "assistant",
        "content": (
            "Storing both raw markdown and rendered HTML is the right tradeoff for "
            "read-heavy comment threads. Sanitise then store is the correct order -- "
            "sanitising on every read means trusting the sanitiser to be safe on "
            "untrusted input every single time, which raises the blast radius of any "
            "sanitiser bug. Sanitise once at write, trust the stored HTML."
        ),
    },
    {   # 32
        "role": "user",
        "content": (
            "Pagination: cursor-based throughout. No page numbers. Cursors are "
            "opaque base64-encoded tuples of (inserted_at, id). The query always "
            "fetches page_size + 1 rows; if we get page_size + 1 we know there is "
            "a next page and we return the extra row's cursor as the next_cursor. "
            "This works correctly with the raw SQL constraint."
        ),
    },
    {   # 33
        "role": "assistant",
        "content": (
            "Cursor-based pagination with (inserted_at, id) is stable under concurrent "
            "inserts, which page-number pagination is not. Fetching n+1 to detect "
            "the next page is the cleanest approach -- no COUNT query, no off-by-one "
            "on the last page. Base64 opaque cursors prevent clients from constructing "
            "arbitrary offsets."
        ),
    },
    {   # 34
        "role": "user",
        "content": (
            "File attachments: tasks and comments can have attachments. "
            "Files upload directly to S3 using presigned URLs -- the server generates "
            "the presigned URL, returns it to the client, the client uploads directly. "
            "The server stores only the S3 key and metadata. "
            "No files pass through our servers. Max file size 50MB enforced by S3 policy."
        ),
    },
    {   # 35
        "role": "assistant",
        "content": (
            "Direct-to-S3 upload via presigned URLs is correct. Proxying uploads "
            "through the application server wastes bandwidth, increases latency, and "
            "requires buffering the full file in memory or disk. S3 enforces the "
            "50MB limit server-side. Store the S3 key plus content_type, size, "
            "and original_filename in the DB. Generate presigned download URLs "
            "on demand rather than storing public URLs."
        ),
    },
    {   # 36
        "role": "user",
        "content": (
            "Search: we will add full-text search using PostgreSQL's built-in FTS. "
            "tsvector columns on tasks (title + description) and comments (body_raw). "
            "Updated via trigger on insert and update. ts_query with websearch_to_tsquery "
            "for the query side. We will not use Elasticsearch or Meilisearch -- "
            "Postgres FTS is sufficient for our data volume."
        ),
    },
    {   # 37
        "role": "assistant",
        "content": (
            "Postgres FTS is the right call at this scale. tsvector maintained by "
            "trigger keeps the index current without application changes. "
            "websearch_to_tsquery handles quoted phrases and negation naturally. "
            "The headline function can highlight matched terms in results. "
            "No external search service means no sync lag and no extra ops."
        ),
    },
    {   # 38
        "role": "user",
        "content": (
            "Notifications: users get in-app notifications and optional email. "
            "In-app notifications are stored in a notifications table and pushed "
            "to the user's browser via PubSub on the user:{id} topic. "
            "Email notifications go through an Oban background job to an SES-backed "
            "SMTP relay. Notification preferences are per-user per-event-type."
        ),
    },
    {   # 39
        "role": "assistant",
        "content": (
            "Oban for background jobs is the Elixir standard. Jobs are persisted "
            "in PostgreSQL so they survive restarts and are exactly-once with "
            "transactional guarantees. SES for email is cost-effective at low "
            "volume. Per-user per-event-type preferences means a preferences table "
            "with (user_id, event_type, channel, enabled) -- sparse but flexible."
        ),
    },
    {   # 40
        "role": "user",
        "content": (
            "Let's discuss the passkey registration flow in detail. "
            "User visits /auth/register. Server generates a challenge and stores it "
            "in the session with a short TTL. Frontend calls navigator.credentials.create "
            "with the challenge. User completes biometric. Frontend sends the credential "
            "response to /auth/register POST. Server verifies the attestation, stores "
            "the public key credential in the passkey_credentials table. User is logged in."
        ),
    },
    {   # 41
        "role": "assistant",
        "content": (
            "The registration flow is correct. Challenge generation should use "
            "cryptographically strong random bytes -- :crypto.strong_rand_bytes(32). "
            "Session-stored challenges with TTL prevent replay. Attestation verification "
            "depends on whether you require device attestation or just basic passkey "
            "support -- most consumer apps skip attestation verification and just "
            "verify the signature. Store credential_id, public_key_cbor, sign_count, "
            "transports, and user_id."
        ),
    },
    {   # 42
        "role": "user",
        "content": (
            "The passkey authentication flow: user visits /auth/login. "
            "Server generates challenge. Frontend calls navigator.credentials.get "
            "with allowCredentials list for that user if username was entered, "
            "or empty for conditional UI. User completes biometric. "
            "Frontend sends assertion response. Server verifies signature against "
            "stored public key. Server checks sign_count to detect cloned credentials. "
            "Session created, user redirected."
        ),
    },
    {   # 43
        "role": "assistant",
        "content": (
            "Sign count verification is critical for detecting cloned credentials. "
            "If stored_sign_count >= received_sign_count the credential may have been "
            "cloned -- policy decision whether to hard-reject or flag and alert. "
            "Conditional UI (discoverable credentials without username entry) is the "
            "modern passkey UX pattern and worth supporting from the start. "
            "The verification library for Elixir is wax or webauthn_components."
        ),
    },
    {   # 44
        "role": "user",
        "content": (
            "Deployment pipeline: Dockerfile uses a multi-stage build. "
            "Stage one: Elixir builder image, fetches deps, compiles, builds release. "
            "Stage two: minimal runtime image, copies the release binary. "
            "Final image is under 100MB. GitHub Actions builds and pushes to ECR. "
            "ECS task definition updated via aws ecs update-service."
        ),
    },
    {   # 45
        "role": "assistant",
        "content": (
            "Multi-stage Docker build for Elixir releases is standard. The mix release "
            "binary is self-contained -- runtime image only needs the OS libraries "
            "the BEAM links against, typically libc and libssl. Alpine-based runtime "
            "images get under 80MB comfortably. ECR + ECS update-service with a "
            "minimum healthy percent of 100 ensures zero-downtime deploys with "
            "Fargate's rolling update."
        ),
    },
    {   # 46
        "role": "user",
        "content": (
            "Database migrations: we use mix ecto.migrate, but since we are raw SQL "
            "only we write the migration files as raw SQL wrapped in Ecto.Migration's "
            "execute/2 calls. Up and down. Migrations run as part of the deployment "
            "pipeline before the new ECS tasks are started. "
            "Migration failures abort the deployment."
        ),
    },
    {   # 47
        "role": "assistant",
        "content": (
            "execute/2 in Ecto.Migration with raw SQL up and down is the correct "
            "approach given the raw SQL constraint. Migrations run before traffic "
            "shifts -- this requires backward-compatible migrations for zero-downtime "
            "deploys. Additive changes first, then clean up old columns in a follow-up "
            "migration after the old code is fully retired."
        ),
    },
    {   # 48
        "role": "user",
        "content": (
            "Rate limiting: Phoenix has plug_attack which we will use for the auth "
            "endpoints. Three failed passkey authentication attempts from the same IP "
            "in 60 seconds triggers a 60-second block. API endpoints get a per-token "
            "rate limit of 1000 requests per minute tracked in Memcached."
        ),
    },
    {   # 49
        "role": "assistant",
        "content": (
            "plug_attack with Memcached as the backing store works well. "
            "Per-IP auth throttling is the right scope for passkey failures -- "
            "account enumeration is not possible with passkeys so IP-based throttling "
            "is sufficient. API rate limiting per-token in Memcached is straightforward: "
            "increment a counter with a 60-second TTL, reject if over 1000."
        ),
    },
    {   # 50
        "role": "user",
        "content": (
            "Let's talk about the Workspace billing model. Each Workspace is on a plan: "
            "free, pro, or enterprise. Plan limits: free gets 5 projects and 10 members, "
            "pro gets unlimited projects and 50 members, enterprise is negotiated. "
            "Plan is stored on the workspace record. Limit enforcement is in the "
            "application layer via a policy module, not in the database."
        ),
    },
    {   # 51
        "role": "assistant",
        "content": (
            "Application-layer policy enforcement with a policy module is testable "
            "and flexible. A Policies.Workspace module with functions like "
            "can_add_project?(workspace) and can_add_member?(workspace) keeps the "
            "enforcement logic in one place. Stripe integration will handle plan "
            "upgrades -- webhook on subscription updates triggers a workspace plan update."
        ),
    },
    {   # 52
        "role": "user",
        "content": (
            "Feature flags: we will use a simple database-backed feature flag system. "
            "A feature_flags table with flag_name, enabled, and rollout_percentage. "
            "A FlagCache module reads flags from Memcached with a 30-second TTL, "
            "falls back to the database on cache miss. Flag evaluation is in the "
            "application layer via a Flags.enabled?(flag_name, user) function."
        ),
    },
    {   # 53
        "role": "assistant",
        "content": (
            "Database-backed feature flags with Memcached caching is appropriately "
            "simple for an early-stage product. No external flag service to maintain. "
            "Rollout percentage with user ID hashing gives deterministic per-user "
            "flag assignment. The 30-second cache TTL means flag changes propagate "
            "within half a minute without a cache flush mechanism."
        ),
    },
    {   # 54
        "role": "user",
        "content": (
            "Audit logging: all mutations to core entities -- Task create/update/delete, "
            "Membership create/update/delete, Project create/update/delete -- "
            "are written to an audit_events table. Schema: entity_type, entity_id, "
            "action, actor_user_id, changes (jsonb), inserted_at. "
            "Written in the same database transaction as the mutation."
        ),
    },
    {   # 55
        "role": "assistant",
        "content": (
            "Same-transaction audit log writes are the correct approach for "
            "consistency -- no audit record without the mutation, no mutation without "
            "the audit record. jsonb changes column is flexible. Indexes on "
            "(entity_type, entity_id) and (actor_user_id, inserted_at) cover the "
            "main query patterns. Partition by inserted_at month if volume grows."
        ),
    },
    {   # 56
        "role": "user",
        "content": (
            "Let's talk through the task assignment notification flow. "
            "User A assigns Task T to User B. The update mutation fires. "
            "Audit log is written in the same transaction. "
            "After commit, an Oban job is enqueued for the email notification. "
            "Simultaneously, a PubSub broadcast goes to user:{B.id}. "
            "User B's dashboard LiveView receives the message and increments "
            "the notification badge."
        ),
    },
    {   # 57
        "role": "assistant",
        "content": (
            "Post-commit Oban job enqueue is critical -- enqueuing inside the "
            "transaction risks a job running before the commit, finding no data, "
            "and failing. Oban's with_job/1 + Multi or manual post-commit hook "
            "handles this correctly. The PubSub broadcast can go inside the "
            "LiveView channel after handle_event returns -- it races with the "
            "commit but that is acceptable for UI updates."
        ),
    },
    {   # 58
        "role": "user",
        "content": (
            "Testing strategy: ExUnit for all unit and integration tests. "
            "Database tests use the sandbox adapter for isolation. "
            "LiveView tests via Phoenix.LiveViewTest -- mount, send_event, "
            "assert_patch, assert_push. No end-to-end browser tests in CI -- "
            "they are too flaky. Instead we rely on comprehensive LiveView "
            "unit tests that cover all user interactions."
        ),
    },
    {   # 59
        "role": "assistant",
        "content": (
            "Phoenix.LiveViewTest covers the full LiveView lifecycle without a "
            "browser. Sandbox adapter gives each test its own transaction that "
            "rolls back at the end, so DB state is isolated without truncation. "
            "Skipping browser tests in CI is pragmatic -- the flakiness cost "
            "outweighs the coverage benefit at early stage. Manual browser smoke "
            "tests before releases is a reasonable substitute."
        ),
    },
    {   # 60
        "role": "user",
        "content": (
            "Secret management: all secrets are stored in AWS Secrets Manager. "
            "The ECS task definition references secrets by ARN. "
            "The Elixir application reads them at startup via the AWS SDK. "
            "No secrets in environment variables baked into the Docker image. "
            "Rotation is handled by Secrets Manager with a Lambda rotation function "
            "for the database credentials."
        ),
    },
    {   # 61
        "role": "assistant",
        "content": (
            "AWS Secrets Manager injection at task startup is correct. ECS native "
            "secrets integration decrypts and injects at container start without "
            "the application needing IAM SDK calls directly. Automatic rotation "
            "with a Lambda rotator for RDS credentials is the managed approach -- "
            "Secrets Manager handles the rotation schedule and the application "
            "re-reads credentials on reconnect after a connection failure."
        ),
    },
    {   # 62
        "role": "user",
        "content": (
            "Backups: RDS automated daily snapshots with 30-day retention. "
            "Point-in-time recovery enabled. S3 buckets with versioning enabled "
            "and a 90-day lifecycle on old versions. "
            "Quarterly restore drills to verify RDS snapshots are usable. "
            "Backup metrics alerted via CloudWatch -- alert if no snapshot "
            "in 26 hours."
        ),
    },
    {   # 63
        "role": "assistant",
        "content": (
            "30-day RDS snapshot retention with PITR is solid. 26-hour CloudWatch "
            "alert gives a buffer over the daily schedule. Quarterly restore drills "
            "are essential -- an untested backup is not a backup. S3 versioning with "
            "90-day lifecycle on old versions balances recovery capability against "
            "storage cost. Object Lock on the backup bucket can prevent accidental "
            "deletion if your threat model includes insider risk."
        ),
    },
    {   # 64
        "role": "user",
        "content": (
            "Good. I think we've covered the main architecture concerns. "
            "The decisions from the start of the conversation remain the "
            "foundation. Let's move to the recall check to make sure we "
            "have not drifted from what we decided."
        ),
    },

    # ── Recall questions (messages 66-80, indices 65-79) ─────────────────────
    # These are posed as user messages. Both agents answer them independently.

    {   # 65
        "role": "user",
        "content": (
            "RECALL CHECK 1: What language and framework did we decide on "
            "at the very start of this conversation, and what did we explicitly "
            "rule out on the frontend side?"
        ),
    },
    {   # 66
        "role": "user",
        "content": (
            "RECALL CHECK 2: What database are we using and where is it hosted? "
            "What was the reasoning given?"
        ),
    },
    {   # 67
        "role": "user",
        "content": (
            "RECALL CHECK 3: What is the query policy for database access? "
            "Is there any ORM or query builder allowed?"
        ),
    },
    {   # 68
        "role": "user",
        "content": (
            "RECALL CHECK 4: What is the authentication method for the main "
            "application? What was explicitly ruled out?"
        ),
    },
    {   # 69
        "role": "user",
        "content": (
            "RECALL CHECK 5: What is the deployment platform and what compute "
            "model are we using? What was explicitly ruled out?"
        ),
    },
    {   # 70
        "role": "user",
        "content": (
            "RECALL CHECK 6: What is the caching technology? "
            "What constraint was placed on in-process caching, and where do sessions live?"
        ),
    },
    {   # 71
        "role": "user",
        "content": (
            "RECALL CHECK 7: For the JSON API that was discussed for the mobile "
            "client, what authentication method is used -- and is this the same "
            "as the main app authentication?"
        ),
    },
    {   # 72
        "role": "user",
        "content": (
            "RECALL CHECK 8: We discussed caching and rate limiting together. "
            "What store backs the API rate limiting, and what are the rate limit parameters?"
        ),
    },
    {   # 73
        "role": "user",
        "content": (
            "RECALL CHECK 9: Where do session credentials live in this system -- "
            "in the cache, in the database, or somewhere else? "
            "Be specific about what was decided."
        ),
    },
    {   # 74
        "role": "user",
        "content": (
            "RECALL CHECK 10: If someone on the team proposed switching from the "
            "current caching technology to Redis, would that be consistent or "
            "inconsistent with what was decided? What was the actual decision and why?"
        ),
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# 2.  RECALL QUESTIONS  (indices 65-74 from the log, extracted for iteration)
# ──────────────────────────────────────────────────────────────────────────────

RECALL_QUESTIONS: dict[int, str] = {
    q_num: CHAT_LOG[64 + q_num]["content"]
    for q_num in range(1, 11)
}

# ──────────────────────────────────────────────────────────────────────────────
# 3.  GROUND TRUTH
# ──────────────────────────────────────────────────────────────────────────────

GROUND_TRUTH: dict[int, str] = {
    1: (
        "Elixir with the Phoenix framework and Phoenix LiveView for all interactive UI. "
        "SPA frameworks (React, Vue, Next.js) were explicitly ruled out."
    ),
    2: (
        "PostgreSQL on a managed RDS instance. The reasoning: the team knows it, "
        "tooling is mature, and the scale does not justify a distributed SQL database. "
        "CockroachDB was explicitly not chosen."
    ),
    3: (
        "Raw SQL only. No Ecto ORM, no Ecto.Repo, no query builders of any kind. "
        "Every database call is a handwritten parameterised SQL query. This is a "
        "hard codebase rule."
    ),
    4: (
        "Passkeys only (WebAuthn). No passwords, no magic links, no OAuth. "
        "All three alternatives were explicitly ruled out."
    ),
    5: (
        "AWS ECS with Fargate as the compute model. Three tasks minimum for "
        "availability. Kubernetes was explicitly ruled out because the team "
        "lacks Kubernetes expertise."
    ),
    6: (
        "Memcached for caching. No in-process caching of any kind. "
        "Sessions live in the database, not in the cache."
    ),
    7: (
        "The JSON API for the mobile client uses JWT authentication. "
        "This is different from the main app, which uses passkeys. "
        "Passkeys are browser-only; the API uses short-lived JWTs with "
        "DB-stored refresh tokens."
    ),
    8: (
        "API rate limiting is backed by Memcached. The limit is 1000 requests "
        "per minute per token, with a 60-second TTL counter."
    ),
    9: (
        "Sessions live in the database (PostgreSQL), not in the cache. "
        "This was an explicit decision: Memcached holds cached data and rate "
        "limit counters, not session credentials."
    ),
    10: (
        "Switching to Redis would be inconsistent with the decision. "
        "The decision was Memcached, chosen because it is simpler, multi-threaded, "
        "and the team does not need Redis data structures or persistence for "
        "their caching use case. Redis was evaluated and not chosen."
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# 4.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _format_chat_log_as_system_prompt() -> str:
    """Pack all 80 messages into a single system prompt string for Agent A."""
    lines: list[str] = [
        "You are a helpful assistant. Below is the full conversation history.",
        "Use it to answer the user's question accurately and consistently.",
        "",
        "CONVERSATION HISTORY:",
        "=" * 60,
    ]
    for i, msg in enumerate(CHAT_LOG):
        role   = msg["role"].upper()
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
        # Extract the first digit found in the response.
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

AGENT_MODEL  = "gemma3:4b"
JUDGE_MODEL  = "gemma3:4b"
CHROMA_PATH  = "./chroma_db_benchmark_v4"

# ──────────────────────────────────────────────────────────────────────────────
# 6.  AGENT A — VANILLA
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 78)
print("  BENCHMARK V4  —  Vanilla vs Active Memory Pipeline")
print("=" * 78)
print(f"\n  Model : {AGENT_MODEL}")
print(f"  Judge : {JUDGE_MODEL}")
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
    latency = time.time() - t0
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

# Ingest all 80 messages silently.
ingest_start = time.time()
for i, msg in enumerate(CHAT_LOG):
    pipeline.chat(msg["content"])
    if (i + 1) % 10 == 0:
        print(f"  Ingested {i + 1}/80 messages ...")
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

COL_Q    = 50
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
    q_text  = f"Q{q_num}: " + _truncate(RECALL_QUESTIONS[q_num], COL_Q - 4)
    sa      = scores_a[q_num]
    sb      = scores_b[q_num]
    la      = agent_a_latencies[q_num]
    lb      = agent_b_latencies[q_num]
    row = (
        f"  {q_text:<{COL_Q}}  "
        f"{sa:>{COL_SCORE}}  "
        f"{sb:>{COL_SCORE}}  "
        f"{la:>{COL_LAT - 1}.1f}s  "
        f"{lb:>{COL_LAT - 1}.1f}s"
    )
    print(row)

_print_divider("-", len(header))

# Averages
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
print(f"  Latency B    : {avg_lb:.1f}s avg per question (excludes {ingest_elapsed:.1f}s ingest)")
print()

# ──────────────────────────────────────────────────────────────────────────────
# 10.  WHAT ACTIVE MEMORY STORED
# ──────────────────────────────────────────────────────────────────────────────

_print_divider("=")
print("  ACTIVE MEMORY STORE — what the Curator persisted")
_print_divider("=")

retrieval = pipeline.retrieval
total_stored = retrieval._collection.count()
print(f"\n  Total items in Chroma: {total_stored}")

BROAD_QUERY = "technology decisions database authentication deployment caching"
print(f"  Query: \"{BROAD_QUERY}\"\n")

scored = retrieval._retrieve_scored(BROAD_QUERY)

if scored:
    for i, item in enumerate(scored, 1):
        print(f"  [{i}]  sim={item['similarity']:.4f}  ts={item['timestamp']:.0f}")
        words  = item["content"].split()
        line   = "       "
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
