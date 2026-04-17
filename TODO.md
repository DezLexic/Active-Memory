# Active Memory — Project TODO

Backlog of improvements to tackle in future sessions. Roughly grouped by area.

---

## Hiring / Portfolio Leverage

- **README results section** — add a crisp “why this matters / how it was
  evaluated / what improved” section near the top of `README.md`. Right now the
  architecture is compelling, but the evidence is not legible fast enough for a
  recruiter, hiring manager, or engineer doing a quick skim.

- **Demo walkthrough** — create a short video or GIF-driven walkthrough of the
  inspector showing the topic tree evolving, warm/cold memories landing, and a
  benchmark question being answered with retrieved context. The project is more
  impressive when people can see it working than when they only read about it.

- **Real integration example** — add a concrete “before / after Active Memory”
  example for a realistic long-running agent loop. Show the baseline flow, the
  integration diff, and one example where memory changes the answer quality.

- **Application positioning** — turn the project into resume / portfolio
  language that frames the work as AI systems engineering rather than “a side
  project.” Include a short summary emphasizing agent memory architecture,
  evals, retrieval infrastructure, and debugging tooling.

- **First-run trust pass** — tighten the repo’s onboarding experience so a
  reviewer can clone it, understand setup, and form confidence quickly. This
  includes fixing public doc mismatches, making the supported startup path
  obvious, and ensuring the repo communicates stability instead of “work in
  progress.”

---

## Immediate Trust / Correctness Fixes

- **Fix duplicate memory writes on eviction** — `Pipeline.update()` and
  `Pipeline.ingest()` auto-store evicted pairs to cold, but `Curator.evaluate()`
  still stores the evaluated pair again. Refactor the Curator path so it
  promotes or re-tiers an existing stored memory instead of creating duplicate
  entries that can distort retrieval and counts.

- **Align public startup docs with actual behavior** — the package docs still
  imply `Pipeline()` has a zero-config Ollama fallback, but the current code
  goes through `backend_from_env()` and raises when no backend is configured.
  Either restore the fallback or update `README.md`, `active_memory/__init__.py`,
  and related examples so first-run behavior matches what the code really does.

---

## Retrieval

- **Cap retrieval results** — the default `n_results` pulls up to 32 memories per
  question, which slows answering and floods the prompt. Add a configurable
  `max_results` cap (e.g. 8) to `Retrieval.retrieve()` with a sensible default,
  and expose it via `Pipeline` config and the inspector's agent-config panel.

- **Warm-first merge** — today warm and cold results are scored independently and
  merged by similarity. Consider always surfacing warm results first (up to a
  cap), then filling remaining slots from cold, so decisions/constraints are
  never buried by narrative matches.

---

## Curator / Pipeline

- **`use_batch_mode=False` full-batch warm evaluation** — the flag exists but is
  never exercised in production. When False, every evicted pair should be sent
  through the Curator for individual warm-promotion scoring instead of just the
  peeked mid-stack pair. Useful for short conversations where every pair matters.

- **Librarian auto-trigger** — `Librarian.run_consolidation()` must currently be
  called manually. Wire a post-conversation trigger into `Pipeline` (e.g. after
  `ingest()` finishes the full batch, or after N evictions) so consolidation
  happens automatically without extra caller code.

- **Re-evaluate the `quotes` slot** — the Observer `quotes` slot was added for
  verbatim phrases but rarely fires in practice (the LLM paraphrases instead).
  Either strengthen the prompt guidance for `quotes`, replace it with a
  `verbatim` or `reactions` slot, or remove it to slim the prompt.

---

## Observer

- **Topic node deduplication** — the Observer can create near-duplicate topic
  nodes (e.g. "Travel preferences" and "Preferences about travel"). Add a
  post-update dedup pass or strengthen the prompt to prefer updating an existing
  node over creating a new one.

- **Topic node pruning** — old topic nodes for resolved threads accumulate
  indefinitely. Add a max-node cap (e.g. 50) and a pruning strategy that removes
  the least-recently-updated nodes when the cap is reached.

---

## Backends / API

- **OpenAI backend** — documented as a known limitation in v0.1.0 but not yet
  implemented. Add `active_memory/backends/openai.py` to match the Anthropic and
  Ollama adapters.

- **Async / concurrent API** — `Pipeline.chat()`, `build_context()`, and
  `update()` are all synchronous. Add async variants (`achat()`, `aupdate()`)
  for callers that already run an event loop (FastAPI apps, notebook kernels).

---

## Inspector UI

- **Agent config panel** — expose `similarity_threshold`, `max_results`,
  `observer` rules, and model selection as editable fields in the inspector so
  parameters can be tuned without restarting the server.

- **Step mode** — advance one pair / one question at a time from the UI
  (button-driven, no automatic loop).

- **Baseline comparison** — run the same conversation twice: once with Active
  Memory enabled and once without (plain context window only). Display scores
  side-by-side so the memory lift is visible.

- **Average score display** — show a final summary panel when a LoCoMo run
  completes: mean score across all QA pairs, warm-hit rate, cold-hit rate, and
  Curator promotion rate.

- **Custom conversation upload** — allow the user to upload a JSON conversation
  file (same schema as LoCoMo) from the UI rather than hard-coding a dataset
  path in the server config.

- **Custom probe query** — type a free-text question in the UI against a paused
  pipeline to see retrieved memories and similarity scores without advancing the
  benchmark state.

- **Export trace** — download the full per-pair / per-question snapshot history
  as JSON for offline analysis.

- **Side-by-side diff** — run two pipelines (different models or configs) on the
  same conversation and diff the snapshots at each step.

- **Chroma browser** — paginated dump of the full warm/cold collections with a
  similarity search box (the current Chroma tab lists all memories but lacks
  pagination and search).

---

## Benchmarks / Evaluation

- **Automated score comparison** — add a CLI command or notebook that runs the
  LoCoMo benchmark end-to-end and outputs a structured report (JSON + Markdown
  table) comparing Active Memory vs baseline, so regression is visible across
  code changes.

- **NIAH sensitivity sweep** — run Needle-in-a-Haystack at multiple needle
  depths and context lengths to surface where retrieval starts to fail and
  confirm that the `similarity_threshold=0.3` default holds across datasets.

---

## Documentation / DX

- **Configuration reference** — add a `docs/configuration.md` listing every
  environment variable and constructor parameter with type, default, and effect.

- **Librarian usage guide** — document when and how to call
  `run_consolidation()`, `promote_frequent()`, `demote_stale()`, and
  `prune_old()` with example schedules.
