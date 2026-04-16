# Active Memory - A context memory layer for AI agents

Human memory doesn't work by replaying everything that ever happened. It maintains a running sense of context, surfaces what's relevant when it's needed, and lets the rest fade into the background — until something brings it back.

Active Memory brings that model to AI agents. It was built with one use case in mind: AI companions and long-running agents that should feel like they know you over time, not like they're meeting you fresh every session.

The practical problem it solves: most agents either run out of context window after a long conversation, or start every session with no memory of the last one. Active Memory sits between your code and your model call, quietly maintaining a structured picture of what matters — without changing your model, your stack, or your prompts.

## How it works

Four layers, modeled loosely on how memory is theorized to function in humans:

- **Observer** — maintains a structured topic summary, updated as the conversation evolves
- **Retrieval** — semantically indexes important decisions and surfaces them when relevant
- **Recent window** — keeps the last N exchanges verbatim
- **Librarian** — runs during downtime, consolidating memories by frequency and promoting what keeps coming up — modeled on the role sleep is thought to play in human memory consolidation

On every turn the first three are assembled into a single enriched context you pass to your own model. The Librarian works quietly in the background.

## Install

Install directly from GitHub (PyPI release coming later):

```bash
pip install "active-memory[anthropic] @ git+https://github.com/DezLexic/Active-Memory.git"
```

Pick the extra that matches your provider:

```bash
pip install "active-memory[anthropic]   @ git+https://github.com/DezLexic/Active-Memory.git"
pip install "active-memory[ollama]      @ git+https://github.com/DezLexic/Active-Memory.git"
pip install "active-memory[huggingface] @ git+https://github.com/DezLexic/Active-Memory.git"
pip install "active-memory[all]         @ git+https://github.com/DezLexic/Active-Memory.git"
pip install "active-memory[inspector]   @ git+https://github.com/DezLexic/Active-Memory.git"
```

## Quick start

Active Memory slots into your existing agent loop. You keep full control of the model call.

**Before** -- your existing code:

```python
context = build_my_context(history)
response = anthropic_client.messages.create(
    model="claude-sonnet-4-5",
    messages=context,
)
```

**After** -- add Active Memory:

```python
from active_memory import Pipeline
from active_memory.backends import AnthropicBackend

pipeline = Pipeline(backend=AnthropicBackend())  # used by Observer + Curator only

context = pipeline.build_context(user_message)   # summary + memories + recent history
response = anthropic_client.messages.create(
    model="claude-sonnet-4-5",
    messages=context,
)
pipeline.update(user_message, response.content[0].text)  # maintain memory + summary
```

`build_context()` returns a standard `[{"role": "system", ...}, {"role": "user", ...}]` message list. `update()` handles eviction, summary updates, and memory storage in the background. Your model, your prompts, your response handling -- Active Memory just enriches the context.

## Full loop

If you want Active Memory to own the model call too:

```python
from active_memory import Pipeline
from active_memory.backends import AnthropicBackend

pipeline = Pipeline(backend=AnthropicBackend())

pipeline.chat("We're building with Elixir and Phoenix on GKE.")
pipeline.chat("Database is CockroachDB, raw SQL only, no ORM.")
response = pipeline.chat("What stack did we decide on?")
```

## Inspector — see memory working live

Active Memory ships with a dashboard that runs the full pipeline against a real long-conversation benchmark (LoCoMo) and streams every step into a browser: pairs ingesting, the topic tree growing, memories landing in Chroma, and questions being answered with their retrieved context.

It's the fastest way to *see* what each layer is doing — and to debug your own configuration before you wire Active Memory into anything real.

```bash
pip install "active-memory[inspector] @ git+https://github.com/DezLexic/Active-Memory.git"

# Get the LoCoMo dataset (one-time)
git clone https://github.com/snap-research/locomo benchmarks/locomo
# Make sure benchmarks/locomo/data/locomo10.json exists.

# Run the dashboard
uvicorn tools.inspector.server:app --reload --port 8000
```

Open `http://localhost:8000` and pick a conversation. Defaults to a local Ollama backend (`gemma3:4b` on `http://localhost:11434`); edit `tools/inspector/server.py` to change models or providers.

What you'll see:

- **Conversation picker** — choose a LoCoMo session
- **Live ingest** — message pairs flowing into the Bucket, Observer summarizing on each eviction
- **Memory store** — warm/cold Chroma counts updating as the Curator stores decisions
- **QA phase** — the benchmark's annotated questions answered against the built-up memory, with retrieved snippets and a 1–5 score per answer

Status: working but minimal. Planned features (step-mode, custom questions, trace export, side-by-side diffs, Chroma browser) are tracked in `tools/TODO.md`.

## Configuration

Set `ACTIVE_MEMORY_BACKEND` in your environment or a `.env` file:

| Provider | Env vars |
|----------|----------|
| **Anthropic** | `ACTIVE_MEMORY_BACKEND=anthropic` `ANTHROPIC_API_KEY=sk-ant-...` |
| **Ollama** | `ACTIVE_MEMORY_BACKEND=ollama` `OLLAMA_BASE_URL=http://localhost:11434` |
| **Hugging Face** | `ACTIVE_MEMORY_BACKEND=huggingface` `ACTIVE_MEMORY_MODEL=meta-llama/...` `HF_TOKEN=hf_...` |

Optional tuning:

```ini
ACTIVE_MEMORY_MODEL=claude-haiku-4-5   # override default model for any provider
ACTIVE_MEMORY_MAX_RECENT=20            # recent message pairs before eviction
ACTIVE_MEMORY_BATCH_REDUCTION=10       # pairs evicted per batch
```

If no backend is configured, `backend_from_env()` raises a `ValueError` with setup instructions.

## Project structure

```
active_memory/    Core package (Pipeline, Bucket, Retrieval, Observer, Curator, backends)
tests/            Unit and integration tests (pytest)
benchmarks/       NIAH benchmark and stress tests
tools/inspector/  Live dashboard (FastAPI) for visualizing memory during a benchmark run
```
