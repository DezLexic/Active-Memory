# Active Memory

Active Memory is a Python library that gives conversational AI agents persistent, semantically searchable long-term memory.  As a conversation progresses a rolling summary captures what matters from older exchanges, and an eviction-triggered pipeline decides what to store permanently in a local [ChromaDB](https://www.trychroma.com/) vector store.  On every turn the agent receives a structured context window — system instructions, conversation summary, relevant memories retrieved by semantic search, and the most recent message pairs — letting it recall decisions made hundreds of turns ago without ever holding the full history in context.

Three LLM backends are supported out of the box: **Ollama** (local, free), **Anthropic Claude** (cloud), and **Hugging Face Inference API** (cloud).  Any object that implements `chat(messages) -> str` works as a drop-in backend.

---

## How It Works

Each call to `pipeline.chat()` runs this sequence:

```
User message
    │
    ▼
Retrieval ──── semantic search on warm + cold ChromaDB ──► inject top-3 memories into Bucket
    │
    ▼
ActiveAgent ── reads Bucket (instructions + summary + memories + recent) ──► response  ◄── user sees this
    │
    ▼
Bucket.push_message()
    │ eviction? (stack full)
    ├─ Yes ──► Observer  (1 LLM call) ── update rolling summary
    │      └─► Curator   (1 LLM call) ── decide what to store in ChromaDB
    └─ No  ──► done
```

| Component | Role | Model calls |
|-----------|------|-------------|
| **Pipeline** | Orchestrator | 0 |
| **ActiveAgent** | Generates the user-visible response | 1 per turn |
| **Bucket** | In-memory context window (summary + recent + memories) | 0 |
| **Retrieval** | Two-tier ChromaDB store (warm / cold) | 0 (vector search) |
| **Observer** | Updates the rolling summary when pairs are evicted | 1 per eviction batch |
| **Curator** | Decides which evicted pairs are worth storing permanently | 1 per eviction batch |
| **Librarian** | Promotes / demotes / prunes memories during downtime | 0 |
| **Scheduler** | Runs Librarian once per night in a background thread | 0 |

---

## Design Principles

- **Zero-config defaults** — `Pipeline()` with no arguments reads `.env` and falls back to a local Ollama instance.
- **Lazy imports** — provider packages (`ollama`, `anthropic`, `huggingface_hub`) are only imported when that backend is actually used.
- **Protocol-based backends** — any object with a `chat(messages: list[dict]) -> str` method works as a backend.
- **Two-tier memory** — warm memories (frequently retrieved) are queried first; cold memories (older / less used) fill remaining slots.
- **Predictable latency** — memory maintenance (Observer, Curator) runs sequentially after the response is returned; no hidden async work blocks the user.

---

## Requirements

- Python **3.10+**
- For local inference: [Ollama](https://ollama.com/) running with at least one model pulled

```bash
pip install -r requirements.txt
```

`requirements.txt` installs the core dependencies (`ollama`, `chromadb`, `python-dotenv`).  Cloud backend packages are optional:

```bash
pip install anthropic        # Anthropic Claude
pip install huggingface_hub  # Hugging Face Inference API
```

---

## Setup

### 1 — Clone and install

```bash
git clone <repo-url>
cd active-memory
pip install -r requirements.txt
```

### 2 — Configure your backend

Copy `.env.example` to `.env` and fill in the values for your chosen backend.

#### Ollama (local — default)

No API key required.  [Install Ollama](https://ollama.com/download), then pull a model:

```bash
ollama pull gemma3:4b
```

`.env`:
```ini
ACTIVE_MEMORY_BACKEND=ollama
ACTIVE_MEMORY_MODEL=gemma3:4b          # any model you have pulled
OLLAMA_BASE_URL=http://localhost:11434  # default — change if Ollama runs elsewhere
```

#### Anthropic Claude

```ini
ACTIVE_MEMORY_BACKEND=anthropic
ACTIVE_MEMORY_MODEL=claude-haiku-4-5   # or claude-sonnet-4-5, etc.
ANTHROPIC_API_KEY=sk-ant-...
ACTIVE_MEMORY_MAX_TOKENS=2048
```

#### Hugging Face Inference API

```ini
ACTIVE_MEMORY_BACKEND=huggingface
ACTIVE_MEMORY_MODEL=meta-llama/Llama-3.1-8B-Instruct   # required — no default
HF_TOKEN=hf_...
ACTIVE_MEMORY_MAX_TOKENS=2048
```

### 3 — Optional pipeline tuning

```ini
# Max Q/A pairs kept in the recent stack before eviction fires.
# Increase for faster ingestion of large logs (fewer evictions = fewer LLM calls).
ACTIVE_MEMORY_MAX_RECENT=20

# Pairs evicted at once when the stack is full.  Must be <= ACTIVE_MEMORY_MAX_RECENT.
ACTIVE_MEMORY_BATCH_REDUCTION=10
```

---

## Quick Start

### Ollama (local)

```python
from active_memory import Pipeline

pipeline = Pipeline()  # reads .env; defaults to OllamaBackend("gemma3:4b")

pipeline.chat("Let's build a REST API using Elixir and Phoenix.")
pipeline.chat("We need JWT authentication and a PostgreSQL database.")

response = pipeline.chat("What framework and database did we decide on?")
print(response)
# → "We decided on Elixir with the Phoenix framework and a PostgreSQL database..."
```

### Anthropic Claude

```python
from active_memory import Pipeline, AnthropicBackend

backend  = AnthropicBackend(model="claude-haiku-4-5")
pipeline = Pipeline(backend=backend)

pipeline.chat("We're using a blue/grey colour palette throughout the app.")
response = pipeline.chat("What colours should the dashboard use?")
print(response)
```

### Custom backend

```python
from active_memory import Pipeline

class MyBackend:
    def chat(self, messages: list[dict]) -> str:
        # call any LLM API here
        return "response"

pipeline = Pipeline(backend=MyBackend())
```

### Separate backends per role

Use a fast local model for the user-facing agent and a larger cloud model for higher-quality summarisation:

```python
from active_memory import Pipeline, OllamaBackend, AnthropicBackend

pipeline = Pipeline(
    backend=OllamaBackend("gemma3:4b"),
    observer_backend=AnthropicBackend("claude-sonnet-4-5"),
)
```

---

## API Reference

### Pipeline

```python
Pipeline(
    backend: LLMBackend | None = None,          # defaults to backend_from_env()
    chroma_path: str = "./chroma_db",
    max_recent_messages: int = 20,
    batch_reduction: int = 10,
    system_instructions: str | None = None,
    observer_backend: LLMBackend | None = None,
    curator_backend: LLMBackend | None = None,
)
```

| Method | Description |
|--------|-------------|
| `pipeline.chat(message, skip_observer=False) -> str` | Send a message; returns the agent's response. |
| `pipeline.ingest(question, response) -> None` | Pre-seed a Q/A pair without invoking the agent (batch loading). |
| `pipeline.bucket` | Read the current Bucket (summary, recent messages, injected memories). |
| `pipeline.retrieval` | Access the Retrieval store directly. |

### Memory maintenance — Librarian & Scheduler

```python
from active_memory import Pipeline
from active_memory.librarian import Librarian
from active_memory.scheduler import Scheduler

pipeline  = Pipeline()
librarian = Librarian(pipeline.retrieval)

# Run manually
stats = librarian.run_consolidation()
# → {"promoted": 2, "demoted": 1, "pruned": 0, "warm_remaining": 5, "cold_remaining": 12}

# Or schedule nightly at 02:00
scheduler = Scheduler(pipeline, run_hour=2)
scheduler.start()
# ...
scheduler.stop()
```

---

## Benchmarks

### Needle-in-a-Haystack (NIAH)

Tests whether Active Memory can surface a passkey buried at a configurable position inside a long synthetic conversation, compared to a vanilla agent that receives the full log as a system prompt.  For large logs (1 000+ pairs) the vanilla agent typically exceeds its context limit and fails; Active Memory answers from its compressed Bucket.

**Step 1 — generate a synthetic chat log**

```bash
python benchmarks/generate_chat.py \
    --pairs 100 \
    --passkey "ALPHA-7731-DELTA" \
    --position 0.3 \
    --output benchmarks/contexts/niah_chat_100.py
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pairs / -n` | 100 | Number of Q/A pairs to generate |
| `--batch-size` | 50 | Pairs per LLM call (avoids token-limit errors) |
| `--passkey` | *(none)* | Secret string injected at `--position` |
| `--position` | 0.5 | Fractional position of the passkey (0.0 = start, 1.0 = end) |
| `--timeout` | 300 | Seconds before an LLM call is aborted |
| `--retries` | 1 | Retry count per batch on failure |
| `--output / -o` | auto | Output file path |

**Step 2 — run the benchmark**

```bash
# Default file: benchmarks/contexts/niah_chat_100.py
python benchmarks/benchmark_niah.py

# Explicit file
python benchmarks/benchmark_niah.py benchmarks/contexts/niah_chat_1000.py

# Tune stack size for faster ingestion of large logs (set in .env or inline)
ACTIVE_MEMORY_MAX_RECENT=100 python benchmarks/benchmark_niah.py
```

The benchmark runs four phases — Vanilla context-stuffing → Active Memory ingestion → Active Memory recall → side-by-side results — and reports binary pass/fail (did the response contain the passkey?).

### Historical benchmarks

`benchmarks/benchmark_v1.py` through `benchmark_v4.py` track performance across successive pipeline versions.  See each file for its own description.

---

## Project Structure

```
active_memory/
├── pipeline.py           Main orchestrator — wires all components together
├── bucket.py             In-memory context window (no external deps)
├── retrieval.py          Two-tier ChromaDB vector store (warm / cold)
├── observer.py           Rolling summary updater (fires on eviction)
├── curator.py            Memory filter — decides what to store permanently
├── active_agent.py       User-facing conversational agent
├── librarian.py          Maintenance — promote / demote / prune memories
├── scheduler.py          Background thread — runs Librarian nightly
├── monitor.py            Hung-process detector — logs stalled LLM calls
├── config.py             backend_from_env() factory
└── backends/
    ├── base.py            LLMBackend protocol definition
    ├── ollama_backend.py
    ├── anthropic_backend.py
    └── huggingface_backend.py

benchmarks/
├── generate_chat.py      Synthetic NIAH chat log generator
├── benchmark_niah.py     Needle-in-a-Haystack benchmark
├── benchmark_v4.py       Latest historical benchmark
└── contexts/             Generated chat log files (.gitignored)

tests/                    pytest unit and integration tests
archive/                  Deprecated v1 pipeline components
```
