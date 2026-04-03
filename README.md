# Active Memory

Active Memory is a Python memory system for AI agents that run on local Ollama models. It gives a conversational agent persistent, semantically searchable memory without any cloud dependencies. As a conversation progresses, a rolling summary captures what matters from older exchanges, and an eviction-triggered pipeline decides what to store permanently in a local ChromaDB vector store. On every turn the agent receives a structured context window containing its system instructions, the conversation summary, the most recent message pairs, and any relevant memories retrieved from the store — allowing it to answer questions about decisions made many turns ago without ever having to keep the full history in context.

## Dependencies

- [Ollama](https://ollama.com/) running locally with a model pulled (default: `gemma3:4b`)
- Python 3.10+

## Install

```bash
pip install -r requirements.txt
```

## Run the pipeline test

```bash
python tests/test_pipeline.py
```

## Quick start

```python
from active_memory import Pipeline

pipeline = Pipeline(
    model="gemma3:4b",       # any model available in your local Ollama
    chroma_path="./chroma_db",
    max_recent_messages=5,
)

response = pipeline.chat("Let's build a backend API using Elixir and Phoenix.")
print(response)

response = pipeline.chat("What framework did we just decide on?")
print(response)
```

## Project structure

```
active_memory/   Core package — Bucket, Retrieval, Observer, Curator, ActiveAgent, Pipeline
tests/           End-to-end and unit tests for each component
benchmarks/      Historical benchmarks comparing the v1 pipeline against a vanilla agent
archive/         Deprecated v1 pipeline components kept for reference
chroma_db/       Default ChromaDB persistence directory (created on first run)
```
