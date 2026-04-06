"""
active_memory

A Python memory system for AI agents with a provider-agnostic LLM backend.

Supported providers out of the box: Ollama, Anthropic Claude, Hugging Face.
Any object that implements chat(messages) -> str can be used as a backend.

Quick start
-----------
    from active_memory import Pipeline

    # Zero-config — defaults to OllamaBackend("gemma3:4b")
    pipeline = Pipeline()
    response = pipeline.chat("Hello, let's start building something.")

Provider examples
-----------------
    from active_memory import Pipeline, OllamaBackend, AnthropicBackend, HuggingFaceBackend

    # Explicit Ollama
    pipeline = Pipeline(backend=OllamaBackend(model="llama3:8b"))

    # Anthropic Claude (reads ANTHROPIC_API_KEY from env)
    pipeline = Pipeline(backend=AnthropicBackend(model="claude-haiku-4-5"))

    # Hugging Face (reads HF_TOKEN from env)
    pipeline = Pipeline(backend=HuggingFaceBackend(model="mistralai/Mistral-7B-Instruct-v0.3"))

    # Mixed roles — different models for agent vs bookkeeping
    pipeline = Pipeline(
        backend=AnthropicBackend(model="claude-sonnet-4-5"),
        observer_backend=OllamaBackend(model="gemma3:4b"),
        curator_backend=OllamaBackend(model="gemma3:4b"),
    )
"""

from .pipeline import Pipeline
from .backends import LLMBackend, OllamaBackend, AnthropicBackend, HuggingFaceBackend

__all__ = [
    "Pipeline",
    "LLMBackend",
    "OllamaBackend",
    "AnthropicBackend",
    "HuggingFaceBackend",
]
