"""
backends/base.py

Structural Protocol for LLM backends.

Any object that implements:

    chat(messages: list[dict[str, str]]) -> str

is a valid LLMBackend.  Third parties can write conforming backends without
importing from this package — no inheritance required.

@runtime_checkable enables isinstance(obj, LLMBackend) checks in Pipeline
for early validation without static typing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """
    Uniform call surface for all LLM providers.

    Implementors receive a flat list of messages in OpenAI/Ollama format and
    return the model's response as a plain stripped string.  Provider-specific
    details — API keys, base URLs, max_tokens, system-message translation —
    are the backend's responsibility and must not leak into callers.

    Parameters
    ----------
    messages    List of message dicts:
                [{"role": "system"|"user"|"assistant", "content": str}, ...]
                Backends handle any translation required by their provider
                (e.g. Anthropic separates system content into a top-level kwarg).

    Returns
    -------
    str         The model's response, stripped of leading/trailing whitespace.
    """

    def chat(self, messages: list[dict[str, str]]) -> str:
        ...

    def __repr__(self) -> str:
        ...
