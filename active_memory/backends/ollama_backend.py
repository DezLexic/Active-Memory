"""
backends/ollama_backend.py

LLM backend for a locally-running Ollama instance.

Lazy import: `ollama` is only imported when OllamaBackend is instantiated,
so the package does not hard-require it to be installed when another
backend is in use.
"""

from __future__ import annotations


class OllamaBackend:
    """
    LLM backend for Ollama.

    Parameters
    ----------
    model       Ollama model tag, e.g. "gemma3:4b" or "llama3:8b".
    base_url    HTTP base URL of the Ollama server.
                Defaults to "http://localhost:11434".
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        try:
            import ollama as _ollama
        except ImportError as exc:
            raise ImportError(
                "OllamaBackend requires the 'ollama' package. "
                "Install it with: pip install ollama"
            ) from exc

        self._model    = model
        self._base_url = base_url
        self._client   = _ollama.Client(host=base_url)

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send messages to Ollama and return the response text."""
        result = self._client.chat(
            model=self._model,
            messages=messages,
        )
        return result["message"]["content"].strip()

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"OllamaBackend("
            f"model={self._model!r}, "
            f"base_url={self._base_url!r})"
        )
