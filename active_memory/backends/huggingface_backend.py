"""
backends/huggingface_backend.py

LLM backend for the Hugging Face Inference API.

Lazy import: `huggingface_hub` is only imported when HuggingFaceBackend
is instantiated, so the package does not hard-require it to be installed
when another backend is in use.

Uses huggingface_hub.InferenceClient.chat_completion(), which supports
any model hosted on the Hugging Face Inference API that exposes a
chat-completion endpoint.
"""

from __future__ import annotations


_DEFAULT_MAX_TOKENS = 2048


class HuggingFaceBackend:
    """
    LLM backend for the Hugging Face Inference API.

    Parameters
    ----------
    model       Hugging Face model repo ID, e.g.
                "mistralai/Mistral-7B-Instruct-v0.3" or
                "HuggingFaceH4/zephyr-7b-beta".
    token       Hugging Face API token.  When None the SDK reads the
                HF_TOKEN environment variable automatically.
    max_tokens  Maximum tokens in the response.  Defaults to 2048.
    """

    def __init__(
        self,
        model: str,
        token: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        try:
            from huggingface_hub import InferenceClient as _InferenceClient
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceBackend requires the 'huggingface_hub' package. "
                "Install it with: pip install huggingface_hub"
            ) from exc

        self._model      = model
        self._max_tokens = max_tokens
        self._client     = _InferenceClient(model=model, token=token)

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict[str, str]]) -> str:
        """Send messages to the HF Inference API and return the response."""
        result = self._client.chat_completion(
            messages=messages,
            max_tokens=self._max_tokens,
        )
        return result.choices[0].message.content.strip()

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HuggingFaceBackend("
            f"model={self._model!r}, "
            f"max_tokens={self._max_tokens})"
        )
