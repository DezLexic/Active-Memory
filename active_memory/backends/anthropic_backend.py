"""
backends/anthropic_backend.py

LLM backend for the Anthropic Claude API.

Lazy import: `anthropic` is only imported when AnthropicBackend is
instantiated, so the package does not hard-require it to be installed
when another backend is in use.

System-message translation
--------------------------
Anthropic does not accept {"role": "system", ...} entries inside the
messages list.  This backend extracts all system-role messages, joins
their content with a newline, and passes the result as the top-level
`system` kwarg.  Remaining messages are forwarded as `messages`.

If no system message is present the `system` kwarg is omitted entirely
so Anthropic does not receive an empty string.
"""

from __future__ import annotations


_DEFAULT_MAX_TOKENS = 2048


class AnthropicBackend:
    """
    LLM backend for the Anthropic Claude API.

    Parameters
    ----------
    model       Anthropic model ID, e.g. "claude-haiku-4-5" or
                "claude-sonnet-4-5".
    api_key     Anthropic API key.  When None the SDK reads the
                ANTHROPIC_API_KEY environment variable automatically.
    max_tokens  Maximum tokens in the response.  Defaults to 2048.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        api_key: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicBackend requires the 'anthropic' package. "
                "Install it with: pip install anthropic"
            ) from exc

        self._model      = model
        self._max_tokens = max_tokens
        # api_key=None lets the SDK read ANTHROPIC_API_KEY from the environment.
        self._client     = _anthropic.Anthropic(api_key=api_key)

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Send messages to the Anthropic Messages API and return the response.

        Translates any system-role messages into Anthropic's top-level
        `system` parameter before dispatching the request.
        """
        system_parts: list[str] = [
            m["content"] for m in messages if m.get("role") == "system"
        ]
        non_system: list[dict[str, str]] = [
            m for m in messages if m.get("role") != "system"
        ]

        kwargs: dict = {
            "model":      self._model,
            "max_tokens": self._max_tokens,
            "messages":   non_system,
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)

        result = self._client.messages.create(**kwargs)
        return result.content[0].text.strip()

    # ── Dunder helpers ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AnthropicBackend("
            f"model={self._model!r}, "
            f"max_tokens={self._max_tokens})"
        )
