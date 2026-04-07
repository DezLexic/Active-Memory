"""
config.py

Builds an LLMBackend from environment variables, loading a .env file
if one is present.

    from active_memory import backend_from_env
    backend = backend_from_env()

Variables read
--------------
ACTIVE_MEMORY_BACKEND    Provider to use: 'ollama' | 'anthropic' | 'huggingface'
                         Defaults to 'ollama' when not set.
ACTIVE_MEMORY_MODEL      Model identifier for the selected provider.
                         Falls back to each backend's built-in default when absent.
                         Required for HuggingFace (no sensible default exists).
ACTIVE_MEMORY_MAX_TOKENS Max response tokens for cloud providers. Default 2048.
OLLAMA_BASE_URL          Ollama server URL. Default http://localhost:11434.
ANTHROPIC_API_KEY        Anthropic API key. When absent the Anthropic SDK reads
                         this variable from the environment automatically.
HF_TOKEN                 HuggingFace API token. When absent the HF SDK reads
                         HF_TOKEN from the environment automatically.
"""

from __future__ import annotations

import os


def backend_from_env() -> object:
    """
    Read environment variables (and .env if present) and return the
    configured LLMBackend instance ready for use with Pipeline.

    Loading order
    -------------
    1. If python-dotenv is installed and a .env file exists in the
       current working directory, its values are loaded into os.environ.
    2. Environment variables already set in the OS take precedence over
       .env values (load_dotenv does not override existing vars).
    3. ACTIVE_MEMORY_BACKEND selects the provider; remaining variables
       are forwarded to the corresponding backend constructor.

    Raises
    ------
    ValueError
        If ACTIVE_MEMORY_BACKEND is set to an unrecognised value, or if
        ACTIVE_MEMORY_BACKEND=huggingface but ACTIVE_MEMORY_MODEL is not set.
    ImportError
        If the selected provider's package is not installed.
    """
    # Load .env if python-dotenv is available and a .env file exists.
    # This is a no-op when .env is absent, preserving normal CI and
    # containerised deployments that inject env vars via the OS.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed; env vars must come from the OS

    provider   = os.getenv("ACTIVE_MEMORY_BACKEND", "ollama").lower().strip()
    model      = os.getenv("ACTIVE_MEMORY_MODEL", "").strip() or None
    max_tokens = int(os.getenv("ACTIVE_MEMORY_MAX_TOKENS", "2048"))

    # ── Ollama ────────────────────────────────────────────────────────────────
    if provider == "ollama":
        from .backends.ollama_backend import OllamaBackend
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        kwargs: dict = {"base_url": base_url}
        if model:
            kwargs["model"] = model
        return OllamaBackend(**kwargs)

    # ── Anthropic ─────────────────────────────────────────────────────────────
    if provider == "anthropic":
        from .backends.anthropic_backend import AnthropicBackend
        kwargs = {"max_tokens": max_tokens}
        if model:
            kwargs["model"] = model
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            kwargs["api_key"] = api_key
        # No api_key kwarg → Anthropic SDK reads ANTHROPIC_API_KEY from env.
        return AnthropicBackend(**kwargs)

    # ── Hugging Face ──────────────────────────────────────────────────────────
    if provider in ("huggingface", "hf"):
        from .backends.huggingface_backend import HuggingFaceBackend
        if not model:
            raise ValueError(
                "ACTIVE_MEMORY_MODEL must be set when using the HuggingFace "
                "backend. Example: "
                "ACTIVE_MEMORY_MODEL=mistralai/Mistral-7B-Instruct-v0.3"
            )
        kwargs = {"model": model, "max_tokens": max_tokens}
        token = os.getenv("HF_TOKEN")
        if token:
            kwargs["token"] = token
        # No token kwarg → HF SDK reads HF_TOKEN from env.
        return HuggingFaceBackend(**kwargs)

    # ── Unknown ───────────────────────────────────────────────────────────────
    raise ValueError(
        f"Unknown ACTIVE_MEMORY_BACKEND={provider!r}. "
        f"Valid options: 'ollama', 'anthropic', 'huggingface'."
    )
