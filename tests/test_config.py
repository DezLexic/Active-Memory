"""
test_config.py

Unit tests for active_memory/config.py — backend_from_env() provider
selection and error paths.

Patches os.environ so tests never require real API keys or a running
Ollama server.  Uses monkeypatch to isolate each test from leaked state.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib
import pytest


def _reload_config():
    """Force config module to re-read env vars on every call."""
    import active_memory.config as cfg
    importlib.reload(cfg)
    return cfg


# ── Default provider ──────────────────────────────────────────────────────────

class TestDefaultProvider:

    def test_no_backend_env_var_uses_ollama(self, monkeypatch):
        monkeypatch.delenv("ACTIVE_MEMORY_BACKEND", raising=False)
        monkeypatch.delenv("ACTIVE_MEMORY_MODEL",   raising=False)
        from active_memory.config import backend_from_env
        backend = backend_from_env()
        assert "OllamaBackend" in repr(backend)

    def test_explicit_ollama_uses_ollama(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MEMORY_BACKEND", "ollama")
        monkeypatch.delenv("ACTIVE_MEMORY_MODEL",   raising=False)
        from active_memory.config import backend_from_env
        backend = backend_from_env()
        assert "OllamaBackend" in repr(backend)

    def test_ollama_base_url_is_forwarded(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MEMORY_BACKEND", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://myserver:11434")
        from active_memory.config import backend_from_env
        backend = backend_from_env()
        assert "myserver" in repr(backend)

    def test_ollama_custom_model_is_forwarded(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MEMORY_BACKEND", "ollama")
        monkeypatch.setenv("ACTIVE_MEMORY_MODEL",   "llama3:8b")
        from active_memory.config import backend_from_env
        backend = backend_from_env()
        assert "llama3" in repr(backend)


# ── Error paths ───────────────────────────────────────────────────────────────

class TestErrorPaths:

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MEMORY_BACKEND", "unknownprovider")
        from active_memory.config import backend_from_env
        with pytest.raises(ValueError, match="unknownprovider"):
            backend_from_env()

    def test_huggingface_without_model_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MEMORY_BACKEND", "huggingface")
        monkeypatch.delenv("ACTIVE_MEMORY_MODEL", raising=False)
        from active_memory.config import backend_from_env
        with pytest.raises(ValueError, match="ACTIVE_MEMORY_MODEL"):
            backend_from_env()

    def test_hf_alias_without_model_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MEMORY_BACKEND", "hf")
        monkeypatch.delenv("ACTIVE_MEMORY_MODEL", raising=False)
        from active_memory.config import backend_from_env
        with pytest.raises(ValueError):
            backend_from_env()

    def test_value_error_message_includes_valid_options(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MEMORY_BACKEND", "badprovider")
        from active_memory.config import backend_from_env
        with pytest.raises(ValueError) as exc_info:
            backend_from_env()
        msg = str(exc_info.value)
        assert "ollama" in msg
        assert "anthropic" in msg
        assert "huggingface" in msg


# ── Case insensitivity ────────────────────────────────────────────────────────

class TestCaseInsensitivity:

    def test_backend_value_is_lowercased(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MEMORY_BACKEND", "OLLAMA")
        monkeypatch.delenv("ACTIVE_MEMORY_MODEL", raising=False)
        from active_memory.config import backend_from_env
        backend = backend_from_env()
        assert "OllamaBackend" in repr(backend)

    def test_mixed_case_backend_value(self, monkeypatch):
        monkeypatch.setenv("ACTIVE_MEMORY_BACKEND", "Ollama")
        monkeypatch.delenv("ACTIVE_MEMORY_MODEL", raising=False)
        from active_memory.config import backend_from_env
        backend = backend_from_env()
        assert "OllamaBackend" in repr(backend)
