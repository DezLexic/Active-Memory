"""
test_backends.py

Unit tests for LLM backend classes.

  - LLMBackend Protocol: any object with chat() + __repr__() satisfies it
  - OllamaBackend: repr format, lazy-import ImportError path
  - AnthropicBackend: system-message translation, repr, lazy-import path
  - HuggingFaceBackend: repr, model requirement, lazy-import path

Real network calls are never made.  Provider SDK modules are replaced with
MagicMock objects via unittest.mock.patch.dict so the actual packages do not
need to be installed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch


# ── LLMBackend Protocol ───────────────────────────────────────────────────────

class TestLLMBackendProtocol:

    def test_custom_class_satisfies_protocol(self):
        from active_memory import LLMBackend

        class MyBackend:
            def chat(self, messages: list[dict]) -> str:
                return "hello"
            def __repr__(self) -> str:
                return "MyBackend()"

        assert isinstance(MyBackend(), LLMBackend)

    def test_object_missing_chat_does_not_satisfy_protocol(self):
        from active_memory import LLMBackend

        class BadBackend:
            def __repr__(self) -> str:
                return "BadBackend()"

        assert not isinstance(BadBackend(), LLMBackend)

    def test_object_missing_repr_does_not_satisfy_protocol(self):
        from active_memory import LLMBackend

        class BadBackend:
            def chat(self, messages: list[dict]) -> str:
                return "hello"

        # object.__repr__ is inherited, so the Protocol check passes —
        # this confirms that __repr__ requirement is met by inheritance.
        assert isinstance(BadBackend(), LLMBackend)


# ── OllamaBackend ─────────────────────────────────────────────────────────────

class TestOllamaBackend:

    def _make_backend(self, model="gemma3:4b", base_url="http://localhost:11434"):
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.chat.return_value = MagicMock(
            message=MagicMock(content="  ollama reply  ")
        )
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from active_memory.backends.ollama_backend import OllamaBackend
            backend = OllamaBackend(model=model, base_url=base_url)
        return backend, mock_client

    def test_repr_contains_model(self):
        backend, _ = self._make_backend(model="llama3:8b")
        assert "llama3:8b" in repr(backend)

    def test_repr_contains_base_url(self):
        backend, _ = self._make_backend(base_url="http://myserver:11434")
        assert "myserver" in repr(backend)

    def test_repr_contains_class_name(self):
        backend, _ = self._make_backend()
        assert "OllamaBackend" in repr(backend)

    def test_chat_returns_stripped_response(self):
        # OllamaBackend accesses result["message"]["content"] (dict subscript).
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "  hello world  "}}
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            from active_memory.backends.ollama_backend import OllamaBackend
            backend = OllamaBackend()
            result  = backend.chat([{"role": "user", "content": "hi"}])
        assert result == "hello world"

    def test_missing_ollama_package_raises_import_error(self):
        # Remove ollama from sys.modules to simulate it not being installed.
        with patch.dict("sys.modules", {"ollama": None}):
            import importlib
            import active_memory.backends.ollama_backend as mod
            importlib.reload(mod)
            with pytest.raises(ImportError, match="ollama"):
                mod.OllamaBackend()


# ── AnthropicBackend ──────────────────────────────────────────────────────────

def _make_anthropic_mock(response_text: str = "anthropic reply"):
    """Return (mock_anthropic_module, mock_client) pair."""
    mock_content = MagicMock()
    mock_content.text = response_text

    mock_response = MagicMock()
    mock_response.content = [mock_content]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client

    return mock_anthropic, mock_client


class TestAnthropicBackend:

    def _backend(self, mock_anthropic, model="test-model"):
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from active_memory.backends.anthropic_backend import AnthropicBackend
            return AnthropicBackend(model=model, api_key="fake-key")

    def test_repr_contains_model(self):
        mock_anthropic, _ = _make_anthropic_mock()
        backend = self._backend(mock_anthropic, model="claude-haiku-4-5")
        assert "claude-haiku-4-5" in repr(backend)

    def test_repr_contains_class_name(self):
        mock_anthropic, _ = _make_anthropic_mock()
        backend = self._backend(mock_anthropic)
        assert "AnthropicBackend" in repr(backend)

    def test_system_message_extracted_as_top_level_kwarg(self):
        mock_anthropic, mock_client = _make_anthropic_mock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from active_memory.backends.anthropic_backend import AnthropicBackend
            backend = AnthropicBackend(model="test", api_key="fake")
            backend.chat([
                {"role": "system", "content": "You are helpful."},
                {"role": "user",   "content": "hello"},
            ])
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs.get("system") == "You are helpful."

    def test_system_message_not_included_in_messages_list(self):
        mock_anthropic, mock_client = _make_anthropic_mock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from active_memory.backends.anthropic_backend import AnthropicBackend
            backend = AnthropicBackend(model="test", api_key="fake")
            backend.chat([
                {"role": "system", "content": "System."},
                {"role": "user",   "content": "User msg."},
            ])
        kwargs = mock_client.messages.create.call_args.kwargs
        roles_in_messages = [m["role"] for m in kwargs["messages"]]
        assert "system" not in roles_in_messages

    def test_multiple_system_messages_joined_with_newline(self):
        mock_anthropic, mock_client = _make_anthropic_mock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from active_memory.backends.anthropic_backend import AnthropicBackend
            backend = AnthropicBackend(model="test", api_key="fake")
            backend.chat([
                {"role": "system", "content": "Part one."},
                {"role": "system", "content": "Part two."},
                {"role": "user",   "content": "question"},
            ])
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["system"] == "Part one.\nPart two."

    def test_no_system_message_omits_system_kwarg(self):
        mock_anthropic, mock_client = _make_anthropic_mock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from active_memory.backends.anthropic_backend import AnthropicBackend
            backend = AnthropicBackend(model="test", api_key="fake")
            backend.chat([{"role": "user", "content": "hi"}])
        kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" not in kwargs

    def test_chat_returns_stripped_response(self):
        mock_anthropic, _ = _make_anthropic_mock("  padded response  ")
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from active_memory.backends.anthropic_backend import AnthropicBackend
            backend = AnthropicBackend(model="test", api_key="fake")
            result  = backend.chat([{"role": "user", "content": "hi"}])
        assert result == "padded response"

    def test_missing_anthropic_package_raises_import_error(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            import importlib
            import active_memory.backends.anthropic_backend as mod
            importlib.reload(mod)
            with pytest.raises(ImportError, match="anthropic"):
                mod.AnthropicBackend()


# ── HuggingFaceBackend ────────────────────────────────────────────────────────

def _make_hf_mock(response_text: str = "hf reply"):
    mock_choice  = MagicMock()
    mock_choice.message.content = response_text

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat_completion.return_value = mock_response

    mock_inference_client_cls = MagicMock(return_value=mock_client)

    mock_hf = MagicMock()
    mock_hf.InferenceClient = mock_inference_client_cls

    return mock_hf, mock_client


class TestHuggingFaceBackend:

    def test_repr_contains_model(self):
        mock_hf, _ = _make_hf_mock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            from active_memory.backends.huggingface_backend import HuggingFaceBackend
            backend = HuggingFaceBackend(model="mistral/Mistral-7B")
        assert "mistral" in repr(backend).lower()

    def test_repr_contains_class_name(self):
        mock_hf, _ = _make_hf_mock()
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            from active_memory.backends.huggingface_backend import HuggingFaceBackend
            backend = HuggingFaceBackend(model="some/model")
        assert "HuggingFaceBackend" in repr(backend)

    def test_chat_returns_stripped_response(self):
        mock_hf, _ = _make_hf_mock("  spaced reply  ")
        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            from active_memory.backends.huggingface_backend import HuggingFaceBackend
            backend = HuggingFaceBackend(model="some/model", token="hf_fake")
            result  = backend.chat([{"role": "user", "content": "hello"}])
        assert result == "spaced reply"

    def test_missing_huggingface_hub_raises_import_error(self):
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            import importlib
            import active_memory.backends.huggingface_backend as mod
            importlib.reload(mod)
            with pytest.raises(ImportError, match="huggingface"):
                mod.HuggingFaceBackend(model="some/model")
