"""
active_memory.backends

Public API for LLM backend implementations.

Usage
-----
    from active_memory.backends import OllamaBackend
    from active_memory.backends import AnthropicBackend
    from active_memory.backends import HuggingFaceBackend
    from active_memory.backends import LLMBackend   # Protocol for type annotations

    # Or import directly from the top-level package:
    from active_memory import OllamaBackend, AnthropicBackend, HuggingFaceBackend

Provider packages are lazy-imported — only the backend(s) you actually
instantiate need their provider package installed.
"""

from .base                import LLMBackend
from .ollama_backend      import OllamaBackend
from .anthropic_backend   import AnthropicBackend
from .huggingface_backend import HuggingFaceBackend

__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "AnthropicBackend",
    "HuggingFaceBackend",
]
