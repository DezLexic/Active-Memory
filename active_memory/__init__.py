"""
active_memory

A Python memory system for AI agents using local Ollama models and ChromaDB.
Import the Pipeline class to get started:

    from active_memory import Pipeline

    pipeline = Pipeline()
    response = pipeline.chat("Hello, let's start building something.")
"""

from .pipeline import Pipeline

__all__ = ["Pipeline"]
