# conftest.py (project root)
#
# Script-style integration tests run module-level code that requires a live
# Ollama server.  Exclude them from pytest collection so `pytest` (and CI)
# only runs the proper unit tests.
#
# To run an integration script manually:
#   python tests/test_pipeline.py

collect_ignore = [
    "tests/test_active_agent.py",
    "tests/test_curator.py",
    "tests/test_observer.py",
    "tests/test_pipeline.py",
]
