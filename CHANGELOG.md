# Changelog

All notable changes to Active Memory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-15

Initial public beta.

### Added

- Core memory pipeline: `Pipeline`, `Bucket`, `Observer`, `Curator`, `Retrieval`, `Librarian`.
- Two-tier vector store (warm/cold) with metadata-driven promotion/demotion.
- Topic-based summary hierarchy maintained by `Observer`.
- Pluggable LLM backends: Anthropic, Ollama, HuggingFace.
- Public API: `Pipeline.chat()` (full loop) and `build_context()` + `update()` (split API).
- Configuration via environment variables / `.env` (`backend_from_env()`).
- Test suite: 216 unit tests across 19 modules, CI on Python 3.10/3.11/3.12.
- Benchmarks: NIAH (Needle in a Haystack) and LoCoMo v1–v4.
- `tools/inspector/` — experimental FastAPI UI for live benchmark inspection.
- MIT license, contributing guide, GitHub issue/PR templates.

### Known limitations

- No async/concurrent call support in the public API.
- `Librarian` consolidation requires manual invocation (`run_consolidation()`); the
  built-in scheduler is stubbed.
- Inspector UI is minimal — see `tools/TODO.md` for planned features.
- OpenAI backend is not yet implemented; planned for a future release.
