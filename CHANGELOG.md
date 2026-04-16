# Changelog

All notable changes to Active Memory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Observer topic nodes now use typed slots.** The single prose `summary`
  field on each node has been replaced with five typed slots: `facts`,
  `decisions`, `preferences`, `open_threads`, and `quotes`. Each slot is a
  list of concrete items, capped at 200 characters each. Context strings
  rendered to the Active Agent preserve specific detail (names, places,
  numbers, verbatim quotes) instead of collapsing into category labels.
- Inspector "Topics" tab now renders topic nodes as structured cards with
  per-slot labels and bullet lists, rather than dumping raw JSON.
- Legacy `Bucket.summary` setter still works — the incoming prose is stored
  as a single entry in the `facts` slot so old callers continue to render.
- Inspector now configures Python logging so silent Curator warnings
  (LLM call failures, JSON parse errors) actually reach stderr, and
  surfaces every Curator decision (`stored (warm)` / `dropped`) plus a
  per-question retrieval summary (`retrieved N mems, top sim=0.XX`) in
  the rolling log feed.
- Inspector Chroma tab now lists every stored memory with its tier,
  timestamp, and retrieval count — previously a TODO stub showing only
  a count.
- Lowered default `Retrieval.similarity_threshold` from `0.7` to `0.3`
  to match the realistic behaviour of Chroma's default MiniLM embedder on
  paraphrased queries. Tests pin their own thresholds explicitly, so
  this is a default-only change with no test impact.

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
