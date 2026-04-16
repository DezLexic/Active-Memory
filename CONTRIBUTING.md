# Contributing to Active Memory

Thanks for your interest in contributing! Active Memory is in early beta, and contributions of all kinds are welcome — bug reports, documentation fixes, new backends, benchmarks, and core improvements.

## Ground rules

- Be respectful and constructive in all interactions.
- Open an issue before starting non-trivial work — it saves everyone time.
- Keep PRs focused. One change per PR. Refactors separate from features.
- Tests are required for new behaviour. Bug fixes should include a regression test.

## Development setup

```bash
git clone https://github.com/<your-fork>/active-memory.git
cd active-memory
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[all]"            # core + all provider extras
pip install -r requirements.txt    # dev/test deps
```

You'll need at least one backend usable for end-to-end testing:
- **Ollama** (no API key, free): install from [ollama.com](https://ollama.com), then `ollama pull gemma3:4b`
- **Anthropic**: set `ANTHROPIC_API_KEY` in a `.env` file (see `.env.example`)
- **HuggingFace**: set `HF_TOKEN` and `ACTIVE_MEMORY_MODEL`

## Running tests

```bash
pytest                                       # unit tests (~216, no network, ~40s)
pytest --cov=active_memory --cov-report=term-missing   # with coverage
```

Integration tests live in `tests/test_pipeline.py`, `tests/test_curator.py`, etc. and are excluded from the default run because they require a live backend. Run them manually after configuring `.env`:

```bash
python tests/test_pipeline.py
```

## Code style

- Python ≥3.10. Type hints encouraged but not required for tests.
- Follow the surrounding style — no auto-formatter is enforced, but keep it readable.
- Module-level `logger = logging.getLogger(__name__)` for any new component that can fail.
- Public API stays in `active_memory/__init__.py`. Don't expose internals by accident.

## Pull request checklist

- [ ] Tests added or updated, all passing locally (`pytest`)
- [ ] Updated `CHANGELOG.md` under `## [Unreleased]`
- [ ] Updated `README.md` if user-facing behaviour changed
- [ ] No new dependencies without discussion in an issue first

## Reporting bugs

Use the bug report template in `.github/ISSUE_TEMPLATE/`. Include:
- Python version and OS
- `pip freeze` output for the relevant packages
- Minimal reproduction (10 lines or fewer is ideal)
- Expected vs. actual behaviour

## Reporting security issues

Don't file public issues for security vulnerabilities. Open a private GitHub Security Advisory on the repository instead.

## Questions

Open a GitHub Discussion or issue with the `question` label.
