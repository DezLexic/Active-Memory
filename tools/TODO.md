# tools/ — Roadmap

## inspector/

Live LoCoMo benchmark inspection UI. See `inspector/server.py` for current
capabilities.

### Planned

- **Step mode** — advance one pair / one question at a time from the UI
  (button-driven, no automatic loop).
- **Custom benchmark** — type a question into the UI against a paused
  pipeline; see retrieved memories without polluting state.
- **Export trace** — download the full per-pair / per-question snapshot
  history as JSON for offline analysis.
- **Side-by-side** — run two pipelines (different models or configs) on
  the same conversation and diff the snapshots.
- **Chroma browser** — paginated dump of the full warm/cold collections,
  with similarity search box.
