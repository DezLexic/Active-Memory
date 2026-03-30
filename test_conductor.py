from observer import Observer
from curator import Curator
from librarian import Librarian
from reporter import Reporter
from conductor import Conductor

# A 15-message conversation designed to exercise the full pipeline:
# - enough messages to push the Observer past 200 words and generate trimmings
# - decisions and preferences so the Curator has things worth storing
# - questions and recall keywords to trigger associative recall
CONVERSATION = [
    ("user",      "Hey, let's build a personal task manager app. I want it to run in the terminal."),
    ("assistant", "Sounds good. Terminal-based, lightweight. Any language preference?"),
    ("user",      "Python. I've decided I want to keep dependencies minimal — no heavy frameworks."),
    ("assistant", "Understood. We'll stick to the standard library where possible and only pull in small focused packages when necessary."),
    ("user",      "One constraint: everything must work offline. No cloud sync, no external APIs."),
    ("assistant", "Offline-first it is. Data will live in a local file — JSON or SQLite, whichever you prefer."),
    ("user",      "Let's use SQLite. I prefer it over JSON for anything that grows over time."),
    ("assistant", "SQLite it is. I'll set up a simple schema: tasks table with id, title, status, and created_at."),
    ("user",      "Good. Also, I want tasks to support tags so I can filter by project or context."),
    ("assistant", "I'll add a tags column — comma-separated strings for simplicity, or a separate tags table if you want proper querying."),
    ("user",      "Separate tags table. Do it properly. What did we decide about the language again?"),
    ("assistant", "Python, standard library preferred, minimal dependencies. SQLite for storage with a proper tags table."),
    ("user",      "Right. And remind me — what was the constraint we agreed on earlier about connectivity?"),
    ("assistant", "Offline-first: no cloud sync, no external APIs. Everything runs and persists locally."),
    ("user",      "Perfect. Let's start with the schema. Can you show me the SQLite table definitions?"),
]

def main():
    observer   = Observer()
    curator    = Curator()
    librarian  = Librarian()
    reporter   = Reporter(librarian)
    conductor  = Conductor(observer, curator, librarian, reporter)

    for i, (role, message) in enumerate(CONVERSATION, start=1):
        print(f"\n{'='*60}")
        print(f"MSG {i:02d} [{role.upper()}]: {message}")
        print(f"{'='*60}")

        context = conductor.process_message(message, role)

        print(f"\n--- Observer Summary ---")
        print(context)

        if observer.recall_trigger:
            print(f"\n[recall_trigger=True]")

        if observer.trimmings:
            print(f"\n--- Trimmings so far ({len(observer.trimmings)}) ---")
            for j, t in enumerate(observer.trimmings, start=1):
                print(f"  [{j}] {t[:120]}{'...' if len(t) > 120 else ''}")

    print(f"\n{'='*60}")
    print("--- Retrieval Logs ---")
    print(librarian.get_retrieval_logs())

    print("\n--- Explicit Lookup: 'offline constraint' ---")
    results = conductor.explicit_lookup("offline constraint")
    for r in results:
        print(f"  • {r}")

    print("\n--- run_consolidation ---")
    conductor.run_consolidation()

if __name__ == "__main__":
    main()
