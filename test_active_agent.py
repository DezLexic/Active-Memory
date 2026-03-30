from observer import Observer
from curator import Curator
from librarian import Librarian
from reporter import Reporter
from conductor import Conductor
from active_agent import ActiveAgent

# A 22-message conversation that:
# - Establishes clear architectural decisions in the first 8 messages
# - Continues building detail through the middle
# - Asks the agent to recall specific early decisions near the end
# This tests whether the assembled context window keeps the agent consistent
# even after the raw early messages have been trimmed from the Observer's summary.
USER_TURNS = [
    # --- Early decisions ---
    "Let's design a backend API for a personal finance tracker.",
    "What language should we use? I'm leaning toward Go.",
    "Actually, let's use Python. FastAPI specifically. That's the decision.",
    "For the database, I want PostgreSQL. No SQLite, no MongoDB.",
    "And no ORM. I want raw SQL with psycopg2. That's a hard constraint.",
    "Authentication should use JWT. No sessions, no cookies.",
    "One more constraint: this must be deployable on a single VPS. No Kubernetes, no cloud-managed services.",
    "Okay, those are the foundations. Let's start talking about the data model.",
    # --- Building detail ---
    "The main entities are: User, Account, Transaction, and Category.",
    "Transactions should belong to an Account, and each Transaction can have one Category.",
    "Users can have multiple Accounts — checking, savings, credit cards, etc.",
    "Let's add a Budget entity too. A Budget ties a Category to a monthly spending limit.",
    "What indexes would you recommend on the transactions table?",
    "Should we use UUIDs or integer IDs for primary keys?",
    "Let's go with UUIDs for all primary keys. Easier to sync if we ever add a mobile client.",
    "Okay walk me through the transactions table schema with all the columns we've decided on.",
    # --- Drift zone: enough messages to push early decisions out of raw context ---
    "What's a good way to handle currency — store as float or integer cents?",
    "Integer cents it is. Less rounding risk.",
    "Let's talk about the API structure. Should we version the endpoints?",
    "Yes, prefix everything with /api/v1/. Good practice.",
    # --- Recall challenge: ask about early decisions ---
    "Remind me — what database and ORM approach did we lock in at the start?",
    "And what was the deployment constraint we agreed on?",
]

def separator(label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

def main():
    observer  = Observer()
    curator   = Curator()
    librarian = Librarian()
    reporter  = Reporter(librarian)
    conductor = Conductor(observer, curator, librarian, reporter)
    agent     = ActiveAgent(conductor)

    for i, user_msg in enumerate(USER_TURNS, start=1):
        separator(f"Turn {i:02d}")
        print(f"USER: {user_msg}\n")

        reply = agent.chat(user_msg)
        print(f"AGENT: {reply}")

        if observer.recall_trigger:
            print("\n  [recall_trigger fired]")

        if observer.trimmings:
            print(f"\n  [trimmings: {len(observer.trimmings)} total]")

    separator("Explicit Lookup — 'database and ORM constraints'")
    print(agent.lookup("database and ORM constraints"))

    separator("Explicit Lookup — 'deployment VPS constraint'")
    print(agent.lookup("deployment VPS constraint"))

    separator("Retrieval Log")
    print(librarian.get_retrieval_logs())

if __name__ == "__main__":
    main()
