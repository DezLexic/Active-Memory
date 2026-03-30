from librarian import Librarian
from reporter import Reporter

librarian = Librarian()
reporter = Reporter(librarian)

memories = [
    ("warm", "The user wants no external network calls — the entire system must work without internet."),
    ("warm", "The user chose SQLite as the database because it requires no server setup."),
    ("warm", "The user gets frustrated when responses are too long and asks for brevity."),
    ("cold", "The project is a personal finance tracker to log income and expenses."),
    ("cold", "Python was selected as the language because the user is a beginner and finds it readable."),
    ("cold", "The user has about one hour per day to spend learning and building."),
]

print("=== Storing memories ===")
for tier, content in memories:
    memory_id = librarian.store(content, tier)
    print(f"  [{tier.upper()}] {content}")
print()

# Queries use different words from what was stored to test semantic retrieval
associative_query = "Does this project have any connectivity requirements?"
deliberate_query = "What do I know about how this person likes to communicate and learn?"

print(f"=== Associative recall ===")
print(f"  Query: \"{associative_query}\"")
results = reporter.associative_recall(associative_query, n_results=3)
for r in results:
    print(f"  -> {r}")
print()

print(f"=== Deliberate lookup ===")
print(f"  Query: \"{deliberate_query}\"")
results = reporter.deliberate_lookup(deliberate_query, n_results=5)
for r in results:
    print(f"  -> {r}")
print()

print("=== Retrieval logs ===")
logs = librarian.get_retrieval_logs()
for memory_id, count in logs.items():
    print(f"  {memory_id}: recalled {count}x")
