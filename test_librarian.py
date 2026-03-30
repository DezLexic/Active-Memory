from librarian import Librarian

librarian = Librarian()

memories = [
    ("warm", "The user decided to build a personal finance tracker using Python and SQLite."),
    ("warm", "The user prefers short, concise answers without long explanations."),
    ("warm", "The app must run fully offline — no external API calls allowed."),
    ("cold", "The user is a beginner who has never coded before and is just getting started."),
    ("cold", "Python was recommended as the best first language due to its readability."),
]

print("=== Storing memories ===")
stored_ids = []
for tier, content in memories:
    memory_id = librarian.store(content, tier)
    stored_ids.append(memory_id)
    print(f"  [{tier.upper()}] {memory_id}")
    print(f"         {content}")
print()

queries = [
    ("warm", "What are the technical constraints for this project?"),
    ("warm", "How does the user like to receive information?"),
    ("cold", "What is the user's experience level with programming?"),
]

print("=== Retrieving memories ===")
for tier, query in queries:
    print(f"  QUERY [{tier.upper()}]: {query}")
    results = librarian.retrieve(query, tier, n_results=2)
    for result in results:
        print(f"    -> {result}")
    print()

print("=== Retrieval logs ===")
logs = librarian.get_retrieval_logs()
for memory_id, count in logs.items():
    print(f"  {memory_id}: recalled {count}x")
