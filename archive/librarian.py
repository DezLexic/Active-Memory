import uuid
import chromadb


class Librarian:
    def __init__(self, path: str = "./chroma_store"):
        self._client = chromadb.PersistentClient(path=path)
        self._collections = {
            "warm": self._client.get_or_create_collection("warm_memory"),
            "cold": self._client.get_or_create_collection("cold_memory"),
        }
        self._retrieval_log: dict[str, int] = {}

    def store(self, content: str, tier: str) -> str:
        memory_id = str(uuid.uuid4())
        self._collections[tier].add(
            ids=[memory_id],
            documents=[content],
        )
        return memory_id

    def retrieve(self, query: str, tier: str, n_results: int = 3) -> list[str]:
        collection = self._collections[tier]
        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for memory_id in ids:
            self.log_retrieval(memory_id)

        return documents

    def log_retrieval(self, memory_id: str) -> None:
        self._retrieval_log[memory_id] = self._retrieval_log.get(memory_id, 0) + 1

    def get_retrieval_logs(self) -> dict[str, int]:
        return dict(self._retrieval_log)
