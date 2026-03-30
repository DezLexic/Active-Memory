from librarian import Librarian


class Reporter:
    def __init__(self, librarian: Librarian):
        self._librarian = librarian

    def associative_recall(self, query: str, n_results: int = 3) -> list[str]:
        warm = self._librarian.retrieve(query, "warm", n_results=n_results)
        cold = self._librarian.retrieve(query, "cold", n_results=n_results)
        # Warm results take priority — prepend them
        return warm + [m for m in cold if m not in warm]

    def deliberate_lookup(self, query: str, n_results: int = 5) -> list[str]:
        warm = self._librarian.retrieve(query, "warm", n_results=n_results)
        cold = self._librarian.retrieve(query, "cold", n_results=n_results)
        return warm + [m for m in cold if m not in warm]
