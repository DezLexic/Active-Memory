from observer import Observer
from curator import Curator
from librarian import Librarian
from reporter import Reporter


class Conductor:
    def __init__(
        self,
        observer: Observer,
        curator: Curator,
        librarian: Librarian,
        reporter: Reporter,
    ):
        self._observer = observer
        self._curator = curator
        self._librarian = librarian
        self._reporter = reporter
        self._processed_trimmings = 0

    def process_message(self, message: str, role: str) -> str:
        self._observer.add_message(role, message)

        # Handle any new trimmings since last message
        new_trimmings = self._observer.trimmings[self._processed_trimmings:]
        for trimming in new_trimmings:
            decision = self._curator.evaluate(trimming)
            if decision["store"]:
                self._librarian.store(trimming, decision["tier"])
        self._processed_trimmings = len(self._observer.trimmings)

        # Associative recall if the Observer flagged a recall trigger
        if self._observer.recall_trigger:
            self._reporter.associative_recall(message)

        return self._observer.summary

    def explicit_lookup(self, query: str) -> list[str]:
        return self._reporter.deliberate_lookup(query)

    def run_consolidation(self) -> None:
        print("Consolidation triggered")
