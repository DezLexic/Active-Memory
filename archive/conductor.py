from .observer_v1 import Observer
from .curator_v1  import Curator
from .librarian   import Librarian
from .reporter    import Reporter


class Conductor:
    def __init__(
        self,
        observer: Observer,
        curator: Curator,
        librarian: Librarian,
        reporter: Reporter,
    ):
        self._observer  = observer
        self._curator   = curator
        self._librarian = librarian
        self._reporter  = reporter
        self._processed_trimmings = 0

    def process_message(self, message: str, role: str) -> str:
        self._observer.add_message(role, message)

        new_trimmings = self._observer.trimmings[self._processed_trimmings:]
        for trimming in new_trimmings:
            decision = self._curator.evaluate(trimming)
            if decision["store"]:
                self._librarian.store(trimming, decision["tier"])
        self._processed_trimmings = len(self._observer.trimmings)

        context = self._observer.summary
        if self._observer.recall_trigger:
            recalled = self._reporter.associative_recall(message)
            if recalled:
                memories = "\n".join(f"- {m}" for m in recalled)
                context = f"{context}\n\nRECALLED FROM MEMORY:\n{memories}"

        return context

    def explicit_lookup(self, query: str) -> list[str]:
        return self._reporter.deliberate_lookup(query)

    def run_consolidation(self) -> None:
        print("Consolidation triggered")
