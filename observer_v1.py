import ollama


class Observer:
    _RECALL_KEYWORDS = {
        "remember", "earlier", "before", "previously", "you said", "we said",
        "we decided", "what did we", "you mentioned", "we agreed", "last time",
        "back to", "going back", "recall", "remind",
    }

    def __init__(self, model: str = "gemma3:4b", max_words: int = 200):
        self.model = model
        self.max_words = max_words
        self.summary = ""
        self.trimmings = []
        self.recall_trigger = False

    def add_message(self, role: str, content: str) -> None:
        self.recall_trigger = self._check_recall(content)

        prompt = self._build_prompt(role, content)
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        new_summary = response.message.content.strip()

        if len(new_summary.split()) > self.max_words:
            new_summary = self._trim(new_summary)

        self.summary = new_summary

    def _check_recall(self, content: str) -> bool:
        lowered = content.lower()
        if "?" in content:
            return True
        return any(kw in lowered for kw in self._RECALL_KEYWORDS)

    def _build_prompt(self, role: str, content: str) -> str:
        if not self.summary:
            return (
                f"A conversation is starting. Summarize this opening message in a few sentences.\n\n"
                f"{role.upper()}: {content}\n\n"
                f"Respond with only the summary, no preamble."
            )
        return (
            f"Below is a running summary of a conversation so far, followed by a new message.\n\n"
            f"CURRENT SUMMARY:\n{self.summary}\n\n"
            f"NEW MESSAGE ({role.upper()}): {content}\n\n"
            f"Update the summary to include the new message. Keep it under {self.max_words} words. "
            f"Respond with only the updated summary, no preamble."
        )

    def _trim(self, text: str) -> str:
        words = text.split()
        excess = words[: len(words) - self.max_words]
        self.trimmings.append(" ".join(excess))
        return " ".join(words[-self.max_words :])
