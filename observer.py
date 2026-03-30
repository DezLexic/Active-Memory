import ollama


class Observer:
    def __init__(self, model: str = "llama3.2", max_words: int = 200):
        self.model = model
        self.max_words = max_words
        self.summary = ""
        self.trimmings = []

    def add_message(self, role: str, content: str) -> None:
        prompt = self._build_prompt(role, content)
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        new_summary = response.message.content.strip()

        if len(new_summary.split()) > self.max_words:
            new_summary = self._trim(new_summary)

        self.summary = new_summary

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
