import ollama
from conductor import Conductor


class ActiveAgent:
    def __init__(self, conductor: Conductor, model: str = "gemma3:4b"):
        self._conductor = conductor
        self._model = model

    def chat(self, message: str) -> str:
        # Update the Observer with the user message and get the assembled context window
        context = self._conductor.process_message(message, "user")

        system_prompt = (
            "You are a helpful assistant. "
            "The following is a summary of the conversation so far — "
            "this is the only context you have. Do not invent history beyond what is shown.\n\n"
            f"CONTEXT:\n{context}"
        )

        response = ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
        )

        reply = response.message.content.strip()

        # Feed the assistant response back through the Conductor so the Observer tracks it
        self._conductor.process_message(reply, "assistant")

        return reply

    def lookup(self, query: str) -> str:
        results = self._conductor.explicit_lookup(query)
        if not results:
            return "No memories found."
        return "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
