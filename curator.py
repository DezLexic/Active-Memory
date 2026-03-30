import json
import ollama


class Curator:
    def __init__(self, model: str = "llama3.2"):
        self.model = model

    def evaluate(self, trimmed_content: str) -> dict:
        prompt = self._build_prompt(trimmed_content)
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse(response.message.content.strip())

    def _build_prompt(self, content: str) -> str:
        return (
            "You are evaluating a piece of trimmed conversation content to decide if it is worth storing in memory.\n\n"
            "RULES:\n"
            "- Store content that contains a decision, preference, constraint, or agreed direction.\n"
            "- Do NOT store generic small talk, greetings, filler, or purely procedural exchanges.\n"
            "- If the content was repeated or clearly notable, lean toward storing it.\n"
            "- Assign tier 'warm' if the content is recent, specific, or likely needed soon.\n"
            "- Assign tier 'cold' if the content is older context, general background, or unlikely to be needed immediately.\n\n"
            "TRIMMED CONTENT:\n"
            f"{content}\n\n"
            "Respond with ONLY a JSON object, no explanation, no markdown. Use this exact shape:\n"
            '{"store": true, "tier": "warm", "reason": "One sentence explanation."}'
        )

    def _parse(self, raw: str) -> dict:
        # Strip markdown code fences if the model wraps the response
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "store": False,
                "tier": "cold",
                "reason": "Could not parse model response.",
                "_raw": raw,
            }

        # Model occasionally returns a list — unwrap the first dict if so
        if isinstance(result, list):
            result = next((item for item in result if isinstance(item, dict)), {})

        if not isinstance(result, dict):
            return {
                "store": False,
                "tier": "cold",
                "reason": "Unexpected response shape from model.",
                "_raw": raw,
            }

        return {
            "store": bool(result.get("store", False)),
            "tier": result.get("tier", "cold") if result.get("store") else None,
            "reason": result.get("reason", ""),
        }
