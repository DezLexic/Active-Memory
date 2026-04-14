"""
loader.py

Parses locomo10.json into Python dataclasses for use by the LoCoMo benchmark.

The LoCoMo dataset contains 10 long conversations between two speakers, each
split into multiple sessions.  Every conversation has a set of QA pairs
annotated with category labels (single-hop, multi-hop, temporal, open-domain).

Usage
-----
    from loader import load_locomo

    conversations = load_locomo("benchmarks/locomo/data/locomo10.json")
    conv = conversations[0]
    print(len(conv.turns))      # total turns across all sessions
    print(len(conv.questions))  # annotated QA pairs

Dataset structure expected in locomo10.json
-------------------------------------------
    {
      "0": {
        "session_1": [{"speaker": "...", "text": "..."}, ...],
        "session_2": [...],
        ...
        "qa": [{"question": "...", "answer": "...", "category": 1}, ...]
      },
      "1": {...},
      ...
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# ── Category labels matching the LoCoMo paper ──────────────────────────────────

CATEGORY_LABELS: dict[int, str] = {
    1: "Single-hop",
    2: "Multi-hop",
    3: "Temporal",
    4: "Open-domain",
}


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class LoCoMoQuestion:
    question:       str
    answer:         str
    category:       int   # 1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain
    category_label: str


@dataclass
class LoCoMoConversation:
    id:        str               # conversation index as string, e.g. "0"
    turns:     list[dict]        # {"speaker": str, "text": str, "session": int}
    questions: list[LoCoMoQuestion]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_text(turn: dict) -> str:
    """Return the turn's text content, trying several field names."""
    for field_name in ("text", "dialog", "utterance", "content"):
        val = turn.get(field_name, "")
        if val:
            return str(val).strip()
    return ""


def _parse_sessions(conv_data: dict) -> list[dict]:
    """
    Flatten all session_N entries into a single ordered list of turns.

    Sessions are sorted numerically (session_1 before session_2, etc.) so
    the conversation flows in the correct temporal order.
    """
    session_keys = sorted(
        [k for k in conv_data if k.startswith("session_")],
        key=lambda k: int(k.split("_", 1)[1]),
    )
    turns: list[dict] = []
    for skey in session_keys:
        session_num = int(skey.split("_", 1)[1])
        for raw_turn in conv_data[skey]:
            text = _extract_text(raw_turn)
            if text:
                turns.append({
                    "speaker": str(raw_turn.get("speaker", "")).strip(),
                    "text":    text,
                    "session": session_num,
                })
    return turns


def _parse_questions(conv_data: dict) -> list[LoCoMoQuestion]:
    """Parse the 'qa' list from a conversation entry."""
    questions: list[LoCoMoQuestion] = []
    for qa in conv_data.get("qa", []):
        question = str(qa.get("question", "")).strip()
        answer   = str(qa.get("answer", "")).strip()
        category = int(qa.get("category", 0))
        if not question:
            continue
        questions.append(LoCoMoQuestion(
            question=question,
            answer=answer,
            category=category,
            category_label=CATEGORY_LABELS.get(category, f"cat{category}"),
        ))
    return questions


# ── Public API ─────────────────────────────────────────────────────────────────

def load_locomo(path: str) -> list[LoCoMoConversation]:
    """
    Load locomo10.json and return all 10 conversations as dataclasses.

    Conversations are returned in ascending numeric order (0 through 9).

    Parameters
    ----------
    path    Filesystem path to locomo10.json.

    Returns
    -------
    list[LoCoMoConversation]
        One entry per conversation, each containing turns and QA pairs.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.  Clone the LoCoMo dataset first:
        https://github.com/snap-research/locomo
    """
    with open(path, encoding="utf-8") as f:
        data: dict = json.load(f)

    conversations: list[LoCoMoConversation] = []
    for conv_id in sorted(data.keys(), key=lambda k: int(k)):
        conv_data = data[conv_id]
        conversations.append(LoCoMoConversation(
            id=conv_id,
            turns=_parse_sessions(conv_data),
            questions=_parse_questions(conv_data),
        ))
    return conversations
