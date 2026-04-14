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

Actual locomo10.json structure
------------------------------
    [
      {
        "sample_id": "conv-26",
        "conversation": {
          "speaker_a": "Caroline",
          "speaker_b": "Melanie",
          "session_1_date_time": "...",
          "session_1": [
            {"speaker": "Caroline", "dia_id": "D1:1", "text": "..."},
            ...
          ],
          "session_2": [...],
          ...
        },
        "qa": [
          {"question": "...", "answer": "...", "category": 2, "evidence": [...]},
          ...
        ],
        ...
      },
      ...   (10 total)
    ]
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# ── Category labels matching the LoCoMo paper ─────────────────────────────────

CATEGORY_LABELS: dict[int, str] = {
    1: "Single-hop",
    2: "Multi-hop",
    3: "Temporal",
    4: "Open-domain",
}


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class LoCoMoQuestion:
    question:       str
    answer:         str
    category:       int   # 1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain
    category_label: str


@dataclass
class LoCoMoConversation:
    id:        str            # sample_id from the dataset, e.g. "conv-26"
    idx:       int            # 0-based position in the file
    turns:     list[dict]     # {"speaker": str, "text": str, "session": int}
    questions: list[LoCoMoQuestion]
    speaker_a: str            # name of the first speaker
    speaker_b: str            # name of the second speaker


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_sessions(conv: dict) -> list[dict]:
    """
    Flatten all session_N entries inside a conversation dict into a single
    ordered list of turns, sorted by session number.

    Skips keys that are not actual session lists (e.g. speaker_a, speaker_b,
    session_N_date_time).
    """
    session_keys = sorted(
        [
            k for k in conv
            if k.startswith("session_") and isinstance(conv[k], list)
        ],
        key=lambda k: int(k.split("_", 1)[1]),
    )
    turns: list[dict] = []
    for skey in session_keys:
        session_num = int(skey.split("_", 1)[1])
        for raw_turn in conv[skey]:
            text = str(raw_turn.get("text", "")).strip()
            if text:
                turns.append({
                    "speaker": str(raw_turn.get("speaker", "")).strip(),
                    "text":    text,
                    "session": session_num,
                })
    return turns


def _parse_questions(item: dict) -> list[LoCoMoQuestion]:
    """Parse the top-level 'qa' list from a dataset entry."""
    questions: list[LoCoMoQuestion] = []
    for qa in item.get("qa", []):
        question = str(qa.get("question", "")).strip()
        answer   = str(qa.get("answer",   "")).strip()
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


# ── Public API ────────────────────────────────────────────────────────────────

def load_locomo(path: str) -> list[LoCoMoConversation]:
    """
    Load locomo10.json and return all 10 conversations as dataclasses.

    Conversations are returned in file order (index 0 through 9).

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
        data: list = json.load(f)

    conversations: list[LoCoMoConversation] = []
    for idx, item in enumerate(data):
        conv      = item.get("conversation", {})
        sample_id = str(item.get("sample_id", idx))
        conversations.append(LoCoMoConversation(
            id=sample_id,
            idx=idx,
            turns=_parse_sessions(conv),
            questions=_parse_questions(item),
            speaker_a=str(conv.get("speaker_a", "speaker_a")),
            speaker_b=str(conv.get("speaker_b", "speaker_b")),
        ))
    return conversations
