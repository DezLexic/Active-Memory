from curator import Curator

trimmings = [
    (
        "The user decided they want to build a personal finance tracker as their first real project "
        "and wants to use Python with a SQLite database.",
        "Decision + preference — should store",
    ),
    (
        "The user said hi and the assistant said hello back and asked how it could help today.",
        "Generic greeting — should NOT store",
    ),
    (
        "The user has a hard constraint: the app must run offline with no external API calls "
        "because they work in an environment with no internet access.",
        "Hard constraint — should store",
    ),
    (
        "The assistant explained what a for-loop is and gave a basic example printing numbers one through ten.",
        "Generic tutorial content — should NOT store",
    ),
    (
        "The user mentioned they prefer short concise answers without long explanations "
        "and has asked for this style twice now.",
        "Repeated preference — should store",
    ),
]

curator = Curator()

for i, (content, label) in enumerate(trimmings, start=1):
    print(f"--- Trimming {i} ---")
    print(f"  LABEL:   {label}")
    print(f"  CONTENT: {content}")
    decision = curator.evaluate(content)
    print(f"  STORE:   {decision['store']}")
    print(f"  TIER:    {decision['tier']}")
    print(f"  REASON:  {decision['reason']}")
    if "_raw" in decision:
        print(f"  RAW:     {decision['_raw']}")
    print()
