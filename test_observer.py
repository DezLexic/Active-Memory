from observer import Observer

conversation = [
    ("user",      "Hi! I'm thinking about learning to code. Where should I start?"),
    ("assistant", "Great choice! Python is a popular first language because it's readable and beginner-friendly."),
    ("user",      "I've heard of Python. How long does it take to get good at it?"),
    ("assistant", "With consistent practice, most people can build simple projects within a few months."),
    ("user",      "What kind of projects should a beginner start with?"),
    ("assistant", "Try things like a to-do list app, a simple calculator, or a number guessing game."),
    ("user",      "That sounds fun. Are there free resources you'd recommend?"),
    ("assistant", "Absolutely — freeCodeCamp, the official Python docs, and Automate the Boring Stuff are all excellent."),
    ("user",      "What about after the basics? How do I level up?"),
    ("assistant", "Work on real projects that interest you, read other people's code on GitHub, and contribute to open source when you're ready."),
]

observer = Observer()

for i, (role, content) in enumerate(conversation, start=1):
    print(f"--- Message {i} ({role.upper()}) ---")
    print(f"  {content}")
    observer.add_message(role, content)
    word_count = len(observer.summary.split())
    print(f"\nSUMMARY ({word_count} words):")
    print(f"  {observer.summary}")
    if observer.trimmings:
        print(f"\nTRIMMINGS ({len(observer.trimmings)} entries):")
        for j, t in enumerate(observer.trimmings, start=1):
            print(f"  [{j}] {t}")
    print()
