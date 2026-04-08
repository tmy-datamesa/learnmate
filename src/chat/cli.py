"""Interactive CLI for LearnMate teach mode.

Run: python -m src.chat.cli
"""

from src.chat.retriever import TeachSession


def main() -> None:
    """Run an interactive teach-mode session in the terminal.

    Maintains conversation memory — follow-up questions work naturally.
    Type 'q' or 'exit' to quit. Type 'clear' to start a new session.
    """
    print("LearnMate — Teach Mode")
    print("Commands: q (quit), clear (new session)\n")

    session = TeachSession()

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("q", "exit", "quit"):
            print("Bye!")
            break

        if question.lower() == "clear":
            session = TeachSession()
            print("Session cleared. Starting fresh.\n")
            continue

        print("\nThinking...\n")
        answer, sources = session.ask(question)
        print(f"LearnMate: {answer}")
        print(f"\nSources: {', '.join(sources)}\n")


if __name__ == "__main__":
    main()
