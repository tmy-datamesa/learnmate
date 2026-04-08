"""Simple CLI for testing the retrieval chain interactively.

Run: python -m src.chat.cli
"""

from src.chat.retriever import ask


def main() -> None:
    """Run an interactive Q&A loop in the terminal.

    Type a question to get a wiki-sourced answer.
    Type 'q' or 'exit' to quit.
    """
    print("LearnMate — Wiki Q&A")
    print("Type your question (q to quit)\n")

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in ("q", "exit", "quit"):
            print("Bye!")
            break

        print("\nThinking...\n")
        answer = ask(question)
        print(f"LearnMate: {answer}\n")


if __name__ == "__main__":
    main()
