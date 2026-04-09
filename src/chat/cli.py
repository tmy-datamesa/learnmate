"""Interactive CLI for LearnMate — teach and test modes.

Run: python -m src.chat.cli
"""

from src.chat.quiz import TestSession
from src.chat.retriever import TeachSession


def run_teach_mode() -> None:
    """Run teach mode — ask questions, get wiki-sourced answers.

    Maintains conversation memory for follow-up questions.
    """
    print("\n--- TEACH MODE ---")
    print("Ask anything. (q: quit, clear: new session)\n")

    session = TeachSession()

    while True:
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("q", "exit", "quit"):
            break
        if question.lower() == "clear":
            session = TeachSession()
            print("Session cleared.\n")
            continue

        print("\nThinking...\n")
        answer, sources = session.ask(question)
        print(f"LearnMate: {answer}")
        if sources:
            print(f"\nSources: {', '.join(sources)}")
        print()


def run_test_mode() -> None:
    """Run test mode — quiz on wiki topics, evaluate answers.

    Generates questions, evaluates responses, gives session summary.
    """
    print("\n--- TEST MODE ---")
    print("Commands: q (quit), skip (next question), summary (show results)\n")

    session = TestSession()

    topic = input("Topic (leave empty for random): ").strip()

    while True:
        # Generate question
        print("\nGenerating question...\n")
        question, sources = session.generate_question(topic)
        print(f"Question: {question}")
        print(f"(Sources: {', '.join(sources)})\n")

        # Get user's answer
        answer = input("Your answer: ").strip()

        if not answer or answer.lower() in ("q", "exit", "quit"):
            break
        if answer.lower() == "skip":
            print("Skipped.\n")
            continue
        if answer.lower() == "summary":
            print(session.get_summary())
            continue

        # Evaluate
        print("\nEvaluating...\n")
        result = session.evaluate_answer(answer)
        print(result["evaluation_text"])
        print()

        # Ask to continue
        cont = input("Continue? (enter: yes, q: quit, summary: results): ").strip()
        if cont.lower() in ("q", "quit"):
            break
        if cont.lower() == "summary":
            print(session.get_summary())
            break

    # Always show summary at the end
    print(session.get_summary())


def main() -> None:
    """Main entry point — select teach or test mode."""
    print("=" * 40)
    print("  LearnMate — Personal AI Tutor")
    print("=" * 40)
    print("\n1. Teach mode (ask & learn)")
    print("2. Test mode (quiz & evaluate)")
    print("q. Quit\n")

    choice = input("Select mode (1/2/q): ").strip()

    if choice == "1":
        run_teach_mode()
    elif choice == "2":
        run_test_mode()
    elif choice.lower() == "q":
        print("Bye!")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
