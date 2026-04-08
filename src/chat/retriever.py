"""Retrieval chain with conversation memory for teach mode.

Connects LangChain's ChatOpenAI to the existing ChromaDB collection.
Maintains conversation history within a session so follow-up questions
understand prior context.
"""

from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.utils.config import (
    CHAT_MODEL,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

# Minimum relevance score to include a document in context.
# Lowered from 0.35 to 0.25 — 0.35 was filtering out valid results
# for broad queries like "AI öğrenmek için nereden başlamalıyım?"
RELEVANCE_THRESHOLD = 0.25

SYSTEM_PROMPT = """\
You are LearnMate, a personal AI tutor. You teach the user about AI/ML \
concepts using their own Obsidian wiki as the knowledge source.

You are a TEACHER, not an encyclopedia. Your job is to make the user \
think deeper, not just hand them answers.

Rules:
- Answer in Turkish. Use technical terms in English as-is.
- Base your answer ONLY on the provided context. If no context is provided, \
tell the user you don't have relevant information in the wiki for this question.
- Cite which source documents you used (by filename) at the end of your answer.
- Explain with analogies and real-world examples. Connect concepts to \
practical scenarios the user might encounter while building AI products.
- When multiple topics come up in a session, ACTIVELY connect them. \
If the user asked about RAG earlier and now asks about context engineering, \
explain how they relate.
- At the end of EVERY answer, suggest exactly 2 follow-up questions under \
a "Daha derine:" heading. These questions must be challenging and \
thought-provoking — never simple definition questions. They should force \
the user to think about edge cases, trade-offs, or apply the concept \
to a real scenario.
- When the user asks a follow-up question, use the conversation history \
to understand what they're referring to."""


def get_vectorstore() -> Chroma:
    """Connect to the existing ChromaDB collection via LangChain.

    Uses the same collection and embedding model as the ingest pipeline.
    This is read-only — documents are added via the ingest pipeline.

    Returns:
        A LangChain Chroma vectorstore ready for similarity search.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )


def _format_docs(scored_docs: list[tuple]) -> tuple[str, list[str]]:
    """Format retrieved documents into context string and source list.

    Only includes documents that pass the relevance threshold.

    Args:
        scored_docs: List of (Document, score) tuples from similarity search.

    Returns:
        Tuple of (formatted context string, list of source names).
        Both are empty if no documents pass the threshold.
    """
    parts = []
    sources = []
    for doc, score in scored_docs:
        if score < RELEVANCE_THRESHOLD:
            continue
        source = doc.metadata.get("filename", "unknown")
        doc_type = doc.metadata.get("type", "unknown")
        parts.append(f"[{doc_type}: {source}]\n{doc.page_content}")
        sources.append(f"{doc_type}: {source}")
    return "\n\n---\n\n".join(parts), sources


class TeachSession:
    """A single teach-mode conversation session with memory.

    Maintains chat history so follow-up questions work naturally.
    Each question triggers a fresh retrieval from ChromaDB, but the
    LLM sees the full conversation history for context.
    """

    def __init__(self) -> None:
        """Initialize a new teach session."""
        self._vectorstore = get_vectorstore()
        self._llm = ChatOpenAI(
            model=CHAT_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.3,
        )
        self._history: list[HumanMessage | AIMessage] = []

    def ask(self, question: str) -> tuple[str, list[str]]:
        """Ask a question with full conversation context.

        Args:
            question: User's question in any language.

        Returns:
            Tuple of (answer string, list of source document names).
        """
        # Retrieve docs with relevance scores, filter by threshold
        scored_docs = self._vectorstore.similarity_search_with_relevance_scores(
            question, k=4
        )
        context, sources = _format_docs(scored_docs)

        # Build messages: system + history + context (if any) + question
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(self._history)

        if context:
            user_msg = f"Context from wiki:\n{context}\n\nQuestion: {question}"
        else:
            user_msg = f"No relevant wiki context found.\n\nQuestion: {question}"

        messages.append(HumanMessage(content=user_msg))

        # Generate answer
        response = self._llm.invoke(messages)
        answer = response.content

        # Save to history (without context to keep history clean)
        self._history.append(HumanMessage(content=question))
        self._history.append(AIMessage(content=answer))

        return answer, sources


# Convenience function for single questions (no memory)
def ask(question: str) -> str:
    """Ask a single question without conversation memory.

    Args:
        question: User's question in any language.

    Returns:
        Generated answer string with source citations.
    """
    session = TeachSession()
    answer, _ = session.ask(question)
    return answer
