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

SYSTEM_PROMPT = """\
You are LearnMate, a personal AI tutor. You teach the user about AI/ML \
concepts using their own Obsidian wiki as the knowledge source.

Rules:
- Answer in Turkish. Use technical terms in English as-is.
- Base your answer ONLY on the provided context. If the context doesn't \
contain enough information, say so honestly.
- Cite which source documents you used (by filename) at the end of your answer.
- Explain clearly, as if teaching someone who is learning AI/ML.
- Keep answers concise but complete.
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


def _format_docs(docs: list) -> tuple[str, list[str]]:
    """Format retrieved documents into context string and source list.

    Args:
        docs: List of LangChain Document objects from retriever.

    Returns:
        Tuple of (formatted context string, list of source names).
    """
    parts = []
    sources = []
    for doc in docs:
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
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})
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
        # Retrieve relevant docs for this specific question
        docs = self._retriever.invoke(question)
        context, sources = _format_docs(docs)

        # Build messages: system + history + new context + question
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(self._history)
        messages.append(
            HumanMessage(
                content=f"Context from wiki:\n{context}\n\nQuestion: {question}"
            )
        )

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
