"""Retrieval chain with conversation memory for teach mode.

Uses Chroma Cloud Search API with hybrid search (dense + sparse via RRF).
Maintains conversation history within a session.
"""

from chromadb import K, Knn, Rrf, Search
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.ingest.embedder import get_collection
from src.utils.config import CHAT_MODEL, OPENAI_API_KEY

# Maximum RRF score to include a document in context.
# RRF scores are negative — more negative = better match (higher rank).
# A score near 0 means the document barely ranked. -0.010 filters those out.
RELEVANCE_THRESHOLD = -0.010

# Maximum number of Q&A turns to keep in session history.
# Each turn = 2 messages (HumanMessage + AIMessage).
# Prevents unbounded context growth across long sessions.
MAX_HISTORY_TURNS = 20

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


def hybrid_search(collection, query: str, n_results: int = 4) -> list[dict]:
    """Run hybrid search combining dense (Qwen) and sparse (Splade) rankings.

    Uses Reciprocal Rank Fusion (RRF) to merge semantic similarity
    and keyword matching results.

    Args:
        collection: Chroma Cloud collection.
        query: User's search query.
        n_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: id, document, metadata, score.
    """
    # Hybrid search: dense (semantic) + sparse (keyword) via RRF
    hybrid_rank = Rrf(
        ranks=[
            Knn(query=query, return_rank=True, limit=20),
            Knn(
                query=query,
                key="sparse_embedding",
                return_rank=True,
                limit=20,
            ),
        ],
        weights=[0.7, 0.3],
        k=60,
    )

    search = (
        Search()
        .rank(hybrid_rank)
        .limit(n_results)
        .select(K.DOCUMENT, K.SCORE, K.METADATA)
    )

    results = collection.search(search)
    rows = results.rows()[0]

    docs = []
    for row in rows:
        docs.append(
            {
                "document": row["document"],
                "metadata": row["metadata"],
                "score": row["score"],
            }
        )

    return docs


def _format_docs(docs: list[dict]) -> tuple[str, list[str]]:
    """Format retrieved documents into context string and source list.

    All document types are included in context for answer generation.
    Only documents from "sources" subfolder appear in the sources list.

    Args:
        docs: List of dicts from _hybrid_search.

    Returns:
        Tuple of (formatted context string, list of source names).
    """
    parts = []
    sources = []
    for doc in docs:
        score = doc["score"]
        if score > RELEVANCE_THRESHOLD:
            continue
        metadata = doc["metadata"]
        source = metadata.get("filename", "unknown")
        doc_type = metadata.get("type", "unknown")
        subfolder = metadata.get("subfolder", "")
        parts.append(f"[{doc_type}: {source}]\n{doc['document']}")
        if subfolder == "sources":
            sources.append(source)
    return "\n\n---\n\n".join(parts), sources


class TeachSession:
    """A single teach-mode conversation session with memory.

    Maintains chat history so follow-up questions work naturally.
    Each question triggers a fresh hybrid search from Chroma Cloud.
    """

    def __init__(self) -> None:
        """Initialize a new teach session."""
        self._collection = get_collection()
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
        # Hybrid search: dense + sparse via RRF
        docs = hybrid_search(self._collection, question)
        context, sources = _format_docs(docs)

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

        # Trim oldest turns if history exceeds the limit
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

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
