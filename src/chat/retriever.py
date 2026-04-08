"""Retrieval chain — query ChromaDB and generate answers with GPT-4o.

Connects LangChain's ChatOpenAI to the existing ChromaDB collection.
Retrieves relevant wiki documents and generates sourced answers.
"""

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.utils.config import (
    CHAT_MODEL,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)

# System prompt for the teach mode — answer questions using wiki sources
SYSTEM_PROMPT = """\
You are LearnMate, a personal AI tutor. You teach the user about AI/ML \
concepts using their own Obsidian wiki as the knowledge source.

Rules:
- Answer in Turkish. Use technical terms in English as-is.
- Base your answer ONLY on the provided context. If the context doesn't \
contain enough information, say so honestly.
- Cite which source documents you used (by filename).
- Explain clearly, as if teaching someone who is learning AI/ML.
- Keep answers concise but complete.

Context from wiki:
{context}
"""


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

    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings,
    )

    return vectorstore


def build_chain():
    """Build a retrieval-augmented generation (RAG) chain.

    Flow:
        1. User question comes in
        2. ChromaDB retriever finds top 4 relevant documents
        3. Documents are formatted as context
        4. GPT-4o generates an answer based on context + question

    Returns:
        A LangChain runnable chain (invoke with {"question": "..."}).
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3,
    )

    def format_docs(docs):
        """Format retrieved documents into a single context string.

        Args:
            docs: List of LangChain Document objects.

        Returns:
            Formatted string with document content and source info.
        """
        parts = []
        for doc in docs:
            source = doc.metadata.get("filename", "unknown")
            doc_type = doc.metadata.get("type", "unknown")
            parts.append(f"[{doc_type}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    # RAG chain: retrieve → format → prompt → LLM → parse output
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(question: str) -> str:
    """Ask a question and get a wiki-sourced answer.

    Args:
        question: User's question in any language.

    Returns:
        Generated answer string with source citations.
    """
    chain = build_chain()
    return chain.invoke(question)
