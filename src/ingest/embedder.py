"""Embed documents and store them in ChromaDB.

Takes parsed wiki documents (from parser.py), generates embeddings via
OpenAI API, and upserts them into a persistent ChromaDB collection.
"""

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from src.utils.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


def get_collection() -> chromadb.Collection:
    """Connect to (or create) the ChromaDB collection with OpenAI embeddings.

    Uses persistent storage so vectors survive between runs.
    The collection uses OpenAI's embedding function directly —
    ChromaDB handles calling the API when documents are added.

    Returns:
        A ChromaDB Collection ready for upsert/query operations.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))

    embedding_fn = OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name=EMBEDDING_MODEL,
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    return collection


def upsert_documents(documents: list[dict]) -> int:
    """Embed and store documents in ChromaDB.

    Uses upsert so re-running the pipeline updates existing documents
    instead of creating duplicates.

    Args:
        documents: List of dicts from load_wiki_documents().
            Each must have keys: id, content, metadata.

    Returns:
        Number of documents upserted.
    """
    if not documents:
        return 0

    collection = get_collection()

    # ChromaDB accepts batch upsert — send all at once
    ids = [doc["id"] for doc in documents]
    contents = [doc["content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]

    collection.upsert(
        ids=ids,
        documents=contents,
        metadatas=metadatas,
    )

    return len(ids)
