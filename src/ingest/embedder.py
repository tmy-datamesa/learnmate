"""Embed documents and store them in ChromaDB.

Takes parsed wiki documents (from parser.py), generates embeddings via
OpenAI API, and upserts them into a persistent ChromaDB collection.
Supports incremental sync — only new/changed docs are embedded.
"""

import hashlib

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


def _content_hash(text: str) -> str:
    """Generate MD5 hash of document content for change detection.

    Args:
        text: Document content string.

    Returns:
        Hex digest of the MD5 hash.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def sync_documents(documents: list[dict]) -> dict[str, int]:
    """Incrementally sync documents to ChromaDB.

    Compares content hashes to detect changes:
    - New documents → embed and add
    - Changed documents → re-embed and update
    - Deleted documents (in DB but not in wiki) → remove
    - Unchanged documents → skip

    Args:
        documents: List of dicts from load_wiki_documents().
            Each must have keys: id, content, metadata.

    Returns:
        Dict with counts: {"added": N, "updated": N, "deleted": N, "skipped": N}
    """
    collection = get_collection()
    stats = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0}

    # Build a map of incoming docs: id → (content, metadata)
    incoming = {}
    for doc in documents:
        doc_hash = _content_hash(doc["content"])
        metadata = {**doc["metadata"], "content_hash": doc_hash}
        incoming[doc["id"]] = {"content": doc["content"], "metadata": metadata}

    # Get all existing doc IDs and their hashes from ChromaDB
    existing_ids: list[str] = []
    existing_hashes: dict[str, str] = {}
    if collection.count() > 0:
        existing = collection.get(include=["metadatas"])
        existing_ids = existing["ids"]
        for doc_id, meta in zip(existing["ids"], existing["metadatas"]):
            existing_hashes[doc_id] = meta.get("content_hash", "")

    # Find docs to add or update
    to_upsert_ids: list[str] = []
    to_upsert_contents: list[str] = []
    to_upsert_metadatas: list[dict] = []

    for doc_id, data in incoming.items():
        new_hash = data["metadata"]["content_hash"]

        if doc_id not in existing_hashes:
            # New document
            to_upsert_ids.append(doc_id)
            to_upsert_contents.append(data["content"])
            to_upsert_metadatas.append(data["metadata"])
            stats["added"] += 1
        elif existing_hashes[doc_id] != new_hash:
            # Content changed
            to_upsert_ids.append(doc_id)
            to_upsert_contents.append(data["content"])
            to_upsert_metadatas.append(data["metadata"])
            stats["updated"] += 1
        else:
            stats["skipped"] += 1

    # Upsert new/changed docs in one batch
    if to_upsert_ids:
        collection.upsert(
            ids=to_upsert_ids,
            documents=to_upsert_contents,
            metadatas=to_upsert_metadatas,
        )

    # Find docs to delete (in DB but no longer in wiki)
    incoming_ids = set(incoming.keys())
    to_delete = [doc_id for doc_id in existing_ids if doc_id not in incoming_ids]

    if to_delete:
        collection.delete(ids=to_delete)
        stats["deleted"] = len(to_delete)

    return stats
