"""Embed documents and store them in Chroma Cloud.

Uses Chroma Cloud Qwen for dense embeddings and Splade for sparse
embeddings. Supports incremental sync via content hashing.
"""

import hashlib

import chromadb
from chromadb import Schema, SparseVectorIndexConfig, VectorIndexConfig, K
from chromadb.utils.embedding_functions import (
    ChromaCloudQwenEmbeddingFunction,
    ChromaCloudSpladeEmbeddingFunction,
)

from src.utils.config import (
    CHROMA_API_KEY,
    CHROMA_COLLECTION_NAME,
    CHROMA_DATABASE,
    CHROMA_TENANT,
)


def get_client() -> chromadb.ClientAPI:
    """Connect to Chroma Cloud.

    Returns:
        A Chroma CloudClient instance.
    """
    return chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY,
    )


def _build_schema() -> Schema:
    """Build collection schema with dense (Qwen) and sparse (Splade) indexes.

    Dense: semantic similarity search via Chroma Cloud Qwen embeddings.
    Sparse: keyword matching via Chroma Cloud Splade embeddings.
    Together they enable hybrid search with RRF.

    Returns:
        A Schema configured for hybrid search.
    """
    schema = Schema()

    # Dense embeddings — Chroma Cloud Qwen (free, no OpenAI cost)
    dense_ef = ChromaCloudQwenEmbeddingFunction(api_key=CHROMA_API_KEY)
    schema.create_index(
        config=VectorIndexConfig(
            space="cosine",
            embedding_function=dense_ef,
        )
    )

    # Sparse embeddings — Chroma Cloud Splade (keyword search)
    sparse_ef = ChromaCloudSpladeEmbeddingFunction(api_key=CHROMA_API_KEY)
    schema.create_index(
        config=SparseVectorIndexConfig(
            source_key=K.DOCUMENT,
            embedding_function=sparse_ef,
        ),
        key="sparse_embedding",
    )

    return schema


def get_collection() -> chromadb.Collection:
    """Get or create the wiki collection with hybrid search schema.

    Returns:
        A Chroma Cloud Collection ready for upsert/search.
    """
    client = get_client()
    schema = _build_schema()

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        schema=schema,
    )


def _content_hash(text: str) -> str:
    """Generate MD5 hash of document content for change detection.

    Args:
        text: Document content string.

    Returns:
        Hex digest of the MD5 hash.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def sync_documents(documents: list[dict]) -> dict[str, int]:
    """Incrementally sync documents to Chroma Cloud.

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

    # Get all existing doc IDs and their hashes from Chroma Cloud
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
            to_upsert_ids.append(doc_id)
            to_upsert_contents.append(data["content"])
            to_upsert_metadatas.append(data["metadata"])
            stats["added"] += 1
        elif existing_hashes[doc_id] != new_hash:
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
