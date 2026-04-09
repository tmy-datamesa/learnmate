"""One-time migration script: local ChromaDB → Chroma Cloud.

Reads all documents from the local PersistentClient and upserts them
into the Chroma Cloud collection with the new hybrid search schema.

Run once after setting up Chroma Cloud credentials in .env:
    python -m src.ingest.migrate
"""

from chromadb import PersistentClient

from src.ingest.embedder import get_collection
from src.utils.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR


def migrate() -> None:
    """Migrate all documents from local ChromaDB to Chroma Cloud.

    Reads every document from the local persistent collection and upserts
    them into the cloud collection. The cloud embedder re-embeds them with
    Qwen + Splade, so the original OpenAI vectors are not transferred.
    """
    # Connect to local ChromaDB
    local_client = PersistentClient(path=str(CHROMA_PERSIST_DIR))

    try:
        local_collection = local_client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        print(
            f"Local collection '{CHROMA_COLLECTION_NAME}' not found at {CHROMA_PERSIST_DIR}."
        )
        print("Nothing to migrate.")
        return

    total = local_collection.count()
    if total == 0:
        print("Local collection is empty. Nothing to migrate.")
        return

    print(f"Found {total} documents in local ChromaDB.")

    # Fetch all docs from local (in batches to avoid memory issues)
    batch_size = 50
    offset = 0
    all_ids = []
    all_documents = []
    all_metadatas = []

    while offset < total:
        result = local_collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        all_ids.extend(result["ids"])
        all_documents.extend(result["documents"])
        all_metadatas.extend(result["metadatas"])
        offset += batch_size
        print(f"  Fetched {min(offset, total)}/{total}...")

    print(f"Uploading {len(all_ids)} documents to Chroma Cloud...")

    # Connect to Chroma Cloud and upsert
    cloud_collection = get_collection()
    cloud_collection.upsert(
        ids=all_ids,
        documents=all_documents,
        metadatas=all_metadatas,
    )

    print(f"Migration complete. {len(all_ids)} documents uploaded.")
    print("Chroma Cloud will re-embed documents with Qwen + Splade automatically.")


if __name__ == "__main__":
    migrate()
