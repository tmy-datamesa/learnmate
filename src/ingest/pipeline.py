"""Main ingest pipeline — parse wiki files and store embeddings.

Run this module to ingest all wiki documents into ChromaDB:
    python -m src.ingest.pipeline
"""

from src.ingest.embedder import upsert_documents
from src.ingest.parser import load_wiki_documents
from src.utils.config import WIKI_PATH


def run_wiki_ingest() -> None:
    """Parse wiki markdown files and embed them into ChromaDB.

    Steps:
        1. Load all .md files from wiki/concepts, entities, sources
        2. Extract frontmatter metadata + body content
        3. Upsert into ChromaDB with OpenAI embeddings
    """
    print(f"Wiki path: {WIKI_PATH}")

    if not WIKI_PATH.exists():
        print(f"ERROR: Wiki path does not exist: {WIKI_PATH}")
        return

    # Step 1-2: Parse
    print("Parsing wiki documents...")
    documents = load_wiki_documents(WIKI_PATH)
    print(f"Found {len(documents)} documents")

    if not documents:
        print("No documents to embed. Check your wiki path.")
        return

    # Preview what we found
    for doc in documents:
        print(f"  - {doc['id']} ({doc['metadata']['type']})")

    # Step 3: Embed and store
    print("\nEmbedding and storing in ChromaDB...")
    count = upsert_documents(documents)
    print(f"Done! {count} documents upserted to ChromaDB.")


if __name__ == "__main__":
    run_wiki_ingest()
