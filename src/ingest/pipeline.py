"""Main ingest pipeline — parse wiki files and store embeddings.

Run this module to ingest all wiki documents into ChromaDB:
    python -m src.ingest.pipeline
"""

from src.ingest.embedder import sync_documents
from src.ingest.parser import load_wiki_documents
from src.utils.config import WIKI_PATH


def run_wiki_ingest() -> None:
    """Parse wiki markdown files and incrementally sync to ChromaDB.

    Steps:
        1. Load all .md files from wiki/concepts, entities, sources
        2. Extract frontmatter metadata + body content
        3. Compare content hashes with ChromaDB
        4. Only embed new/changed docs, remove deleted ones
    """
    print(f"Wiki path: {WIKI_PATH}")

    if not WIKI_PATH.exists():
        print(f"ERROR: Wiki path does not exist: {WIKI_PATH}")
        return

    # Step 1-2: Parse
    print("Parsing wiki documents...")
    documents = load_wiki_documents(WIKI_PATH)
    print(f"Found {len(documents)} documents in wiki")

    if not documents:
        print("No documents to embed. Check your wiki path.")
        return

    # Step 3-4: Incremental sync
    print("Syncing with ChromaDB...")
    stats = sync_documents(documents)

    print("\nSync complete:")
    print(f"  Added:   {stats['added']}")
    print(f"  Updated: {stats['updated']}")
    print(f"  Deleted: {stats['deleted']}")
    print(f"  Skipped: {stats['skipped']} (unchanged)")


if __name__ == "__main__":
    run_wiki_ingest()
