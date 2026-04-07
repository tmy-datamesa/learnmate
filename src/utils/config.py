"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = "text-embedding-3-small"

# ChromaDB
CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb"))
CHROMA_COLLECTION_NAME: str = "wiki"

# Source paths
WIKI_PATH: Path = Path(os.getenv("WIKI_PATH", ""))
RAW_PATH: Path = Path(os.getenv("RAW_PATH", ""))
