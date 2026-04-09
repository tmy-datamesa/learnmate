"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# OpenAI (used for chat generation only — embeddings now via Chroma Cloud)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL: str = "gpt-4o"

# Chroma Cloud
CHROMA_API_KEY: str = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT: str = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE: str = os.getenv("CHROMA_DATABASE", "")
CHROMA_COLLECTION_NAME: str = "wiki"

# Local ChromaDB (kept for migration only)
CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb"))

# Source paths
WIKI_PATH: Path = Path(os.getenv("WIKI_PATH", ""))
RAW_PATH: Path = Path(os.getenv("RAW_PATH", ""))
