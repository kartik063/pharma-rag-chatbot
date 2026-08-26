"""OpenAI embedding wrapper for ChromaDB ingestion and retrieval."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class EmbeddingModel:
    """Loads and exposes the configured OpenAI embedding model."""

    MODEL_NAME = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def __init__(self):
        print(f"Loading embedding model: {self.MODEL_NAME}", file=sys.stderr)
        self.model = OpenAIEmbeddings(
            model=self.MODEL_NAME,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        print("Embedding model ready.", file=sys.stderr)

    def get(self) -> OpenAIEmbeddings:
        """Return the underlying OpenAIEmbeddings instance."""
        return self.model
