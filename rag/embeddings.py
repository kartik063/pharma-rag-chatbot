"""Embedding model wrapper for ChromaDB ingestion and retrieval."""

import sys
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    """Loads and exposes the all-MiniLM-L6-v2 sentence-transformer model."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        print(f"Loading embedding model: {self.MODEL_NAME}", file=sys.stderr)
        self.model = HuggingFaceEmbeddings(
            model_name=self.MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("Embedding model ready.", file=sys.stderr)

    def get(self) -> HuggingFaceEmbeddings:
        """Return the underlying HuggingFaceEmbeddings instance."""
        return self.model
