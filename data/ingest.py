"""Ingestion pipeline: maps source folders to ChromaDB collections."""

import sys
from pathlib import Path

from langchain_chroma import Chroma
from rag.vectorstore import VectorStoreManager

SOURCES_ROOT = Path(__file__).resolve().parent / "sources"

COLLECTION_MAP = {
    "drug_info":        "drug_labels",
    "competitor_intel": "clinical_trials",
    "pitch_content":    "call_notes",
}


class IngestionPipeline:
    """Builds or loads all three ChromaDB collections."""

    def __init__(self):
        self.manager = VectorStoreManager()
        self.stores: dict[str, Chroma] = {}

    def run(self) -> dict[str, Chroma]:
        """
        Build or load each collection defined in COLLECTION_MAP.

        Returns:
            Dict mapping collection name to its ready Chroma vectorstore.
        """
        print("=== Starting ingestion pipeline ===\n", file=sys.stderr)
        for collection_name, subfolder in COLLECTION_MAP.items():
            docs_path = str(SOURCES_ROOT / subfolder)
            print(f"--- Collection: {collection_name} | Source: {docs_path}", file=sys.stderr)
            self.stores[collection_name] = self.manager.build_or_load_collection(
                name=collection_name,
                docs_path=docs_path,
            )
            print(file=sys.stderr)
        print("=== Ingestion complete ===", file=sys.stderr)
        return self.stores
