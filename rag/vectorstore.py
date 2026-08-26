"""Builds or loads ChromaDB collections from folders of .txt files."""

import os
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from rag.embeddings import EmbeddingModel

CHROMA_BASE_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


class VectorStoreManager:
    """Manages building and loading ChromaDB vectorstore collections."""

    def __init__(self):
        self.embeddings = EmbeddingModel().get()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )

    def build_or_load_collection(self, name: str, docs_path: str) -> Chroma:
        """
        Load an existing persisted collection or build a new one from source files.

        Args:
            name:      Collection name; used as the ChromaDB collection name and
                       as the subdirectory name under chroma_db/.
            docs_path: Path to a folder of .txt files to ingest.

        Returns:
            A Chroma vectorstore ready for .similarity_search(query, k).
        """
        persist_dir = str(CHROMA_BASE_DIR / name)

        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print(f"[{name}] Loading existing collection from {persist_dir}", file=sys.stderr)
            return Chroma(
                collection_name=name,
                persist_directory=persist_dir,
                embedding_function=self.embeddings,
            )

        print(f"[{name}] Building new collection from {docs_path}", file=sys.stderr)
        loader = DirectoryLoader(
            docs_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        raw_docs = loader.load()
        print(f"[{name}] Loaded {len(raw_docs)} source files", file=sys.stderr)

        chunks = self.splitter.split_documents(raw_docs)
        print(f"[{name}] Split into {len(chunks)} chunks", file=sys.stderr)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=name,
            persist_directory=persist_dir,
        )
        print(f"[{name}] Collection saved to {persist_dir}", file=sys.stderr)
        return vectorstore
