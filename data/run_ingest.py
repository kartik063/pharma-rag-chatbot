"""
Helper script — run this to ingest all data and verify the pipeline.

Usage (from project root, with venv active):
    python -m data.run_ingest
"""

# sys — standard Python module for interacting with the Python interpreter itself.
# sys.path is the list of folders Python searches when you do "import something".
# We insert the project root at position 0 so Python can find the "rag" and "data" packages
# regardless of which folder you run this script from.
import sys

# os — standard Python module for operating system utilities.
# Used here for:
#   os.path.dirname / os.path.abspath → build the absolute path to the project root
#   os.path.basename                  → strip the folder from a file path to show just the filename
import os

# Insert project root into sys.path so that "from rag.xxx import ..." works correctly.
# os.path.abspath(__file__)         → absolute path to this file (run_ingest.py)
# os.path.dirname(...)              → folder containing this file (data/)
# os.path.dirname(...) again        → one level up = project root (pharma-rag-mcp/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# IngestionPipeline — our class from data/ingest.py.
# .run():
#   - takes:  nothing (reads COLLECTION_MAP and SOURCES_ROOT internally)
#   - does:   loops over all three source folders, builds or loads their ChromaDB collections
#   - gives:  dict of { collection_name → Chroma vectorstore }
from data.ingest import IngestionPipeline

# ── 1. Run ingestion ──────────────────────────────────────────────────────────────────
# IngestionPipeline() creates a VectorStoreManager (which loads the embedding model).
# .run() triggers the actual file reading, splitting, embedding, and ChromaDB saving.
pipeline = IngestionPipeline()
stores = pipeline.run()   # stores = { "drug_info": <Chroma>, "competitor_intel": <Chroma>, ... }

# ── 2. Spot-check each collection with a sample query ─────────────────────────────────
# One natural-language question per collection — chosen to match content we know is in the files.
SAMPLE_QUERIES = {
    "drug_info":        "What are the indications for Jardiance?",
    "competitor_intel": "What were the key results of the Farxiga trial?",
    "pitch_content":    "How did the rep pitch Dupixent to the doctor?",
}

print("\n=== Similarity search spot-checks ===\n")

for collection_name, query in SAMPLE_QUERIES.items():
    print(f"Collection : {collection_name}")
    print(f"Query      : {query}")

    # .similarity_search(query, k=2):
    #   - takes:  query (plain English string), k (number of results to return)
    #   - does:   embeds the query using the same HuggingFace model, then finds the k
    #             stored chunks whose vectors are closest to the query vector (cosine distance)
    #   - gives:  list of k Document objects — doc.page_content = the chunk text,
    #             doc.metadata["source"] = path of the original .txt file the chunk came from
    results = stores[collection_name].similarity_search(query, k=2)

    for i, doc in enumerate(results, start=1):
        # doc.metadata["source"] is the full file path → basename() gives just the filename
        source = doc.metadata.get("source", "unknown")
        print(f"  Result {i} | source: {os.path.basename(source)}")
        # Show only the first 200 characters of the chunk so output stays readable
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"  Result {i} | source: {os.path.basename(source)}")
        print(f"            {snippet}...")

    print()

print("=== All checks passed — pipeline is working ===")
