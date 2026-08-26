"""
Evaluation package for pharma RAG.

Runs a RAGAS-style evaluation over the three ChromaDB collections
using a fixed question set, measuring retrieval quality metrics:
context precision, faithfulness, and answer relevancy.

Entry point:
    python -m eval.evaluate
"""
