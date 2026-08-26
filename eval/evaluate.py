"""
Retrieval evaluation for pharma RAG.

Evaluates the quality of the three ChromaDB collections using a fixed set
of question-answer pairs.  For each question the retrieval pipeline is called
directly (no LLM generation), and three metrics are computed:

  - Context Precision  : are the top-k chunks relevant to the question?
  - Hit Rate           : does at least one of the top-k chunks contain the answer?
  - MRR                : at what rank does the first relevant chunk appear?
                         (Mean Reciprocal Rank — higher is better)

No external LLM is required for these metrics; they are computed purely by
string-matching the expected keywords against the retrieved chunk text.  This
makes the eval fast, deterministic, and runnable without Ollama.

Run with (from project root, venv active):
    python -m eval.evaluate
    python -m eval.evaluate --collection drug_info
    python -m eval.evaluate --k 5
"""

# argparse — standard Python library for parsing command-line arguments.
#   - takes:  nothing to import
#   - does:   ArgumentParser().parse_args() reads sys.argv and returns a
#             Namespace object with one attribute per registered argument
#   - gives:  a Namespace whose fields control which collection to eval and k
import argparse

# os — standard Python module for operating system utilities.
#   - takes:  nothing to import
#   - does:   os.path.basename() strips the directory from a file path,
#             leaving just the filename for display in the results table
#   - gives:  string filenames safe to print in the report
import os

# Path — built-in Python class for cross-platform file path handling.
#   - takes:  __file__ at construction time
#   - does:   .resolve().parent.parent finds the project root so load_dotenv()
#             and IngestionPipeline can locate chroma_db/ and data/sources/
#   - gives:  a Path object used by load_dotenv()
from pathlib import Path

# load_dotenv — reads .env at the project root and injects KEY=VALUE into os.environ.
#   - takes:  optional path to the .env file
#   - does:   parses KEY=VALUE lines and calls os.environ.setdefault() for each
#   - gives:  True if the file was found; False otherwise (silent on missing)
from dotenv import load_dotenv

# IngestionPipeline — our own class from data/ingest.py.
#   - takes:  nothing at construction; .run() takes nothing
#   - does:   loads (or builds) the three ChromaDB collections from chroma_db/;
#             returns a dict of ready Chroma vectorstores keyed by collection name
#   - gives:  { "drug_info": Chroma, "competitor_intel": Chroma, "pitch_content": Chroma }
from data.ingest import IngestionPipeline


# ── Environment ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ── Evaluation question sets ──────────────────────────────────────────────────
# Each entry is a dict with:
#   "question"  — the natural-language retrieval query
#   "keywords"  — a list of strings; a chunk is considered relevant if it
#                 contains ALL keywords (case-insensitive)
#
# Keywords are chosen to be specific enough to distinguish relevant chunks
# from noise, while forgiving enough that reasonable paraphrases still match.

EVAL_SETS: dict[str, list[dict]] = {
    "drug_info": [
        {
            "question": "What are the indications for Jardiance?",
            "keywords": ["jardiance", "indication"],
        },
        {
            "question": "What is the dosing for Dupixent?",
            "keywords": ["dupixent", "dose"],
        },
        {
            "question": "What are the contraindications for Eliquis?",
            "keywords": ["eliquis", "contraindication"],
        },
        {
            "question": "What warnings does Xarelto carry?",
            "keywords": ["xarelto", "warning"],
        },
        {
            "question": "What is the mechanism of action of Farxiga?",
            "keywords": ["farxiga", "mechanism"],
        },
    ],
    "competitor_intel": [
        {
            "question": "What were the key results of the Farxiga DAPA-HF trial?",
            "keywords": ["farxiga", "trial"],
        },
        {
            "question": "What endpoints did the Jardiance EMPEROR trial measure?",
            "keywords": ["jardiance", "trial"],
        },
        {
            "question": "What safety outcomes were reported in the Eliquis ARISTOTLE trial?",
            "keywords": ["eliquis", "trial"],
        },
        {
            "question": "What were the efficacy results for Dupixent in its pivotal study?",
            "keywords": ["dupixent", "trial"],
        },
        {
            "question": "What did the Skyrizi trial report about skin clearance?",
            "keywords": ["skyrizi", "trial"],
        },
    ],
    "pitch_content": [
        {
            "question": "How did the rep pitch Dupixent to the dermatologist?",
            "keywords": ["dupixent", "rep"],
        },
        {
            "question": "What objections did the HCP raise about Jardiance?",
            "keywords": ["jardiance", "objection"],
        },
        {
            "question": "How was Eliquis positioned against warfarin in the call?",
            "keywords": ["eliquis", "warfarin"],
        },
        {
            "question": "What questions did the doctor ask about Skyrizi?",
            "keywords": ["skyrizi", "doctor"],
        },
        {
            "question": "How did the rep close the Rinvoq conversation?",
            "keywords": ["rinvoq", "rep"],
        },
    ],
}


# ── Metric helpers ────────────────────────────────────────────────────────────

def _is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    """
    Return True if all keywords appear in the chunk text (case-insensitive).

    Args:
        chunk_text: The raw text content of a retrieved chunk.
        keywords:   List of keyword strings that must ALL appear for relevance.

    Returns:
        True if every keyword is found in the lowercased chunk text.
    """
    # Lower-case the chunk once and check each keyword — avoids repeated .lower() calls.
    lower = chunk_text.lower()
    return all(kw.lower() in lower for kw in keywords)


def _hit_rate(results: list, keywords: list[str]) -> float:
    """
    Compute hit rate: 1.0 if any result is relevant, 0.0 otherwise.

    Args:
        results:  list[Document] from similarity_search().
        keywords: Relevance keywords for this question.

    Returns:
        1.0 if at least one chunk is relevant, else 0.0.
    """
    # Any single relevant chunk is sufficient for a hit.
    return 1.0 if any(_is_relevant(doc.page_content, keywords) for doc in results) else 0.0


def _mrr(results: list, keywords: list[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) for a single query.

    MRR is 1/rank of the first relevant result.  If no result is relevant,
    MRR is 0.0.  For a single query, MRR equals the reciprocal rank (RR).

    Args:
        results:  list[Document] from similarity_search(), ordered by score.
        keywords: Relevance keywords for this question.

    Returns:
        Reciprocal rank of the first relevant result (e.g. 1.0, 0.5, 0.33...)
        or 0.0 if no result is relevant.
    """
    # Enumerate starting at 1 so rank = i (1-indexed).
    for i, doc in enumerate(results, start=1):
        if _is_relevant(doc.page_content, keywords):
            return 1.0 / i
    return 0.0


def _context_precision(results: list, keywords: list[str]) -> float:
    """
    Compute context precision: fraction of retrieved chunks that are relevant.

    Args:
        results:  list[Document] from similarity_search().
        keywords: Relevance keywords for this question.

    Returns:
        Float in [0.0, 1.0] — ratio of relevant chunks to total retrieved.
    """
    if not results:
        return 0.0
    # Count how many of the retrieved chunks pass the relevance check.
    relevant_count = sum(1 for doc in results if _is_relevant(doc.page_content, keywords))
    return relevant_count / len(results)


# ── Collection evaluator ──────────────────────────────────────────────────────

def evaluate_collection(
    store,
    questions: list[dict],
    k: int,
    collection_name: str,
) -> dict:
    """
    Run all questions against one Chroma vectorstore and return aggregate metrics.

    Args:
        store:           A Chroma vectorstore ready for .similarity_search().
        questions:       List of {"question": str, "keywords": list[str]} dicts.
        k:               Number of chunks to retrieve per question.
        collection_name: Display name used in printed results.

    Returns:
        Dict with keys "hit_rate", "mrr", "context_precision" — each is the
        mean across all questions, rounded to 4 decimal places.
    """
    hit_rates, mrrs, precisions = [], [], []

    print(f"\n{'='*60}")
    print(f"Collection: {collection_name}  |  k={k}")
    print(f"{'='*60}")

    for item in questions:
        question = item["question"]
        keywords = item["keywords"]

        # .similarity_search():
        #   - takes:  query (str), k (int)
        #   - does:   embeds the query and retrieves the k nearest stored chunks
        #   - gives:  list[Document], length <= k, ordered by similarity score
        results = store.similarity_search(question, k=k)

        # Compute all three metrics for this question.
        hr = _hit_rate(results, keywords)
        rr = _mrr(results, keywords)
        cp = _context_precision(results, keywords)

        hit_rates.append(hr)
        mrrs.append(rr)
        precisions.append(cp)

        # Print a per-question result row.
        status = "HIT " if hr == 1.0 else "MISS"
        # Truncate long questions to 50 chars for table alignment.
        q_short = question[:50].ljust(50)
        print(
            f"  [{status}]  {q_short}  "
            f"MRR={rr:.2f}  Prec={cp:.2f}"
        )

        # Show the top result's source file for quick inspection.
        if results:
            top_src = os.path.basename(results[0].metadata.get("source", "?"))
            print(f"           Top result: {top_src}")

    # Compute means across all questions.
    n = len(questions)
    agg = {
        "hit_rate":         round(sum(hit_rates) / n, 4),
        "mrr":              round(sum(mrrs) / n, 4),
        "context_precision": round(sum(precisions) / n, 4),
    }

    print(f"\n  Aggregate ({n} questions):")
    print(f"    Hit Rate          : {agg['hit_rate']:.4f}")
    print(f"    MRR               : {agg['mrr']:.4f}")
    print(f"    Context Precision : {agg['context_precision']:.4f}")

    return agg


_CACHED_METRICS: dict[str, dict] | None = None


def get_all_metrics(k: int = 3) -> dict[str, dict]:
    """Return cached aggregate evaluation metrics for every collection."""
    global _CACHED_METRICS
    if _CACHED_METRICS is not None:
        return _CACHED_METRICS

    stores = IngestionPipeline().run()
    _CACHED_METRICS = {}
    for collection_name, questions in EVAL_SETS.items():
        store = stores[collection_name]
        hit_rates, mrrs, precisions = [], [], []
        for item in questions:
            results = store.similarity_search(item["question"], k=k)
            hit_rates.append(_hit_rate(results, item["keywords"]))
            mrrs.append(_mrr(results, item["keywords"]))
            precisions.append(_context_precision(results, item["keywords"]))

        count = len(questions)
        _CACHED_METRICS[collection_name] = {
            "hit_rate": round(sum(hit_rates) / count, 4),
            "mrr": round(sum(mrrs) / count, 4),
            "context_precision": round(sum(precisions) / count, 4),
        }

    collection_metrics = list(_CACHED_METRICS.values())
    collection_count = len(collection_metrics)
    _CACHED_METRICS["overall"] = {
        "hit_rate": round(
            sum(metrics["hit_rate"] for metrics in collection_metrics) / collection_count,
            4,
        ),
        "mrr": round(
            sum(metrics["mrr"] for metrics in collection_metrics) / collection_count,
            4,
        ),
        "context_precision": round(
            sum(metrics["context_precision"] for metrics in collection_metrics) / collection_count,
            4,
        ),
    }

    return _CACHED_METRICS


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """
    Parse CLI arguments, load the vectorstores, and run evaluation.
    """
    # ── CLI argument parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Evaluate pharma RAG retrieval quality."
    )
    # --collection: optionally restrict to one collection; default = all three.
    parser.add_argument(
        "--collection",
        choices=["drug_info", "competitor_intel", "pitch_content"],
        default=None,
        help="Evaluate only this collection (default: all three).",
    )
    # --k: number of chunks to retrieve per question.
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per question (default: 3).",
    )
    args = parser.parse_args()

    # ── Load vectorstores ─────────────────────────────────────────────────────
    # IngestionPipeline().run():
    #   - does:   loads (or builds) all three ChromaDB collections from chroma_db/
    #   - gives:  dict[str, Chroma] — all three stores ready for .similarity_search()
    print("Loading vectorstores...")
    stores = IngestionPipeline().run()

    # ── Determine which collections to evaluate ───────────────────────────────
    # If --collection was specified, evaluate only that one; otherwise all three.
    if args.collection:
        collections_to_eval = [args.collection]
    else:
        collections_to_eval = list(EVAL_SETS.keys())

    # ── Run evaluation ────────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for cname in collections_to_eval:
        result = evaluate_collection(
            store=stores[cname],
            questions=EVAL_SETS[cname],
            k=args.k,
            collection_name=cname,
        )
        all_results[cname] = result

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Collection':<22}  {'Hit Rate':>10}  {'MRR':>8}  {'Ctx Prec':>10}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*10}")
    for cname, metrics in all_results.items():
        print(
            f"  {cname:<22}  "
            f"{metrics['hit_rate']:>10.4f}  "
            f"{metrics['mrr']:>8.4f}  "
            f"{metrics['context_precision']:>10.4f}"
        )

    # If more than one collection was evaluated, print overall means.
    if len(all_results) > 1:
        n_collections = len(all_results)
        mean_hr = sum(m["hit_rate"] for m in all_results.values()) / n_collections
        mean_mrr = sum(m["mrr"] for m in all_results.values()) / n_collections
        mean_cp = sum(m["context_precision"] for m in all_results.values()) / n_collections
        print(f"  {'OVERALL':<22}  {mean_hr:>10.4f}  {mean_mrr:>8.4f}  {mean_cp:>10.4f}")

    print(f"\n{'='*60}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
