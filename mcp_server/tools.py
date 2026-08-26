"""
MCP tool definitions for pharma RAG retrieval.

Each tool queries one (or all) ChromaDB collections and returns a
formatted string of ranked results.  Tools are registered onto the
MCPServer instance via register_tools(), which builds closures over
the shared `stores` dict so no module-level globals are needed.

Closures are the right pattern here: the @mcp.tool() decorator fires
at the time register_tools() is called (not at import time), so the
inner functions naturally capture the `stores` reference from the
enclosing scope without any mutation-order risk.
"""

# os — standard Python module for operating system utilities.
#   - takes:  nothing to import; exposes os.path.basename() and os.path.join()
#   - does:   os.path.basename() strips the directory portion from a full file path,
#             returning just the filename (e.g. "jardiance.txt" from a Windows absolute path)
#   - gives:  a plain string filename safe to display in tool output
import os

# Chroma — LangChain's interface to ChromaDB.
#   - takes:  (used as a type hint only here; actual Chroma objects come from IngestionPipeline)
#   - does:   each Chroma object exposes .similarity_search(query, k) which embeds the
#             query using the same HuggingFace model used at index time, then returns
#             the k nearest stored chunks ranked by cosine similarity
#   - gives:  list[Document], where doc.page_content = chunk text,
#             doc.metadata["source"] = absolute path of the originating .txt file
from langchain_chroma import Chroma

# FastMCP — used as a type hint for the mcp parameter of register_tools().
#   - takes:  (type hint only; the live instance is created in server.py)
#   - does:   nothing at import; the real FastMCP object is passed in at call time
#   - gives:  nothing at import
from mcp.server.fastmcp import FastMCP


def register_tools(mcp: FastMCP, stores: dict[str, Chroma]) -> None:
    """
    Register all four retrieval tools on the given MCPServer instance.

    Args:
        mcp:    The MCPServer instance created in server.py.  Each inner function
                is decorated with @mcp.tool(), which registers it as a callable
                MCP tool available to any connected client.
        stores: Dict mapping collection name to its ready Chroma vectorstore.
                Expected keys: "drug_info", "competitor_intel", "pitch_content".
                This dict is captured by closure in each inner tool function, so
                it must be fully populated before register_tools() is called.

    Returns:
        None.  Side-effect: the four tools are registered on `mcp` in-place.
    """

    # ── helper: format a single-collection result list ────────────────────────
    # _format_results() is a local helper shared by the three single-collection
    # tools.  It keeps the output format consistent without repeating logic.
    #
    # _format_results():
    #   - takes:  results (list[Document] from .similarity_search()),
    #             collection_name (str, used only if show_collection=True)
    #             show_collection (bool, True for search_all to add a Collection: line)
    #   - does:   iterates the result list, extracts source filename and chunk text,
    #             and assembles a multi-line formatted string
    #   - gives:  a single str ready to return from an MCP tool
    def _format_results(
        results: list,
        collection_name: str = "",
        *,
        show_collection: bool = False,
    ) -> str:
        # If the vectorstore returned no results (empty collection or no match),
        # return a clear message instead of an empty string.
        if not results:
            return "No results found."

        lines = []

        for i, doc in enumerate(results, start=1):
            # doc.metadata["source"] is the absolute path of the originating file.
            # os.path.basename() strips the directory, giving just the filename.
            source = os.path.basename(doc.metadata.get("source", "unknown"))

            # Build the result block header.
            lines.append(f"[Result {i}]")

            # Include the collection name only for search_all, so the caller
            # knows which knowledge base each chunk came from.
            if show_collection:
                lines.append(f"Collection: {collection_name}")

            lines.append(f"Source: {source}")
            lines.append("---")

            # doc.page_content is the raw chunk text stored in ChromaDB.
            # .strip() removes leading/trailing whitespace that may appear
            # at chunk boundaries after splitting.
            lines.append(doc.page_content.strip())

            # Blank line between results for readability.
            lines.append("")

        # .strip() on the joined string removes the trailing blank line
        # so the returned string ends cleanly at the last chunk.
        return "\n".join(lines).strip()

    # ── Tool 1: search_drug_info ──────────────────────────────────────────────
    # Queries the "drug_info" collection which contains drug label documents:
    # indications, dosing schedules, contraindications, warnings, and
    # regulatory approval information for each drug in the portfolio.
    @mcp.tool(
        description=(
            "Search drug label documents for indications, dosing, "
            "contraindications, and regulatory information."
        )
    )
    def search_drug_info(query: str, k: int = 3) -> str:
        """
        Search the drug_info collection (drug label documents).

        Args:
            query: Natural-language question or keyword phrase to search for.
            k:     Number of top-ranked chunks to return (default 3).

        Returns:
            Formatted string of up to k results, each showing source filename
            and the relevant chunk text.
        """
        # .similarity_search():
        #   - takes:  query (str) — embedded on the fly with the same HuggingFace model
        #             k (int)    — how many nearest chunks to return
        #   - does:   embeds the query, computes cosine similarity against all stored
        #             384-dim vectors in the drug_info collection, ranks by similarity
        #   - gives:  list[Document] sorted nearest-first, length <= k
        results = stores["drug_info"].similarity_search(query, k=k)
        return _format_results(results, "drug_info")

    # ── Tool 2: search_competitor_intel ──────────────────────────────────────
    # Queries the "competitor_intel" collection which contains clinical trial
    # documents: study design, primary endpoints, efficacy outcomes, safety
    # profiles, and head-to-head comparison data for competitor drugs.
    @mcp.tool(
        description=(
            "Search clinical trial documents for competitor efficacy data, "
            "trial outcomes, and study results."
        )
    )
    def search_competitor_intel(query: str, k: int = 3) -> str:
        """
        Search the competitor_intel collection (clinical trial documents).

        Args:
            query: Natural-language question or keyword phrase to search for.
            k:     Number of top-ranked chunks to return (default 3).

        Returns:
            Formatted string of up to k results, each showing source filename
            and the relevant chunk text.
        """
        # .similarity_search() on the competitor_intel store — same mechanics
        # as drug_info above, but searching clinical trial content.
        results = stores["competitor_intel"].similarity_search(query, k=k)
        return _format_results(results, "competitor_intel")

    # ── Tool 3: search_pitch_content ──────────────────────────────────────────
    # Queries the "pitch_content" collection which contains sales call notes:
    # representative-to-HCP conversations, objection-handling scripts, approved
    # talking points, and observed prescriber concerns.
    @mcp.tool(
        description=(
            "Search sales call notes for pitch strategies, objection handling, "
            "and HCP interaction patterns."
        )
    )
    def search_pitch_content(query: str, k: int = 3) -> str:
        """
        Search the pitch_content collection (sales call notes).

        Args:
            query: Natural-language question or keyword phrase to search for.
            k:     Number of top-ranked chunks to return (default 3).

        Returns:
            Formatted string of up to k results, each showing source filename
            and the relevant chunk text.
        """
        # .similarity_search() on the pitch_content store — same mechanics,
        # but searching across sales call note content.
        results = stores["pitch_content"].similarity_search(query, k=k)
        return _format_results(results, "pitch_content")

    # ── Tool 4: search_all ────────────────────────────────────────────────────
    # Queries all three collections and merges the results.  Useful when the
    # caller doesn't know which knowledge base holds the relevant information,
    # or wants a cross-domain view (e.g. "what do we know about Jardiance?").
    @mcp.tool(
        description=(
            "Search all three collections (drug info, competitor intel, pitch content) "
            "and return merged, deduplicated results with collection labels."
        )
    )
    def search_all(query: str, k: int = 3) -> str:
        """
        Search all three collections and return a merged, deduplicated result list.

        Queries drug_info, competitor_intel, and pitch_content in that order,
        collecting up to k results from each.  Chunks that appear in more than
        one collection (which should not happen since the source folders are
        disjoint, but is guarded against) are included only once.

        Args:
            query: Natural-language question or keyword phrase to search for.
            k:     Number of top-ranked chunks to return *per collection* (default 3).
                   Total results will be at most k × 3 after deduplication.

        Returns:
            Formatted string of merged results, each showing collection name,
            source filename, and chunk text.
        """
        # seen — a set of chunk texts already added to the output.
        # Deduplication is by exact string match on page_content.strip().
        # The three source folders (drug_labels/, clinical_trials/, call_notes/)
        # are disjoint, so duplicates are very unlikely, but the guard is cheap.
        seen: set[str] = set()

        # merged — ordered list of (collection_name, Document) pairs.
        # We preserve the order: drug_info first, then competitor, then pitch,
        # so the output reads from most-regulatory to most-commercial.
        merged: list[tuple[str, object]] = []

        # Iterate the three collections in a fixed, meaningful order.
        for collection_name in ("drug_info", "competitor_intel", "pitch_content"):
            # .similarity_search():
            #   - takes:  query (str), k (int)
            #   - does:   embeds query, ranks stored chunks by cosine similarity
            #   - gives:  list[Document] nearest-first, length <= k
            docs = stores[collection_name].similarity_search(query, k=k)

            for doc in docs:
                # Use the stripped chunk text as the deduplication key.
                # Stripping ensures whitespace differences don't create false duplicates.
                key = doc.page_content.strip()

                # Only add the chunk if we haven't seen identical text before.
                if key not in seen:
                    seen.add(key)
                    merged.append((collection_name, doc))

        # If no results were found across all three collections, return a clear message.
        if not merged:
            return "No results found across any collection."

        # Build the formatted output, passing show_collection=True so each result
        # block includes a "Collection: <name>" line indicating its source knowledge base.
        lines = []
        for i, (collection_name, doc) in enumerate(merged, start=1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            lines.append(f"[Result {i}]")
            lines.append(f"Collection: {collection_name}")
            lines.append(f"Source: {source}")
            lines.append("---")
            lines.append(doc.page_content.strip())
            lines.append("")

        return "\n".join(lines).strip()
