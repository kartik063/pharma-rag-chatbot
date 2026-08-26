"""
MCP server entrypoint for pharma RAG.

Loads environment variables, initialises the three ChromaDB vectorstores
via IngestionPipeline, registers the four retrieval tools, and starts the
MCP server over stdio transport so MCP-compatible clients can call the tools.

Run with (from project root, venv active):
    python -m mcp_server.server

The -m flag is required: it adds the project root to sys.path so that
`from data.ingest import ...` and `from rag.vectorstore import ...` resolve
correctly regardless of the current working directory.
"""

# os — standard Python module for interacting with the operating system.
#   - takes:  nothing to import
#   - does:   os.getenv(key, default) reads a named environment variable and
#             returns its value as a string, or the default if it is not set
#   - gives:  a string value (or the supplied default) for each env var we read
import os

# sys — standard Python module for interpreter state.
#   - takes:  nothing to import
#   - does:   sys.stderr is the standard error stream; printing to it keeps
#             diagnostic messages off stdout, which the MCP client reads
#             exclusively for JSON-RPC messages
#   - gives:  access to sys.stderr for print(file=sys.stderr) calls
import sys

# Path — built-in Python class for cross-platform file path manipulation.
#   - takes:  a string path (or __file__) at construction time
#   - does:   .resolve() converts a relative path to an absolute one,
#             .parent navigates up one directory level,
#             / "name" appends a path segment using the OS-correct separator
#   - gives:  a Path object; pass it to load_dotenv() or open() directly
from pathlib import Path

# load_dotenv — reads a .env file and injects each KEY=VALUE line into os.environ.
#   - takes:  an optional path to the .env file (defaults to searching upward from cwd)
#   - does:   opens the file, parses KEY=VALUE lines, calls os.environ.setdefault()
#             for each — so it never overwrites a variable already set in the shell
#   - gives:  True if a .env file was found and loaded, False if none was found;
#             the return value is rarely needed and is ignored here
from dotenv import load_dotenv

# FastMCP — the high-level MCP server class from the mcp SDK.
#   - takes:  name (str, shown to MCP clients as the server identity),
#             optional instructions (str, human-readable summary of what this server does)
#   - does:   manages tool/resource/prompt registration; handles the JSON-RPC MCP
#             protocol over whatever transport is chosen (stdio here);
#             exposes .tool() as a decorator to register callables as MCP tools
#   - gives:  a FastMCP instance; call mcp.run(transport="stdio") to start serving
from mcp.server.fastmcp import FastMCP

# IngestionPipeline — our own class from data/ingest.py.
#   - takes:  nothing at construction time
#   - does:   on .run(), iterates COLLECTION_MAP, loads (or builds) each ChromaDB
#             collection from the chroma_db/ directory, and caches them
#   - gives:  a dict { "drug_info": Chroma, "competitor_intel": Chroma,
#             "pitch_content": Chroma } from .run(); each Chroma supports
#             .similarity_search(query, k) for retrieval
from data.ingest import IngestionPipeline

# register_tools — our own function from mcp_server/tools.py.
#   - takes:  mcp (MCPServer instance), stores (dict[str, Chroma])
#   - does:   defines four inner tool functions as closures over `stores` and
#             registers each with @mcp.tool(); after this call, the four tools
#             are live on the server and callable by connected MCP clients
#   - gives:  None (side-effect: tools are registered on `mcp` in-place)
from mcp_server.tools import register_tools


# ── Step 1: resolve project root ─────────────────────────────────────────────
# __file__ is the path to this file (mcp_server/server.py).
# .resolve() makes it absolute so it works regardless of how Python was invoked.
# .parent gives mcp_server/, .parent again gives the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Step 2: load environment variables from .env ──────────────────────────────
# load_dotenv() reads _PROJECT_ROOT/.env and injects KEY=VALUE pairs into
# os.environ.  If the file doesn't exist, load_dotenv() silently does nothing —
# it will not raise an error, so the server still starts using the defaults below.
load_dotenv(_PROJECT_ROOT / ".env")


# ── Step 3: read configuration from environment ───────────────────────────────
# OLLAMA_MODEL — the local Ollama model name used for generation tasks in the
# agent layer (not used by the MCP server itself, stored here as a convenience
# so the agent layer can import it from this module later).
# Default: "llama3.2" — a widely available small Ollama model.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

# OLLAMA_BASE_URL — the HTTP base URL of the running Ollama daemon.
# Default: "http://localhost:11434" — Ollama's standard local port.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ── Step 4: create the MCPServer instance ────────────────────────────────────
# MCPServer():
#   - name:        identifier shown to MCP clients in tool listings
#   - description: free-text description; surfaced by clients that display server info
# This object is module-level so that:
#   (a) tools.py can import it if needed in future, and
#   (b) the `if __name__ == "__main__"` block below can call mcp.run()
mcp = FastMCP(
    name="pharma-rag",
    instructions=(
        "Pharmaceutical RAG retrieval server. "
        "Exposes similarity-search tools over drug labels, "
        "clinical trial data, and sales call notes."
    ),
)


# ── Step 5: run the ingestion pipeline ───────────────────────────────────────
# IngestionPipeline().run():
#   - does:   loads the HuggingFace embedding model once, then for each of the
#             three ChromaDB collections checks whether a persisted store already
#             exists on disk — if yes, loads it; if no, builds and saves it.
#   - gives:  { "drug_info": Chroma, "competitor_intel": Chroma,
#               "pitch_content": Chroma } — all three vectorstores ready to query.
#
# We call this at module level (not inside __main__) so that any code that
# imports `from mcp_server.server import stores` gets the live stores dict.
# The embedding model is loaded once here and reused for every tool call —
# loading it per-call would add ~5 seconds of latency to each query.
print("=== Pharma RAG MCP Server -- starting up ===\n", file=sys.stderr)
stores = IngestionPipeline().run()


# ── Step 6: register tools ────────────────────────────────────────────────────
# register_tools():
#   - takes:  mcp (the FastMCP instance), stores (the dict from Step 5)
#   - does:   defines four inner functions as closures over `stores` and decorates
#             each with @mcp.tool(), making them callable MCP tools
#   - gives:  None (tools are attached to `mcp` in-place)
#
# This must run after IngestionPipeline().run() because the tool closures
# capture `stores` by reference — the dict must be fully populated first.
register_tools(mcp, stores)

# All status messages go to stderr, not stdout.
# stdout is reserved exclusively for MCP JSON-RPC messages — any non-JSON
# text on stdout causes the MCP client to log "Failed to parse JSONRPC message".
print("\n=== Tools registered -- server ready ===", file=sys.stderr)
print(f"  Ollama model    : {OLLAMA_MODEL}", file=sys.stderr)
print(f"  Ollama base URL : {OLLAMA_BASE_URL}", file=sys.stderr)
print("\nWaiting for MCP client connections on stdin...\n", file=sys.stderr)


# ── Entrypoint ────────────────────────────────────────────────────────────────
# mcp.run(transport="stdio"):
#   - takes:  transport="stdio" — tells the SDK to communicate over stdin/stdout
#             using the MCP JSON-RPC framing; the SDK calls anyio.run() internally
#   - does:   enters a blocking event loop that reads JSON-RPC requests from stdin,
#             dispatches them to the registered tools, and writes responses to stdout
#   - gives:  nothing; this call blocks until the client closes the connection
#
# The `if __name__ == "__main__"` guard ensures mcp.run() is only called when
# this file is executed directly (python -m mcp_server.server).  If another
# module imports from server.py, the import completes without starting the server.
if __name__ == "__main__":
    mcp.run(transport="stdio")
