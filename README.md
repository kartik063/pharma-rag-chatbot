# pharma-rag-mcp

A fully local, end-to-end **Retrieval-Augmented Generation (RAG)** system for pharmaceutical sales intelligence — built with LangChain, ChromaDB, Ollama, and the Model Context Protocol (MCP).

The system ingests drug labels, clinical trial documents, and sales call notes into a local vector database, exposes them as MCP tools, and answers natural-language questions through a LangGraph ReAct agent backed by a locally-running LLM.

---

## Architecture

```
data/sources/          ← raw .txt files (drug labels, trials, call notes)
      │
      ▼
data/ingest.py         ← loads, splits into chunks, embeds with all-MiniLM-L6-v2
      │
      ▼
chroma_db/             ← persisted ChromaDB collections (384-dim vectors)
  ├── drug_info/
  ├── competitor_intel/
  └── pitch_content/
      │
      ▼
mcp_server/server.py   ← MCP server over stdio — exposes 4 retrieval tools
      │   (MCP JSON-RPC)
      ▼
agent/agent.py         ← LangGraph ReAct agent (ChatOllama + MCP tools)
      │
      ▼
ui/app.py              ← Gradio chat interface (browser)
```

**Supporting modules**

| Module | Purpose |
|---|---|
| `rag/embeddings.py` | HuggingFace embedding model wrapper (`all-MiniLM-L6-v2`) |
| `rag/vectorstore.py` | ChromaDB collection builder / loader |
| `eval/evaluate.py` | Retrieval quality evaluation (Hit Rate, MRR, Context Precision) |

---

## Knowledge Base

11 drugs × 3 document types = **33 source files**:

| Collection | Source folder | Contents |
|---|---|---|
| `drug_info` | `data/sources/drug_labels/` | FDA-style drug label summaries |
| `competitor_intel` | `data/sources/clinical_trials/` | Clinical trial outcomes |
| `pitch_content` | `data/sources/call_notes/` | Sales rep call transcripts |

**Drugs:** Dupixent, Eliquis, Entresto, Farxiga, Fasenra, Jardiance, Rinvoq, Skyrizi, Trelegy Ellipta, Trulicity, Xarelto

---

## Prerequisites

- **Python 3.13+**
- **[Ollama](https://ollama.ai)** running locally with a model pulled:
  ```bash
  ollama pull llama3.2
  ```
- A Python virtual environment with dependencies installed (see Setup).

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd pharma-rag-mcp

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional — defaults work out of the box)
cp .env.example .env
# Edit .env to set OLLAMA_MODEL and OLLAMA_BASE_URL if needed

# 5. Build the vector database (only needed once)
python -m data.run_ingest
```

---

## Running the System

Each layer can be used independently. Start Ollama before using the agent or UI.

### Agent (CLI)

```bash
python -m agent.agent
```

Interactive command-line chat with two-tier memory:

- **Session memory** — every turn's query, tools called, source mode, and answer are held in RAM for the current run.
- **Long-term memory** — on exit, the session is appended (with timestamps) to `memory/long_term.json` on disk.

Type any of `bye`, `close`, `end`, `exit`, `goodbye`, `quit` to exit — the agent will print a session summary and flush memory before quitting.

#### Answer source modes

The agent detects and labels how each answer was produced:

| Badge | Meaning |
|---|---|
| `[Source: Knowledge Base]` | Answer grounded entirely in retrieved chunks |
| `[Source: Knowledge Base + General Knowledge]` | Retrieved facts combined with general expertise; out-of-KB points marked `[GK]` inline |
| `[Source: General Knowledge only]` | Tools returned nothing relevant; answered from general pharma/sales knowledge |

#### Multi-turn context

The agent passes the last **N turns** of conversation history to the LLM on each question, so follow-up questions like *"he ignored me, how do I re-engage?"* are answered in context. N is controlled by `config.yaml`:

```yaml
agent:
  history_window: 3   # number of prior turns to include
```

### Gradio UI (browser)

```bash
python -m ui.app
```

Opens a chat interface at `http://localhost:7860`.

### MCP Server only (stdio transport)

```bash
python -m mcp_server.server
```

Loads the three ChromaDB collections and waits for MCP JSON-RPC messages on stdin.

**Registered tools:**

| Tool | Description |
|---|---|
| `search_drug_info` | Search drug label documents |
| `search_competitor_intel` | Search clinical trial data |
| `search_pitch_content` | Search sales call notes |
| `search_all` | Search all three collections, merged |

### Retrieval Evaluation

```bash
# Evaluate all three collections
python -m eval.evaluate

# Evaluate one collection with k=5
python -m eval.evaluate --collection drug_info --k 5
```

Prints Hit Rate, MRR, and Context Precision per collection and in aggregate.

---

## Configuration

| File | Purpose |
|---|---|
| `.env` | Ollama model and base URL (copy from `.env.example`) |
| `config.yaml` | Agent behaviour (conversation history window) |

### `.env`

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name (must be pulled first) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP daemon URL |

### `config.yaml`

```yaml
agent:
  history_window: 3   # prior turns passed to LLM for multi-turn context
```

---

## Project Structure

```
pharma-rag-mcp/
├── config.yaml             # Agent configuration
├── data/
│   ├── ingest.py           # IngestionPipeline class
│   ├── run_ingest.py       # CLI: build + spot-check all collections
│   └── sources/
│       ├── drug_labels/    # 11 × drug label .txt files
│       ├── clinical_trials/# 11 × clinical trial .txt files
│       └── call_notes/     # 11 × sales call note .txt files
├── rag/
│   ├── embeddings.py       # EmbeddingModel (all-MiniLM-L6-v2)
│   └── vectorstore.py      # VectorStoreManager (ChromaDB)
├── mcp_server/
│   ├── server.py           # MCP server entrypoint (stdio)
│   └── tools.py            # 4 retrieval tool definitions
├── agent/
│   └── agent.py            # PharmaAgent + CLI loop with memory
├── ui/
│   └── app.py              # Gradio chat UI
├── eval/
│   └── evaluate.py         # Hit Rate / MRR / Context Precision
├── memory/
│   └── long_term.json      # Persisted session history (auto-created)
├── chroma_db/              # Persisted vector collections (git-ignored)
├── .env.example
├── pyproject.toml
└── requirements.txt
```

---

## Key Design Decisions

**Local-first** — no cloud APIs, no API keys. Embeddings via HuggingFace, vector storage via ChromaDB, generation via Ollama.

**MCP as the retrieval layer** — the MCP server cleanly separates retrieval from generation. Any MCP-compatible client can call the search tools.

**Adaptive answer modes** — the agent automatically detects whether a question can be answered from the knowledge base alone, requires blending with general expertise, or falls entirely outside the knowledge base. Each answer is clearly labelled so the user always knows the source.

**Sliding history window** — only the last N turns are sent to the LLM, keeping context window usage bounded while still supporting natural multi-turn conversations.

**Two-tier memory** — session memory (in RAM) is flushed to a persistent JSON log on exit, giving a full audit trail of every query, tool used, source mode, and answer across all runs.
