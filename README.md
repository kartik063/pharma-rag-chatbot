# pharma-rag-mcp

A **Retrieval-Augmented Generation (RAG)** system for pharmaceutical sales intelligence — built with LangChain, LangGraph, ChromaDB, OpenAI, and the Model Context Protocol (MCP).

The system ingests drug labels, clinical trial documents, and sales call notes into ChromaDB, exposes retrieval functions as MCP tools, and answers natural-language questions through a LangGraph-backed ReAct agent created with LangChain.

---

## Architecture

```
data/sources/          ← raw .txt files (drug labels, trials, call notes)
      │
      ▼
data/ingest.py         ← loads, splits into chunks, embeds with OpenAI
      │
      ▼
chroma_db/             ← persisted ChromaDB collections (OpenAI embeddings)
  ├── drug_info/
  ├── competitor_intel/
  └── pitch_content/
      │
      ▼
mcp_server/server.py   ← MCP server over stdio — exposes 4 retrieval tools
      │   (MCP JSON-RPC)
      ▼
agent/agent.py         ← LangChain agent API creates the LangGraph ReAct workflow
      │
      ▼
ui/app.py              ← Gradio chat interface (browser)
```

### Framework responsibilities

| Component | Role in this project |
|---|---|
| **LangChain** | Provides `ChatOpenAI`, `OpenAIEmbeddings`, Chroma integration, document loaders, text splitters, message types, and the MCP client adapter. |
| **LangGraph** | Provides the agent orchestration behind `langchain.agents.create_agent`: the ReAct loop decides when to call retrieval tools and when to produce the final answer. The project does not directly instantiate `StateGraph`; LangChain creates and manages the graph. |
| **MCP** | Provides the retrieval transport. `mcp_server/server.py` starts the stdio server and `mcp_server/tools.py` exposes `search_drug_info`, `search_competitor_intel`, `search_pitch_content`, and `search_all`. |
| **OpenAI** | Supplies the chat model and the `text-embedding-3-small` embedding model through LangChain integrations. |
| **ChromaDB** | Persists document embeddings and performs similarity search. |
| **Gradio** | Provides the browser-based chat interface. |
| **Databricks SDK** | Retrieves the OpenAI API key from Databricks Secrets when `OPENAI_API_KEY` is not already set. |

### Request flow

```text
Gradio UI or CLI
      │
      ▼
PharmaAgent (LangChain API)
      │
      ▼
LangGraph ReAct workflow
      │
      ├── OpenAI ChatOpenAI
      ├── MCP client ── stdio ── MCP retrieval server
      │                         │
      │                         ▼
      │                    ChromaDB
      │
      └── final answer and retrieval/evaluation metrics
```

**Supporting modules**

| Module | Purpose |
|---|---|
| `rag/embeddings.py` | OpenAI embedding model wrapper (`text-embedding-3-small`) |
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
- An [OpenAI](https://platform.openai.com) API key.
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

# 4. Configure environment
cp .env.example .env
# Edit .env and replace the OpenAI API key placeholder.

# 5. Build the vector database (only needed once)
python -m data.run_ingest
```

---

## Running the System

Each layer can be used independently. Internet access and a valid OpenAI API key
are required when using the agent or UI. The MCP server is launched automatically
by `PharmaAgent`; it does not need to be started separately for normal use.

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
This command is useful for testing the retrieval server directly. The CLI and UI
start it automatically through `MultiServerMCPClient`.

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
| `.env` | OpenAI model and API key (copy from `.env.example`) |
| `config.yaml` | Agent behaviour (conversation history window) |

### `.env`

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `OPENAI_API_KEY` | none | OpenAI API key |

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
│   ├── embeddings.py       # EmbeddingModel (text-embedding-3-small)
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

**LangChain and LangGraph** — LangChain supplies the models, embeddings, document/vector-store integrations, messages, and MCP adapter. Its `create_agent` API creates the LangGraph ReAct workflow that orchestrates tool calls and answer generation.

**OpenAI generation and embeddings** — model responses and vector embeddings use the OpenAI API, with vectors stored locally in ChromaDB.

**MCP as the retrieval layer** — the MCP server cleanly separates retrieval from generation. Any MCP-compatible client can call the search tools.

**Adaptive answer modes** — the agent automatically detects whether a question can be answered from the knowledge base alone, requires blending with general expertise, or falls entirely outside the knowledge base. Each answer is clearly labelled so the user always knows the source.

**Sliding history window** — only the last N turns are sent to the LLM, keeping context window usage bounded while still supporting natural multi-turn conversations.

**Two-tier memory** — session memory (in RAM) is flushed to a persistent JSON log on exit, giving a full audit trail of every query, tool used, source mode, and answer across all runs.
