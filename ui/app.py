"""
Gradio chat UI for pharma RAG.

Launches a browser-based chat interface that calls PharmaAgent under
the hood.  Each user message triggers one full ReAct agent loop; the
final answer is returned through the OpenAI API.

Run with (from project root, venv active):
    python -m ui.app

The MCP server is launched automatically as a subprocess by PharmaAgent —
you do not need to start mcp_server/server.py separately.
"""

# os — standard Python module for operating system utilities.
#   - takes:  nothing to import
#   - does:   os.getenv() reads named environment variables with optional defaults
#   - gives:  string values of the requested env vars
import os

# asyncio — built-in Python library for running async coroutines.
#   - takes:  nothing to import
#   - does:   asyncio.run(coro) runs a coroutine on a new event loop, blocking
#             until it finishes; used here to call PharmaAgent.ask() from the
#             synchronous Gradio callback
#   - gives:  the return value of the coroutine
import asyncio

# Path — built-in Python class for cross-platform path handling.
#   - takes:  a string path or __file__ at construction time
#   - does:   .resolve().parent.parent navigates from ui/ up to the project root
#   - gives:  a Path object for load_dotenv() and os.path operations
from pathlib import Path

# load_dotenv — reads .env at the project root and injects KEY=VALUE into os.environ.
#   - takes:  optional path to the .env file
#   - does:   parses each KEY=VALUE line and calls os.environ.setdefault(),
#             never overwriting variables already present
#   - gives:  True if the file was found; False otherwise (silent on missing file)
from dotenv import load_dotenv

# gr (gradio) — open-source Python library for building browser-based ML demos.
#   - takes:  nothing to import; gr.ChatInterface, gr.Blocks, gr.Textbox, etc.
#             are the primary UI components used below
#   - does:   renders an HTTP server with a WebSocket-backed UI in the browser;
#             gr.ChatInterface wraps a Python function as a streaming chat window
#   - gives:  a Blocks or Interface object; call .launch() to start the server
import gradio as gr

# PharmaAgent — our async agent from agent/agent.py.
#   - takes:  optional model (str) and base_url (str) at construction time
#   - does:   on .ask(question), connects to the MCP server, loads tools, runs
#             the LangGraph ReAct loop, and returns the final answer string
#   - gives:  a PharmaAgent instance whose .ask() coroutine returns str
from agent.agent import PharmaAgent, OPENAI_MODEL


# ── Environment ───────────────────────────────────────────────────────────────
# Load .env from the project root so OpenAI settings are available.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ── Shared agent instance ─────────────────────────────────────────────────────
# Create one PharmaAgent for the lifetime of the Gradio server.  The MCP server
# subprocess is spawned fresh per ask() call — no persistent subprocess state
# is shared between requests, which keeps things simple and predictable.
_agent = PharmaAgent(model=OPENAI_MODEL)


# ── Chat callback ─────────────────────────────────────────────────────────────

def _chat(message: str, history: list[list[str]]) -> str:
    """
    Gradio chat callback — called once per user message.

    Gradio passes the current user message and the full conversation history.
    We only use `message` here (the agent has no built-in memory across calls;
    each call is an independent ReAct loop).

    Args:
        message: The user's current message text.
        history: List of prior turns as dicts with "role" and "content" keys.
                 Not currently used — each question is answered independently.

    Returns:
        The agent's final answer string, displayed as the assistant reply.
    """
    # Guard against empty or whitespace-only input.
    if not message or not message.strip():
        return "Please enter a question."

    # asyncio.run() executes the async ask() coroutine synchronously.
    # Gradio's default threading model runs callbacks in a thread pool, so
    # asyncio.run() creates a new event loop per call — safe in this context.
    try:
        answer, _, _, metrics = asyncio.run(_agent.ask(message.strip()))
        retrieval = metrics.get("retrieval", {})
        sources = ", ".join(retrieval.get("sources", [])) or "none"
        metrics_text = (
            "\n\n---\n**Metrics**\n"
            f"- Retrieval: {retrieval.get('status', 'unknown')}  "
            f"(tool calls: {retrieval.get('tool_calls', 0)}, "
            f"chunks: {retrieval.get('chunks', 0)})\n"
            f"- Sources: {sources}\n"
        )
        evaluation = metrics.get("evaluation", {})
        if "error" in evaluation:
            metrics_text += f"- Evaluation: unavailable ({evaluation['error']})"
        else:
            metrics_text += "- Evaluation benchmarks:\n"
            for collection, values in evaluation.items():
                metrics_text += (
                    f"  - {collection}: Hit Rate `{values['hit_rate']:.4f}`, "
                    f"MRR `{values['mrr']:.4f}`, "
                    f"Context Precision `{values['context_precision']:.4f}`\n"
                )
        answer = f"{answer}{metrics_text}"
    except Exception as exc:
        # Surface the error as a readable message rather than crashing the UI.
        answer = f"Error: {exc}"

    return answer


# ── Example questions ─────────────────────────────────────────────────────────
# Pre-loaded example questions shown below the chat input so users can click
# them to instantly see what the agent can answer.
_EXAMPLES = [
    "What are the approved indications for Jardiance?",
    "What were the key efficacy results of the Farxiga DAPA-HF trial?",
    "How did the sales rep handle objections when pitching Dupixent?",
    "Compare the safety profiles of Eliquis and Xarelto.",
    "What talking points work best for Skyrizi in psoriasis?",
]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    """
    Assemble and return the Gradio Blocks UI.

    Returns:
        A gr.Blocks instance.  Call .launch() on the returned object to start
        the HTTP server and open the browser.
    """
    # gr.Blocks — the low-level Gradio layout container.
    #   - takes:  theme (gr.Theme), title (str shown in browser tab)
    #   - does:   wraps all child components in a single-page app layout;
    #             components declared inside the `with` block are rendered top-to-bottom
    #   - gives:  a Blocks object; call .launch() to serve it
    with gr.Blocks(title="Pharma RAG Assistant") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # Pharma RAG Assistant
            Ask anything about **drug labels**, **clinical trials**, or **sales strategy**.
            The agent searches the relevant knowledge base and generates a grounded answer.

            **Brands:** Dupixent, Eliquis, Entresto, Farxiga, Fasenra, Jardiance, Rinvoq, Skyrizi, Trelegy Ellipta, Trulicity, Xarelto
            """
        )

        # ── Chat interface ────────────────────────────────────────────────────
        # gr.ChatInterface — a high-level component that wires a Python function
        # to a chat window with message history, an input box, and a submit button.
        #   - takes:  fn (callable: (message, history) -> str | Generator),
        #             examples (list[str] — clickable starter questions),
        #             type ("messages" — use dict-based history format)
        #   - does:   on each submit, calls fn(message, history) and appends the
        #             return value as the assistant's reply in the chat window
        #   - gives:  a ChatInterface component rendered inside the Blocks layout
        gr.ChatInterface(
            fn=_chat,
            examples=_EXAMPLES,
            chatbot=gr.Chatbot(
                label="Pharma RAG",
                height=520,
            ),
            textbox=gr.Textbox(
                placeholder="Ask about a drug, trial, or pitch strategy...",
                label="Your question",
                lines=2,
                submit_btn="Send",
            ),
        )

        # ── Footer info ───────────────────────────────────────────────────────
        gr.Markdown(
            f"""
            ---
            **Model:** `{OPENAI_MODEL}` via OpenAI &nbsp;|&nbsp;
            **Collections:** drug labels · clinical trials · call notes &nbsp;|&nbsp;
            **Embeddings:** text-embedding-3-small via OpenAI
            """
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Pharma RAG UI ===")
    print(f"  Model   : {OPENAI_MODEL}")
    print("  Provider: OpenAI API")
    print("Starting Gradio server...\n")

    # build_ui() assembles the Blocks layout; .launch() starts the HTTP server.
    ui = build_ui()
    port = int(os.environ.get("DATABRICKS_APP_PORT", 7860))
    ui.launch(server_name="0.0.0.0", server_port=port, share=False, inbrowser=False)
