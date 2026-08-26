"""
Pharma RAG CLI agent.

Connects to the pharma-rag MCP server via stdio transport, loads its four
retrieval tools, and runs a LangGraph ReAct loop backed by a local Ollama LLM.

Run from project root (venv active):
    python -m agent.agent
"""

import os
import sys
import json
import yaml
import asyncio
import textwrap
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import create_agent as create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- Config ------------------------------------------------------------------

_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"
try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as _fh:
        _cfg = yaml.safe_load(_fh) or {}
    HISTORY_WINDOW: int = int(_cfg.get("agent", {}).get("history_window", 3))
except (OSError, yaml.YAMLError, ValueError):
    HISTORY_WINDOW = 3

# --- Environment -------------------------------------------------------------

OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MCP_CMD = [sys.executable, "-m", "mcp_server.server"]

# --- Memory paths ------------------------------------------------------------

_MEMORY_DIR = _PROJECT_ROOT / "memory"
_LONG_TERM_PATH = _MEMORY_DIR / "long_term.json"

# --- System prompt -----------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a pharmaceutical sales intelligence assistant.
You have access to three retrieval tools:
  - search_drug_info:        drug label information (indications, dosing, contraindications)
  - search_competitor_intel: clinical trial data (efficacy, safety, outcomes)
  - search_pitch_content:    sales call notes (pitch strategies, objection handling)
  - search_all:              all three collections at once

Always use one or more tools first to retrieve relevant information before answering.

After retrieving, classify the question into one of three modes and respond accordingly:

MODE 1 — KNOWLEDGE BASE ONLY
  Use this when the retrieved chunks fully answer the question with no context gap.
  Example: "What are the indications for Dupixent?"
  → Answer strictly from retrieved chunks. No additions from outside the knowledge base.
  → Do NOT add any inline markers.

MODE 2 — AMALGAMATION (Knowledge Base + General Expertise)
  Use this when the question requires retrieved facts AND broader context
  (e.g. HCP psychology, audience-specific framing, sales strategy, specialty nuance)
  that the knowledge base does not cover.
  Example: "How do I pitch Dupixent to a pediatric pulmonologist who prefers competitor brands?"
  → Lead with knowledge base content (retrieved drug facts, trial data, call note tactics).
  → Then add your general pharmaceutical sales expertise to fill contextual gaps.
  → Mark EVERY sentence or bullet that comes from outside the knowledge base with the
    inline tag: [GK] at the start of that point.
  → After your answer add a one-line summary:
    Sources: Knowledge Base + General Knowledge

MODE 3 — GENERAL KNOWLEDGE ONLY
  Use this when the tools return no useful information for the question.
  Example: "How do I handle a cold call to a doctor I've never met?"
  → Begin your response with this flag on its own line:
    [General Knowledge — not grounded in the knowledge base]
  → Then answer from your general pharmaceutical and sales expertise.

If you are genuinely uncertain even with general knowledge, say so clearly.
Never fabricate drug facts, dosing, or trial data not present in retrieved chunks.
"""


# --- Agent -------------------------------------------------------------------

class PharmaAgent:
    """
    ReAct agent for pharmaceutical sales intelligence.

    Spawns the MCP server as a subprocess per call, loads its tools, and
    runs a LangGraph ReAct loop to retrieve and synthesise an answer.
    """

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> None:
        self.model = model
        self.base_url = base_url

    async def ask(
        self,
        question: str,
        history: list | None = None,
    ) -> tuple[str, list[str], str]:
        """
        Run one ReAct loop for the given question.

        Args:
            question: The user's natural-language question.
            history:  Optional list of prior HumanMessage / AIMessage objects
                      (last N turns) to give the LLM multi-turn context.

        Returns:
            Tuple of (answer, tools_called, source_mode) where source_mode is
            one of "kb_only", "amalgamation", or "general_only".
        """
        llm = ChatOllama(model=self.model, base_url=self.base_url)

        client = MultiServerMCPClient({
            "pharma-rag": {
                "command": _MCP_CMD[0],
                "args": _MCP_CMD[1:],
                "transport": "stdio",
                "env": dict(os.environ),
            }
        })
        tools = await client.get_tools()

        agent = create_react_agent(llm, tools, system_prompt=_SYSTEM_PROMPT)

        prior = list(history) if history else []
        result = await agent.ainvoke(
            {"messages": prior + [HumanMessage(content=question)]}
        )

        answer = str(result["messages"][-1].content)

        seen: set[str] = set()
        tools_called: list[str] = []
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage) and msg.name not in seen:
                seen.add(msg.name)
                tools_called.append(msg.name)

        answer_lower = answer.lower()
        if "[general knowledge" in answer_lower:
            source_mode = "general_only"
        elif "[gk]" in answer_lower:
            source_mode = "amalgamation"
        else:
            source_mode = "kb_only"

        return answer, tools_called, source_mode


# --- CLI helpers -------------------------------------------------------------

_W = 60
_EXIT_WORDS = {"exit", "quit", "close", "bye", "goodbye", "end"}
_MODE_BADGE = {
    "kb_only":      "  [Source: Knowledge Base]",
    "amalgamation": "  [Source: Knowledge Base + General Knowledge]",
    "general_only": "  [Source: General Knowledge only]",
}
_MODE_LABEL = {
    "kb_only": "KB", "amalgamation": "KB+GK", "general_only": "GK", "error": "ERR",
}


def _hr(char: str = "-") -> str:
    """Return a horizontal rule of _W characters."""
    return char * _W


def _load_long_term() -> list[dict]:
    """Load session history from disk, returning an empty list on any failure."""
    if not _LONG_TERM_PATH.exists():
        return []
    try:
        with open(_LONG_TERM_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _flush_session(session_turns: list[dict]) -> None:
    """
    Append session_turns to the long-term JSON file as one session record.

    Does nothing if session_turns is empty. Each session record has the shape:
        {"session_start": str, "session_end": str, "turns": list[dict]}
    """
    if not session_turns:
        return
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    history = _load_long_term()
    history.append({
        "session_start": session_turns[0]["timestamp"],
        "session_end": datetime.now().isoformat(),
        "turns": session_turns,
    })
    with open(_LONG_TERM_PATH, "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2, ensure_ascii=False)


def _print_session_summary(session_turns: list[dict]) -> None:
    """Print a one-line-per-turn summary of the current session."""
    if not session_turns:
        return
    print(_hr("-"))
    print(f"  Session summary  ({len(session_turns)} turn(s) saved to memory)")
    for i, turn in enumerate(session_turns, 1):
        tools_str = ", ".join(turn["tools_called"]) if turn["tools_called"] else "none"
        label = _MODE_LABEL.get(turn.get("source_mode", ""), "?")
        query = turn["query"]
        truncated = query[:42] + ("..." if len(query) > 42 else "")
        print(f"    [{i}] [{label}] {truncated}")
        print(f"           tools: {tools_str}")


# --- CLI loop ----------------------------------------------------------------

async def _run_cli() -> None:
    """
    Interactive CLI loop with two-tier memory.

    Session memory   — list of turn dicts held in RAM for the current run.
    Long-term memory — JSON file on disk; session is appended on exit.
    Conversation history — last HISTORY_WINDOW turns passed to the LLM for
                           multi-turn context.
    """
    print(_hr("="))
    print("  PHARMA RAG AGENT")
    print(_hr("="))
    print(f"  Model      : {OLLAMA_MODEL}")
    print(f"  Ollama URL : {OLLAMA_BASE_URL}")
    print(f"  Tools      : search_drug_info, search_competitor_intel,")
    print(f"               search_pitch_content, search_all")
    print(_hr("-"))
    print("  Type your question and press Enter.")
    print(f"  Type any of {sorted(_EXIT_WORDS)} to stop.")
    print(_hr("="))
    print()

    session_turns: list[dict] = []
    conv_history: list = []
    agent = PharmaAgent()

    def _do_exit() -> None:
        _print_session_summary(session_turns)
        _flush_session(session_turns)
        if session_turns:
            print(_hr("-"))
            print(f"  Memory saved to: {_LONG_TERM_PATH.relative_to(_PROJECT_ROOT)}")
        print(_hr("-"))
        print("  Goodbye! Have a great day.")
        print(_hr("="))

    while True:
        try:
            question = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _do_exit()
            break

        if not question:
            continue

        if question.lower() in _EXIT_WORDS:
            _do_exit()
            break

        print(_hr("-"))
        print("  Agent > thinking...")
        print(_hr("-"))
        turn_ts = datetime.now().isoformat()

        try:
            windowed = conv_history[-(HISTORY_WINDOW * 2):]
            answer, tools_called, source_mode = await agent.ask(question, history=windowed)
            wrapped = textwrap.fill(answer, width=_W - 2,
                                    initial_indent="  ", subsequent_indent="  ")
            print(wrapped)
            print(_MODE_BADGE.get(source_mode, ""))
            conv_history.append(HumanMessage(content=question))
            conv_history.append(AIMessage(content=answer))
        except Exception as exc:
            answer = f"[Error] {exc}"
            tools_called = []
            source_mode = "error"
            print(f"  {answer}")

        session_turns.append({
            "timestamp": turn_ts,
            "query": question,
            "tools_called": tools_called,
            "source_mode": source_mode,
            "output": answer,
        })

        print(_hr("="))
        print()


if __name__ == "__main__":
    asyncio.run(_run_cli())
