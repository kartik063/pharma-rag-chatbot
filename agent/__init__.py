"""
Agent package for pharma RAG.

Contains the PharmaAgent class which connects to the MCP server,
loads its retrieval tools, and runs a LangGraph ReAct agent loop
backed by a local Ollama LLM to answer questions about drugs,
clinical trials, and sales strategy.

Entry point:
    python -m agent.agent
"""
