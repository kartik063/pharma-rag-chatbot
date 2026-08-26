"""
MCP server package for pharma RAG.

Exposes the three ChromaDB vectorstores (drug_info, competitor_intel,
pitch_content) as MCP tools over stdio transport so that any MCP-compatible
client (Claude Desktop, LangChain agent, etc.) can call them by name.

Entry point:
    python -m mcp_server.server
"""
