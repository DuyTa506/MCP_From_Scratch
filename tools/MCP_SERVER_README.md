# MCP Server for IntraMind Tools

This file describes how to use the MCP (Model Context Protocol) Server to expose tools.

## Overview

The MCP Server provides a standard MCP interface for the following tools:
- **SummaryTool**: Text summarization (abstractive/extractive)
- **MindmapTool**: Create mindmap structures from text

## Structure

```
tools/
├── mcp_server.py          # Core MCP server implementation
├── mcp_server_main.py     # Standalone server runner with stdio transport
└── MCP_SERVER_README.md   # This file
```

## Installation and Usage

### 1. Using as a module

```python
from tools import MCPServer, create_mcp_server_from_settings

# From settings
from app.settings import settings
mcp_server = create_mcp_server_from_settings(settings)

# Or create directly
llm_config = {
    "provider": "ollama",
    "model_name": "llama3.2",
    "temperature": 0.7
}

tool_configs = {
    "summary": {
        "summary_type": "abstractive",
        "language": "vietnamese",
        "max_length": 2000
    },
    "mindmap": {
        "language": "vietnamese",
        "max_nodes": 50,
        "max_depth": 4
    }
}

mcp_server = MCPServer(llm_config, tool_configs)
```

### 2. Running standalone server

```bash
# With default config
python -m tools.mcp_server_main

# With config file
python -m tools.mcp_server_main --config config.json

# With settings module
python -m tools.mcp_server_main --settings-module app.settings
```

### 3. Config file format (JSON)

```json
{
  "llm_config": {
    "enabled": true,
    "provider": "ollama",
    "model_name": "llama3.2",
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "tool_configs": {
    "summary": {
      "summary_type": "abstractive",
      "language": "vietnamese",
      "max_length": 2000
    },
    "mindmap": {
      "language": "vietnamese",
      "max_nodes": 50,
      "max_depth": 4
    }
  }
}
```

## API Reference

### Tools

#### `list_tools() -> List[Dict]`
List all available tools.

**Returns:**
- `summarize_text`: Summarize text
- `create_mindmap`: Create mindmap

#### `call_tool(name: str, arguments: Dict) -> Dict`
Call a tool.

**Parameters:**
- `name`: Tool name (`summarize_text` or `create_mindmap`)
- `arguments`: Tool arguments

**Example:**
```python
result = await mcp_server.call_tool("summarize_text", {
    "text": "Text to summarize...",
    "summary_type": "abstractive",
    "language": "vietnamese",
    "max_length": 1000
})
```

### Resources

#### `list_resources() -> List[Dict]`
List all available resources (prompt templates).

**Available resources:**
- `prompt://summary/abstractive/vietnamese`
- `prompt://summary/extractive/vietnamese`
- `prompt://summary/abstractive/english`
- `prompt://summary/extractive/english`
- `prompt://mindmap/vietnamese`
- `prompt://mindmap/english`
- `prompt://mindmap/markdown/vietnamese`
- `prompt://mindmap/markdown/english`

#### `read_resource(uri: str) -> Dict`
Read resource by URI.

**Example:**
```python
result = await mcp_server.read_resource("prompt://summary/abstractive/vietnamese")
```

### Prompts

#### `list_prompts() -> List[Dict]`
List all available prompt templates.

**Available prompts:**
- `summarize_abstractive`: Abstractive summarization
- `summarize_extractive`: Extractive summarization
- `create_mindmap_json`: Create JSON mindmap
- `create_mindmap_markdown`: Create Markdown mindmap

#### `get_prompt(name: str, arguments: Dict) -> Dict`
Get prompt template with filled arguments.

**Example:**
```python
prompt = await mcp_server.get_prompt("summarize_abstractive", {
    "text": "Text to summarize...",
    "language": "vietnamese",
    "max_length": 1000
})
```

## JSON-RPC Protocol

The server uses JSON-RPC 2.0 protocol over stdio transport.

### Request Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_text",
    "arguments": {
      "text": "...",
      "language": "vietnamese"
    }
  }
}
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "..."
      }
    ],
    "isError": false
  }
}
```

## Integration with Claude Desktop

To integrate with Claude Desktop, add to the config file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "intramind-tools": {
      "command": "python",
      "args": [
        "-m",
        "tools.mcp_server_main",
        "--settings-module",
        "app.settings"
      ]
    }
  }
}
```

## Integration with other applications

The MCP server can be integrated with any application that supports the MCP protocol:

1. **Claude Desktop**: See section above
2. **Custom MCP Client**: Use JSON-RPC over stdio
3. **HTTP Server wrapper**: Can wrap stdio server into HTTP endpoint

## Troubleshooting

### Tool initialization errors
- Check if LLM config is correct
- Ensure LLM provider is installed and configured

### Server runtime errors
- Check Python path
- Ensure all dependencies are installed
- Check log output for error details

### Connection errors
- Check if stdio transport is working
- Ensure JSON-RPC format is correct

## License

Part of IntraMind RAG System.