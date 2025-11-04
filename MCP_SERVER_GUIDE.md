# MCP Server Guide

## Overview

The MCP Server uses **stdio transport** (stdin/stdout) to communicate via MCP (Model Context Protocol). This is the standard way to integrate with MCP clients like Claude Desktop.

## How to Run

### 1. Run with Default Config

```bash
python mcp_server_main.py
```

Default config uses Ollama with model `llama3.2`.

### 2. Run with Config File

Create a `config.json` file:

```json
{
  "llm_config": {
    "enabled": true,
    "provider": "openai",
    "model_name": "Arcee-Vylinh",
    "api_key": "your_api_key",
    "base_url": "https://lqd-test1.hpda.vn/v1",
    "default_temperature": 0.7,
    "default_max_tokens": 2000
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

Run:

```bash
python mcp_server_main.py --config config.json
```

### 3. Run with Settings Module

If you have a settings module in your project:

```bash
python mcp_server_main.py --settings-module app.settings
```

## Integration with Claude Desktop

### macOS

1. Find config file: `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Add config:

```json
{
  "mcpServers": {
    "intramind-tools": {
      "command": "python",
      "args": [
        "/absolute/path/to/llm_tools/mcp_server_main.py",
        "--config",
        "/absolute/path/to/config.json"
      ]
    }
  }
}
```

### Windows

1. Tìm file config: `%APPDATA%\Claude\claude_desktop_config.json`

2. Thêm config:

```json
{
  "mcpServers": {
    "intramind-tools": {
      "command": "python",
      "args": [
        "E:\\absolute\\path\\to\\llm_tools\\mcp_server_main.py",
        "--config",
        "E:\\absolute\\path\\to\\config.json"
      ],
      "env": {
        "OPENAI_API_KEY": "your_api_key"
      }
    }
  }
}
```

3. Restart Claude Desktop

## Testing MCP Server

### Quick Test

Run the test script:

```bash
python test_mcp_server.py
```

This will:
- Start the MCP server
- Test initialization
- List available tools
- Test summarize_text tool
- Test create_mindmap tool
- List resources

### Using Python

```python
import asyncio
import json
import subprocess
import sys

# See test_mcp_server.py for a complete example
```

### Using MCP SDK

```bash
pip install mcp
```

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_with_mcp_sdk():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server_main.py", "--config", "config.json"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            print("Tools:", tools)
            
            # Call tool
            result = await session.call_tool(
                "summarize_text",
                {"text": "Your text here..."}
            )
            print("Result:", result)

asyncio.run(test_with_mcp_sdk())
```

## Debugging

### Enable verbose logging

Edit `mcp_server_main.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check stderr

MCP server writes errors to stderr. If using Claude Desktop, check logs in:
- macOS: `~/Library/Logs/Claude/`
- Windows: `%APPDATA%\Claude\logs\`

## Troubleshooting

### Error: "Command not found"

- Ensure Python is in PATH
- Or use full path: `"C:\\Python39\\python.exe"`

### Error: "Module not found"

- Install dependencies: `pip install -r requirements.txt`
- Ensure you're in the correct directory

### Error: "Connection refused"

- MCP server is NOT an HTTP server
- It communicates via stdio, no port needed
- No need to open any ports

### Server not responding

- Check config file format is correct
- Check API key is valid
- Enable debug logging to see details

## Important Notes

1. **MCP server is NOT an HTTP server** - it communicates via stdin/stdout
2. **No port** - no need to open ports or configure firewall
3. **Designed for MCP clients** - integrates well with Claude Desktop, MCP SDK
4. **Stdio transport** - each client creates its own process

