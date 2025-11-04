# MCP Server From Scratch

A comprehensive guide to building a Model Context Protocol (MCP) server from scratch. This repository demonstrates a complete MCP server implementation with text summarization and mindmap generation tools, using stdio transport for communication with MCP clients.

## What is MCP?

Model Context Protocol (MCP) is a protocol that enables AI assistants to securely access external tools, data sources, and capabilities. MCP servers expose functionality through:

- **Tools**: Executable functions that can be called by AI assistants
- **Resources**: Readable data sources (files, databases, APIs)
- **Prompts**: Template-based prompts that can be customized

## Architecture Overview

This MCP server implementation consists of:

```
llm_tools/
├── mcp_server.py          # Core MCP server logic (Tools, Resources, Prompts)
├── mcp_server_main.py     # Stdio transport & JSON-RPC handler
├── llms/                  # LLM provider abstractions
│   ├── base.py           # Base LLM interface
│   ├── factory.py        # LLM factory pattern
│   ├── openai.py         # OpenAI provider
│   ├── ollama.py         # Ollama provider
│   └── vllm.py           # vLLM provider
├── tools/                 # Tool implementations
│   ├── base.py           # Base tool interface
│   ├── factory.py        # Tool factory pattern
│   ├── summary.py        # Text summarization tool
│   └── mindmap.py        # Mindmap generation tool
└── config.json           # Server configuration
```

## Building an MCP Server: Step by Step

### Step 1: Understand the MCP Protocol

MCP uses JSON-RPC 2.0 over stdio (stdin/stdout). The protocol flow:

1. **Initialize**: Client sends `initialize` request, server responds with capabilities
2. **Initialized**: Server sends `notifications/initialized` notification
3. **Tools List**: Client requests available tools via `tools/list`
4. **Tool Call**: Client calls tools via `tools/call`
5. **Resources/Prompts**: Similar pattern for resources and prompts

### Step 2: Create the Core MCP Server Class

The `MCPServer` class wraps your tools and exposes them via MCP:

```python
class MCPServer:
    """MCP Server wrapper for tools."""
    
    def __init__(self, llm_config, tool_configs):
        # Initialize your tools here
        self.summary_tool = ToolFactory.create_tool("summary", llm_config, tool_configs.get("summary", {}))
        self.mindmap_tool = ToolFactory.create_tool("mindmap", llm_config, tool_configs.get("mindmap", {}))
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP-compatible tool definitions."""
        return [
            {
                "name": "summarize_text",
                "description": "Generate summary from text",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to summarize"}
                    },
                    "required": ["text"]
                }
            }
        ]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name."""
        if name == "summarize_text":
            result = await self.summary_tool.process(arguments)
            return {
                "content": [{"type": "text", "text": json.dumps(result)}],
                "isError": False
            }
```

### Step 3: Implement Stdio Transport

MCP servers communicate via stdin/stdout. Create a transport handler:

```python
class StdioMCPServer:
    """Handles JSON-RPC over stdio."""
    
    async def read_request(self) -> Dict[str, Any]:
        """Read JSON-RPC request from stdin."""
        line_bytes = await asyncio.get_event_loop().run_in_executor(
            None, sys.stdin.buffer.readline
        )
        if not line_bytes:
            return None
        line = line_bytes.decode('utf-8').strip()
        return json.loads(line)
    
    def write_response(self, response: Dict[str, Any]):
        """Write JSON-RPC response to stdout."""
        response_str = json.dumps(response, ensure_ascii=False)
        response_bytes = response_str.encode('utf-8')
        sys.stdout.buffer.write(response_bytes)
        sys.stdout.buffer.write(b'\n')
        sys.stdout.buffer.flush()
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle JSON-RPC request and return response."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                    "serverInfo": self.mcp_server.get_server_info()
                }
            }
        elif method == "tools/list":
            tools = self.mcp_server.list_tools()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools}
            }
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            result = await self.mcp_server.call_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        # ... handle other methods
```

### Step 4: Implement the Main Loop

The server runs in an async loop, reading requests and writing responses:

```python
async def run(self):
    """Run the stdio MCP server."""
    # Send initialized notification
    init_notification = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
    self.write_response(init_notification)
    
    # Main loop
    while True:
        try:
            request = await self.read_request()
            if not request:
                break
            
            response = await self.handle_request(request)
            if response:
                self.write_response(response)
        except Exception as e:
            logger.error("Error: %s", e, exc_info=True)
            continue
```

### Step 5: Define Your Tools

Tools are the core functionality. Each tool needs:

1. **Schema Definition**: JSON schema describing inputs
2. **Execution Logic**: Async function that processes inputs

Example tool definition:

```python
def list_tools(self) -> List[Dict[str, Any]]:
    tools = []
    
    if self.summary_tool:
        tools.append({
            "name": "summarize_text",
            "description": "Generate abstractive or extractive summary from text",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text content to summarize"
                    },
                    "summary_type": {
                        "type": "string",
                        "enum": ["abstractive", "extractive"],
                        "default": "abstractive"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["vietnamese", "english"],
                        "default": "vietnamese"
                    }
                },
                "required": ["text"]
            }
        })
    
    return tools
```

### Step 6: Implement Resources (Optional)

Resources expose readable data. Example:

```python
def list_resources(self) -> List[Dict[str, Any]]:
    """List available resources."""
    return [
        {
            "uri": "prompt://summary/abstractive/vietnamese",
            "name": "Summary Abstractive Prompt (Vietnamese)",
            "description": "Prompt template for abstractive summarization",
            "mimeType": "text/plain"
        }
    ]

async def read_resource(self, uri: str) -> Dict[str, Any]:
    """Read a resource by URI."""
    if uri.startswith("prompt://"):
        # Parse and return prompt content
        prompt_text = get_prompt(...)
        return {
            "contents": [{
                "uri": uri,
                "mimeType": "text/plain",
                "text": prompt_text
            }]
        }
```

### Step 7: Implement Prompts (Optional)

Prompts are template-based prompts that can be customized:

```python
def list_prompts(self) -> List[Dict[str, Any]]:
    """List available prompts."""
    return [
        {
            "name": "summarize_abstractive",
            "description": "Generate abstractive summary from text",
            "arguments": [
                {
                    "name": "text",
                    "description": "Text content to summarize",
                    "required": True
                }
            ]
        }
    ]

async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Get a prompt template with arguments filled in."""
    if name == "summarize_abstractive":
        text = arguments.get("text", "")
        language = arguments.get("language", "vietnamese")
        
        system_prompt = get_prompt("summary", language, "abstractive")
        user_prompt = f"Summarize: {text}"
        
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced functionality:

```bash
# Token counting
pip install tiktoken>=0.5.0

# Transformers (for Qwen models)
pip install transformers>=4.30.0 torch>=2.0.0

# LangChain (for advanced text splitting)
pip install langchain>=0.1.0 langchain-text-splitters>=0.0.1

# Ollama support
pip install ollama>=0.1.0 requests>=2.31.0

# vLLM support
pip install vllm>=0.2.0
```

## Configuration

### Create Config File

Create `config.json`:

```json
{
  "llm_config": {
    "enabled": true,
    "provider": "openai",
    "model_name": "gpt-4",
    "api_key": "your_api_key",
    "base_url": "https://api.openai.com/v1",
    "default_temperature": 0.7,
    "default_max_tokens": 2000
  },
  "tool_configs": {
    "summary": {
      "summary_type": "abstractive",
      "language": "english",
      "max_length": 2000
    },
    "mindmap": {
      "language": "english",
      "max_nodes": 50,
      "max_depth": 4
    }
  }
}
```

### Environment Variables (Alternative)

Create `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
```

## Running the MCP Server

### Basic Usage

```bash
# With default config (Ollama)
python mcp_server_main.py

# With config file
python mcp_server_main.py --config config.json

# With settings module
python mcp_server_main.py --settings-module app.settings
```

### Testing the Server

```bash
python test_mcp_server.py
```

This will:
- Initialize the server
- List available tools
- Test `summarize_text` tool
- Test `create_mindmap` tool
- List resources

## Integration with Claude Desktop

### macOS

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

Edit `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "intramind-tools": {
      "command": "python",
      "args": [
        "E:\\path\\to\\llm_tools\\mcp_server_main.py",
        "--config",
        "E:\\path\\to\\config.json"
      ],
      "env": {
        "OPENAI_API_KEY": "your_api_key"
      }
    }
  }
}
```

Restart Claude Desktop after configuration.

## Using MCP SDK (Alternative)

You can also use the official MCP SDK:

```bash
pip install mcp
```

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp_server():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server_main.py", "--config", "config.json"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
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

asyncio.run(use_mcp_server())
```

## Key Concepts

### JSON-RPC 2.0 Protocol

MCP uses JSON-RPC 2.0. Request format:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_text",
    "arguments": {"text": "..."}
  }
}
```

Response format:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{"type": "text", "text": "..."}],
    "isError": false
  }
}
```

### Stdio Transport

- **No HTTP**: MCP uses stdin/stdout, not HTTP
- **No Ports**: No need to open ports or configure firewall
- **Process-based**: Each client creates its own server process
- **UTF-8 Encoding**: Always use UTF-8 encoding for text

### Error Handling

MCP error response format:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32603,
    "message": "Internal error: ..."
  }
}
```

Standard error codes:
- `-32700`: Parse error
- `-32600`: Invalid Request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error

## Project Structure Explained

### `mcp_server.py`

Core MCP server implementation:
- `MCPServer` class: Wraps tools and exposes MCP interface
- `list_tools()`: Returns tool definitions
- `call_tool()`: Executes tools
- `list_resources()` / `read_resource()`: Resource management
- `list_prompts()` / `get_prompt()`: Prompt templates

### `mcp_server_main.py`

Stdio transport and entry point:
- `StdioMCPServer` class: Handles JSON-RPC over stdio
- `read_request()` / `write_response()`: I/O operations
- `handle_request()`: Routes requests to handlers
- `main()`: Entry point with config loading

### Tool System

Tools are implemented in `tools/`:
- `BaseTool`: Abstract base class
- `SummaryTool`: Text summarization
- `MindmapTool`: Mindmap generation
- Factory pattern for tool creation

### LLM System

LLM providers in `llms/`:
- `BaseLLM`: Abstract interface
- `OpenAILLM`: OpenAI API
- `OllamaLLM`: Local Ollama
- `vLLMLLM`: Local vLLM
- Factory pattern for provider selection

## Debugging

### Enable Debug Logging

Edit `mcp_server_main.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Check Server Logs

Server logs go to stderr. In Claude Desktop:
- **macOS**: `~/Library/Logs/Claude/`
- **Windows**: `%APPDATA%\Claude\logs\`

### Common Issues

**"Command not found"**
- Ensure Python is in PATH
- Or use full path: `"C:\\Python39\\python.exe"`

**"Module not found"**
- Install dependencies: `pip install -r requirements.txt`
- Check Python environment

**"Connection refused"**
- MCP is NOT an HTTP server
- No ports needed - uses stdio
- Check config file format

**Server not responding**
- Check JSON format in config
- Verify API keys
- Enable debug logging

## Extending the Server

### Adding a New Tool

1. Create tool class in `tools/`:

```python
class MyTool(BaseTool):
    async def process(self, input_data, **kwargs):
        # Your logic here
        return {"result": "..."}
```

2. Register in `MCPServer.list_tools()`:

```python
tools.append({
    "name": "my_tool",
    "description": "My tool description",
    "inputSchema": {...}
})
```

3. Add handler in `MCPServer.call_tool()`:

```python
if name == "my_tool":
    return await self._call_my_tool(arguments)
```

### Adding a New LLM Provider

1. Create provider class in `llms/`:

```python
class MyLLM(BaseLLM):
    async def generate(self, prompt, **kwargs):
        # Your logic here
        return response
```

2. Register in `LLMFactory`:

```python
if provider == "my_llm":
    return MyLLM(config)
```

## Best Practices

1. **Always validate inputs**: Check tool arguments before processing
2. **Handle errors gracefully**: Return proper error responses
3. **Use async/await**: MCP operations should be async
4. **Log appropriately**: Use logging for debugging, not stdout
5. **UTF-8 encoding**: Always use UTF-8 for text I/O
6. **Idempotent tools**: Tools should be safe to retry
7. **Schema validation**: Use JSON Schema for input validation
8. **Resource cleanup**: Clean up resources in error cases

## Testing

### Manual Testing

Use the test script:

```bash
python test_mcp_server.py
```

### Unit Testing

Test individual components:

```python
import pytest
from mcp_server import MCPServer

async def test_summarize_tool():
    server = MCPServer(llm_config, tool_configs)
    result = await server.call_tool(
        "summarize_text",
        {"text": "Test text"}
    )
    assert "content" in result
```

## Resources

- [MCP Specification](https://modelcontextprotocol.io/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [Claude Desktop MCP Guide](https://docs.anthropic.com/claude/docs/mcp)

## License

[Add your license here]

## Contributing

Contributions welcome! Please submit pull requests or open issues for bugs and feature requests.
