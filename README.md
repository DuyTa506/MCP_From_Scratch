# IntraMind LLM Tools

A comprehensive toolkit for text processing using Large Language Models (LLMs), featuring text summarization and mindmap generation capabilities. The tools support multiple LLM providers and can be used as a standalone library or via Model Context Protocol (MCP) server.

## Features

- **Text Summarization**: Generate abstractive or extractive summaries with support for long documents using map-reduce and recursive collapse strategies
- **Mindmap Generation**: Create hierarchical mindmaps from text in multiple formats (JSON, Markdown, Mermaid, HTML)
- **Multi-LLM Support**: Compatible with OpenAI, Ollama, and vLLM providers
- **MCP Server**: Expose tools via Model Context Protocol for integration with MCP-compatible clients
- **Async Support**: Full async/await support for efficient processing
- **Multi-language**: Support for Vietnamese and English prompts

## Installation

### Basic Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced functionality, install optional dependencies:

```bash
# Token counting (improves accuracy)
pip install tiktoken>=0.5.0

# Transformers (for Qwen models)
pip install transformers>=4.30.0 torch>=2.0.0

# LangChain (for advanced text splitting)
pip install langchain>=0.1.0 langchain-text-splitters>=0.0.1

# Ollama support (for local inference)
pip install ollama>=0.1.0 requests>=2.31.0

# vLLM support (for local inference)
pip install vllm>=0.2.0
```

## Quick Start

### 1. Setup Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_BASE_URL=https://lqd-test1.hpda.vn/v1
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=Arcee-Vylinh

# Tool Configuration
SUMMARY_DEFAULT_TYPE=abstractive
TOOLS_LANGUAGE=vietnamese
SUMMARY_MAX_TOKENS=2000
MINDMAP_DEFAULT_NODES=50
MINDMAP_DEFAULT_DEPTH=4
```

You can copy `.env.example` to `.env` and update the values.

### 2. Run Demo

```bash
python demo.py
```

This will demonstrate:
- Text summarization
- Mindmap generation (JSON format)
- Mindmap generation (Markdown format)

## Usage

### As a Python Library

#### Text Summarization

```python
import asyncio
from tools.factory import ToolFactory
from llms.factory import LLMFactory

async def summarize_text():
    # Configure LLM
    llm_config = {
        "enabled": True,
        "provider": "openai",
        "model_name": "Arcee-Vylinh",
        "api_key": "your_api_key",
        "base_url": "",
        "default_temperature": 0.7,
        "default_max_tokens": 2000
    }
    
    # Create LLM
    llm = LLMFactory.create_llm(llm_config)
    
    # Create summary tool
    tool_config = {
        "summary_type": "abstractive",
        "language": "vietnamese",
        "max_length": 2000
    }
    summary_tool = ToolFactory.create_tool("summary", llm, tool_config)
    
    # Process text
    result = await summary_tool.process({
        "text": "Your long text here..."
    })
    
    print(result["summary"])
    print(result["metadata"])

asyncio.run(summarize_text())
```

#### Mindmap Generation

```python
import asyncio
from tools.factory import ToolFactory
from llms.factory import LLMFactory

async def create_mindmap():
    # Configure LLM
    llm_config = {
        "enabled": True,
        "provider": "openai",
        "model_name": "Arcee-Vylinh",
        "api_key": "your_api_key",
        "base_url": "https://lqd-test1.hpda.vn/v1",
        "default_temperature": 0.5,
        "default_max_tokens": 2000
    }
    
    # Create LLM
    llm = LLMFactory.create_llm(llm_config)
    
    # Create mindmap tool
    tool_config = {
        "language": "vietnamese",
        "max_nodes": 50,
        "max_depth": 4
    }
    mindmap_tool = ToolFactory.create_tool("mindmap", llm, tool_config)
    
    # Process text
    result = await mindmap_tool.process({
        "text": "Your text here...",
        "output_format": "markdown"  # or "json", "mermaid", "html"
    })
    
    print(result["mindmap"])
    print(result["metadata"])

asyncio.run(create_mindmap())
```

### As MCP Server

The MCP Server uses **stdio transport** (stdin/stdout) to communicate via MCP protocol. This is the standard way to integrate with MCP clients like Claude Desktop.

#### Run MCP Server

**1. Run with default config (Ollama):**

```bash
python mcp_server_main.py
```

**2. Run with config file (JSON):**

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

**3. Run with settings module:**

```bash
python mcp_server_main.py --settings-module app.settings
```

#### Integration with Claude Desktop

Add to Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "intramind-tools": {
      "command": "python",
      "args": [
        "E:/V4_IntraMind_BK/t1-notebook-lm-v2/production_version/llm_tools/mcp_server_main.py",
        "--config",
        "E:/path/to/config.json"
      ],
      "env": {
        "OPENAI_API_KEY": "your_api_key"
      }
    }
  }
}
```

**Note:** The MCP server runs via stdio, so it does not run directly as an HTTP server. It is designed to communicate with MCP clients via stdin/stdout.

#### Use MCP Server in Code

```python
import asyncio
from mcp_server import MCPServer

async def use_mcp_server():
    # Configure LLM and tools
    llm_config = {
        "enabled": True,
        "provider": "openai",
        "model_name": "Arcee-Vylinh",
        "api_key": "your_api_key",
        "base_url": "https://lqd-test1.hpda.vn/v1",
        "default_temperature": 0.7,
        "default_max_tokens": 2000
    }
    
    tool_configs = {
        "summary": {
            "summary_type": "abstractive",
            "language": "vietnamese"
        },
        "mindmap": {
            "language": "vietnamese",
            "max_nodes": 50
        }
    }
    
    # Create MCP server
    mcp_server = MCPServer(llm_config, tool_configs)
    
    # List available tools
    tools = mcp_server.list_tools()
    print(tools)
    
    # Call summarize tool
    result = await mcp_server.call_tool(
        "summarize_text",
        {"text": "Your text here..."}
    )
    print(result)

asyncio.run(use_mcp_server())
```

## Configuration

### LLM Configuration

```python
llm_config = {
    "enabled": True,              # Enable/disable LLM
    "provider": "openai",          # Provider: "openai", "ollama", "vllm"
    "model_name": "Arcee-Vylinh", # Model name
    "api_key": "your_key",         # API key (for OpenAI)
    "base_url": "https://...",     # Base URL (for OpenAI)
    "default_temperature": 0.7,    # Default temperature
    "default_max_tokens": 2000     # Default max tokens
}
```

### Summary Tool Configuration

```python
tool_config = {
    "summary_type": "abstractive",  # "abstractive" or "extractive"
    "language": "vietnamese",        # "vietnamese" or "english"
    "max_length": 2000              # Maximum summary length
}
```

### Mindmap Tool Configuration

```python
tool_config = {
    "language": "vietnamese",        # "vietnamese" or "english"
    "max_nodes": 50,                # Maximum number of nodes
    "max_depth": 4,                 # Maximum depth
    "context_window": 16000,        # Context window size
    "chunk_size_tokens": 2048,      # Chunk size for processing
    "chunk_overlap_tokens": 200     # Chunk overlap
}
```

## MCP Server Usage

### How to Run MCP Server

The MCP Server uses **stdio transport** (stdin/stdout) to communicate via MCP protocol. This is the standard way to integrate with MCP clients.

#### 1. Run Directly

```bash
# Default config (Ollama)
python mcp_server_main.py

# With config file
python mcp_server_main.py --config config.json

# With settings module
python mcp_server_main.py --settings-module app.settings
```

#### 2. Integration with Claude Desktop

Add to Claude Desktop config (`claude_desktop_config.json` on macOS or Windows):

**macOS:**
```json
{
  "mcpServers": {
    "intramind-tools": {
      "command": "python",
      "args": [
        "/path/to/llm_tools/mcp_server_main.py",
        "--config",
        "/path/to/config.json"
      ]
    }
  }
}
```

**Windows:**
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

#### 3. Test MCP Server

To test the MCP server, you can use the MCP SDK or send JSON-RPC messages directly:

```python
import asyncio
import json
import subprocess

async def test_mcp_server():
    # Start MCP server as subprocess
    process = await asyncio.create_subprocess_exec(
        'python', 'mcp_server_main.py',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Send initialize request
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    process.stdin.write(json.dumps(init_request).encode() + b'\n')
    await process.stdin.drain()
    
    # Read response
    response_line = await process.stdout.readline()
    response = json.loads(response_line.decode())
    print("Response:", response)
    
    process.terminate()

asyncio.run(test_mcp_server())
```

#### 4. Config File Format

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

**Important Notes:**
- MCP server does **NOT** run as an HTTP server
- It communicates via **stdio** (stdin/stdout)
- Designed to integrate with MCP clients (like Claude Desktop)
- To test, you need to use MCP SDK or send JSON-RPC messages directly

## Architecture

```
llm_tools/
├── llms/              # LLM provider implementations
│   ├── base.py       # Base LLM class
│   ├── factory.py    # LLM factory
│   ├── openai.py     # OpenAI provider
│   ├── ollama.py     # Ollama provider
│   └── vllm.py       # vLLM provider
├── tools/             # Tool implementations
│   ├── base.py       # Base tool class
│   ├── factory.py    # Tool factory
│   ├── summary.py    # Summary tool
│   ├── mindmap.py    # Mindmap tool
│   └── mindmap_utils/ # Mindmap utilities
├── mcp_server.py     # MCP server implementation
├── mcp_server_main.py # MCP server entry point (stdio transport)
└── demo.py           # Demo script
```

## Supported LLM Providers

### OpenAI (and compatible APIs)

- Supports any OpenAI-compatible API
- Async support with native AsyncOpenAI client
- Streaming support
- Function calling support

### Ollama (Local Inference)

- Local LLM inference
- Supports all Ollama models
- Async support with native AsyncClient

### vLLM (Local Inference)

- High-performance local inference
- Supports various model architectures
- Optimized for production use

## Text Processing Strategies

### Summarization

- **Map-Reduce**: Split large text into chunks, summarize each, then combine
- **Recursive Collapse**: Recursively collapse intermediate summaries to maintain context
- **Structure-Aware Chunking**: Preserves document structure during chunking

### Mindmap Generation

- **Structure-Aware Chunking**: Detects headings, lists, and paragraphs
- **Hierarchical Processing**: Builds mindmap hierarchy from text structure
- **Multi-Format Output**: Supports JSON, Markdown, Mermaid, and HTML formats

## Error Handling

The tools include comprehensive error handling:

- Input validation
- Graceful degradation on tokenizer failures
- Fallback mechanisms for chunking
- Detailed error messages

## Logging

The library uses Python's standard logging module. Configure logging level:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license here]

## Support

For issues and questions, please open an issue on the project repository.

