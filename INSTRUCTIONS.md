# MCP Tools Usage Instructions

## Table of Contents

1. [Overview](#overview)
2. [What is MCP?](#what-is-mcp)
3. [MCP Architecture](#mcp-architecture)
4. [Tools in MCP](#tools-in-mcp)
5. [Workflow: LLM Using MCP Tools](#workflow-llm-using-mcp-tools)
6. [Tool Implementation Details](#tool-implementation-details)
7. [Best Practices](#best-practices)

## Overview

This document explains how the Model Context Protocol (MCP) works, how tools are exposed and used, and the workflow when an LLM interacts with MCP tools for text processing tasks like summarization and mindmap generation.

## What is MCP?

**Model Context Protocol (MCP)** is a standardized protocol that enables AI applications (like LLMs) to interact with external tools and resources. It provides:

- **Tool Discovery**: LLMs can discover available tools and their capabilities
- **Tool Execution**: LLMs can call tools with parameters and receive structured results
- **Resource Access**: LLMs can access external resources like prompts, templates, or data
- **Standardized Communication**: Uses JSON-RPC 2.0 over stdio for reliable communication

### Key Benefits

1. **Separation of Concerns**: Tools are separate from the LLM, allowing independent development
2. **Reusability**: Tools can be used by multiple LLM applications
3. **Extensibility**: New tools can be added without modifying the LLM
4. **Standardization**: Follows MCP specification for compatibility

## MCP Architecture

### Communication Flow

```
┌─────────────┐         JSON-RPC 2.0         ┌─────────────┐
│   LLM       │ ←──────────────────────────→ │  MCP Server │
│ (Client)    │      (via stdin/stdout)       │  (Server)   │
└─────────────┘                                └─────────────┘
                                                      │
                                                      │
                                                ┌─────┴─────┐
                                                │   Tools    │
                                                │  (Summary, │
                                                │  Mindmap)  │
                                                └────────────┘
```

### Components

1. **MCP Client (LLM)**: Initiates requests, sends tool calls, receives results
2. **MCP Server**: Processes requests, routes to appropriate tools, returns responses
3. **Tools**: Actual implementations that perform the work (e.g., summarize text, create mindmap)
4. **Transport**: stdio (stdin/stdout) for communication

### Protocol Layers

```
┌─────────────────────────────────────┐
│  Application Layer (LLM + Tools)    │
├─────────────────────────────────────┤
│  MCP Protocol Layer                 │
│  (JSON-RPC 2.0 messages)            │
├─────────────────────────────────────┤
│  Transport Layer (stdio)            │
└─────────────────────────────────────┘
```

## Tools in MCP

### Tool Definition

A tool in MCP is defined with:
- **Name**: Unique identifier (e.g., `summarize_text`, `create_mindmap`)
- **Description**: What the tool does
- **Input Schema**: Parameters the tool accepts (JSON Schema)
- **Output Format**: Structure of the result

### Available Tools

#### 1. Summarize Text Tool

**Name**: `summarize_text`

**Purpose**: Generate abstractive or extractive summaries from text input.

**Input Schema**:
```json
{
  "text": "string (required) - Text content to summarize",
  "summary_type": "string (optional) - 'abstractive' or 'extractive'",
  "language": "string (optional) - 'vietnamese' or 'english'",
  "max_length": "integer (optional) - Maximum summary length in words"
}
```

**Output Format**:
```json
{
  "summary": "string - Generated summary",
  "metadata": {
    "summary_type": "string",
    "language": "string",
    "processing_time_seconds": "float"
  }
}
```

#### 2. Create Mindmap Tool

**Name**: `create_mindmap`

**Purpose**: Generate hierarchical mindmap structures from text.

**Input Schema**:
```json
{
  "text": "string (required) - Text content to create mindmap from",
  "language": "string (optional) - 'vietnamese' or 'english'",
  "max_nodes": "integer (optional) - Maximum number of nodes",
  "max_depth": "integer (optional) - Maximum depth",
  "output_format": "string (optional) - 'json', 'markdown', 'mermaid', or 'html'"
}
```

**Output Format**:
```json
{
  "mindmap": "object - Hierarchical mindmap structure",
  "formats": {
    "json": "object",
    "markdown": "string",
    "mermaid": "string",
    "html": "string"
  },
  "metadata": {
    "language": "string",
    "output_format": "string",
    "node_count": "integer",
    "max_depth": "integer",
    "processing_time_seconds": "float"
  }
}
```

## Workflow: LLM Using MCP Tools

### Step-by-Step Process

#### 1. Initialization Phase

```
LLM → MCP Server: Initialize Request
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "llm-client",
      "version": "1.0.0"
    }
  }
}

MCP Server → LLM: Initialize Response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {...},
    "serverInfo": {
      "name": "intramind-tools",
      "version": "1.0.0"
    }
  }
}
```

#### 2. Tool Discovery Phase

```
LLM → MCP Server: List Tools Request
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}

MCP Server → LLM: List Tools Response
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "summarize_text",
        "description": "Generate abstractive or extractive summary",
        "inputSchema": {...}
      },
      {
        "name": "create_mindmap",
        "description": "Generate hierarchical mindmap structure",
        "inputSchema": {...}
      }
    ]
  }
}
```

#### 3. Tool Execution Phase

**Scenario: User asks LLM to summarize a document**

```
User → LLM: "Please summarize this text: [long text]"

LLM (Decision):
  - User wants summarization
  - Available tool: "summarize_text"
  - Extract parameters from user request
  - Call tool with appropriate parameters

LLM → MCP Server: Tool Call Request
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "summarize_text",
    "arguments": {
      "text": "[user's text]",
      "summary_type": "abstractive",
      "language": "english",
      "max_length": 200
    }
  }
}

MCP Server (Processing):
  1. Validate tool name exists
  2. Validate input parameters
  3. Route to appropriate tool handler
  4. Execute tool processing:
     - Text chunking (if needed)
     - LLM API call for summarization
     - Result formatting
  5. Return structured result

MCP Server → LLM: Tool Call Response
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"summary\": \"[generated summary]\", \"metadata\": {...}}"
      }
    ],
    "isError": false
  }
}

LLM (Response):
  - Parse tool result
  - Extract summary from JSON
  - Format response for user
  - Return to user

LLM → User: "Here's the summary: [summary text]"
```

### Complete Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Workflow                            │
└─────────────────────────────────────────────────────────────┘

1. Initialize Connection
   └─→ Handshake with MCP Server
   └─→ Exchange capabilities

2. Discover Available Tools
   └─→ Request tool list
   └─→ Cache tool definitions

3. User Request Arrives
   └─→ Parse user intent
   └─→ Determine if tool is needed

4. Tool Selection
   └─→ Match user intent to tool
   └─→ Extract parameters from user input
   └─→ Validate parameters

5. Tool Execution
   └─→ Send tool call request
   └─→ Wait for response
   └─→ Handle errors if any

6. Result Processing
   └─→ Parse tool result
   └─→ Extract relevant information
   └─→ Format for user

7. User Response
   └─→ Present result to user
   └─→ Provide context if needed
```

## Tool Implementation Details

### Tool Processing Pipeline

#### Summarize Tool Pipeline

```
Input Text
    │
    ├─→ Validate Input
    │   └─→ Check text presence
    │   └─→ Validate text length
    │
    ├─→ Text Chunking (if needed)
    │   └─→ Split into manageable chunks
    │   └─→ Preserve context
    │
    ├─→ Map-Reduce Processing
    │   ├─→ Map: Summarize each chunk
    │   ├─→ Reduce: Combine summaries
    │   └─→ Recursive Collapse (if needed)
    │
    ├─→ LLM API Call
    │   └─→ Send to OpenAI/Ollama/vLLM
    │   └─→ Receive summary
    │
    └─→ Result Formatting
        └─→ Structure output
        └─→ Add metadata
        └─→ Return to MCP Server
```

#### Mindmap Tool Pipeline

```
Input Text
    │
    ├─→ Validate Input
    │   └─→ Check text presence
    │
    ├─→ Structure-Aware Chunking
    │   └─→ Detect headings, lists, paragraphs
    │   └─→ Preserve document structure
    │
    ├─→ Hierarchical Processing
    │   ├─→ Process chunks
    │   ├─→ Extract key concepts
    │   ├─→ Build relationships
    │   └─→ Create hierarchy
    │
    ├─→ LLM API Call
    │   └─→ Generate mindmap structure
    │   └─→ Extract nodes and edges
    │
    ├─→ Format Conversion
    │   └─→ JSON format
    │   └─→ Markdown format
    │   └─→ Mermaid format
    │   └─→ HTML format
    │
    └─→ Result Formatting
        └─→ Structure output
        └─→ Add metadata
        └─→ Return to MCP Server
```

### Error Handling

```
Tool Call Request
    │
    ├─→ Validation Error
    │   └─→ Return error with details
    │
    ├─→ Tool Not Found
    │   └─→ Return error: "Tool not found"
    │
    ├─→ Processing Error
    │   └─→ Log error
    │   └─→ Return error with message
    │
    └─→ Success
        └─→ Return result
```

### Async Processing

MCP tools use async/await for:
- Non-blocking I/O operations
- Concurrent processing of multiple chunks
- Efficient resource utilization
- Better scalability

## Best Practices

### For LLM Developers

1. **Always Initialize First**
   - Establish connection before any operations
   - Handle initialization errors gracefully

2. **Cache Tool Definitions**
   - Discover tools once at startup
   - Cache tool schemas for validation

3. **Validate Before Calling**
   - Check tool availability
   - Validate parameters against schema
   - Provide clear error messages

4. **Handle Errors Gracefully**
   - Check `isError` flag in responses
   - Provide fallback behavior
   - Log errors for debugging

5. **Use Appropriate Timeouts**
   - Set reasonable timeouts for tool calls
   - Consider text length and complexity
   - Provide progress feedback for long operations

### For Tool Developers

1. **Clear Tool Descriptions**
   - Describe what the tool does
   - Explain input parameters
   - Document output format

2. **Robust Input Validation**
   - Validate all required parameters
   - Check parameter types and ranges
   - Provide clear error messages

3. **Structured Output**
   - Use consistent output format
   - Include metadata (processing time, etc.)
   - Handle edge cases gracefully

4. **Error Handling**
   - Catch and handle exceptions
   - Return structured error responses
   - Log errors for debugging

5. **Performance Considerations**
   - Use async processing for I/O
   - Implement chunking for large inputs
   - Optimize LLM API calls

### For Users

1. **Provide Clear Instructions**
   - Specify what you want done
   - Include relevant context
   - Mention any preferences (language, format, etc.)

2. **Understand Limitations**
   - Large texts may take time
   - Some operations have token limits
   - Complex requests may require multiple tool calls

3. **Iterate if Needed**
   - If result is not satisfactory, refine request
   - Try different parameters
   - Break complex tasks into steps

## Example Scenarios

### Scenario 1: Simple Summarization

**User Request**: "Summarize this article about AI"

**LLM Process**:
1. Identifies need for summarization
2. Calls `summarize_text` tool
3. Extracts summary from result
4. Presents to user

**Tool Call**:
```json
{
  "name": "summarize_text",
  "arguments": {
    "text": "[article content]",
    "summary_type": "abstractive",
    "language": "english"
  }
}
```

### Scenario 2: Complex Mindmap Creation

**User Request**: "Create a mindmap of this technical document in Markdown format"

**LLM Process**:
1. Identifies need for mindmap
2. Extracts format preference (Markdown)
3. Calls `create_mindmap` tool with format parameter
4. Extracts Markdown from result
5. Presents formatted mindmap to user

**Tool Call**:
```json
{
  "name": "create_mindmap",
  "arguments": {
    "text": "[document content]",
    "language": "english",
    "output_format": "markdown",
    "max_nodes": 50,
    "max_depth": 4
  }
}
```

### Scenario 3: Multi-Step Processing

**User Request**: "Summarize this document and then create a mindmap of the summary"

**LLM Process**:
1. First tool call: `summarize_text` to get summary
2. Extract summary from first result
3. Second tool call: `create_mindmap` with summary as input
4. Combine both results
5. Present to user

## Conclusion

MCP provides a powerful and standardized way for LLMs to interact with external tools. By following the protocol and best practices, developers can create robust, reusable tools that enhance LLM capabilities for specific tasks like text summarization and mindmap generation.

The workflow is designed to be:
- **Transparent**: Clear communication between LLM and tools
- **Reliable**: Error handling and validation at each step
- **Extensible**: Easy to add new tools without modifying existing code
- **Efficient**: Async processing for optimal performance

Understanding this architecture helps in:
- Developing better LLM applications
- Creating effective MCP tools
- Debugging issues when they occur
- Optimizing performance

