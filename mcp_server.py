"""
MCP (Model Context Protocol) Server for IntraMind Tools.
Exposes SummaryTool and MindmapTool as MCP-compatible tools.

This server provides:
- Tools: summary, mindmap generation
- Resources: prompt templates
- Prompts: summary and mindmap generation prompts
"""

import json
import logging
from typing import Dict, Any, Optional, List

from tools.factory import ToolFactory
from tools.prompts import get_prompt

logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP Server wrapper for IntraMind tools.
    
    Provides MCP-compatible interface for:
    - Summary generation
    - Mindmap creation
    """
    
    def __init__(self, llm_config: Dict[str, Any], tool_configs: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize MCP Server with LLM and tool configurations.
        
        Args:
            llm_config: LLM configuration dictionary
            tool_configs: Optional tool-specific configurations
        """
        self.llm_config = llm_config
        self.tool_configs = tool_configs or {}
        
        # Initialize tools
        self.summary_tool = None
        self.mindmap_tool = None
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize tool instances."""
        try:
            # Create summary tool
            summary_config = self.tool_configs.get("summary", {})
            self.summary_tool = ToolFactory.create_tool(
                "summary",
                self.llm_config,
                summary_config
            )
            logger.info("✓ Summary tool initialized")
        except Exception as e:
            logger.error("Failed to initialize summary tool: %s", e)
        
        try:
            # Create mindmap tool
            mindmap_config = self.tool_configs.get("mindmap", {})
            self.mindmap_tool = ToolFactory.create_tool(
                "mindmap",
                self.llm_config,
                mindmap_config
            )
            logger.info("✓ Mindmap tool initialized")
        except Exception as e:
            logger.error("Failed to initialize mindmap tool: %s", e)
    
    # ==================== MCP Tools Interface ====================
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available MCP tools.
        
        Returns:
            List of tool definitions compatible with MCP protocol
        """
        tools = []
        
        # Summary tool
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
                            "default": "abstractive",
                            "description": "Type of summary to generate"
                        },
                        "language": {
                            "type": "string",
                            "enum": ["vietnamese", "english"],
                            "default": "vietnamese",
                            "description": "Language for summary output"
                        },
                        "max_length": {
                            "type": "integer",
                            "default": 2000,
                            "description": "Maximum length for summary (in words)"
                        }
                    },
                    "required": ["text"]
                }
            })
        
        # Mindmap tool
        if self.mindmap_tool:
            tools.append({
                "name": "create_mindmap",
                "description": "Generate hierarchical mindmap structure from text",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text content to create mindmap from"
                        },
                        "language": {
                            "type": "string",
                            "enum": ["vietnamese", "english"],
                            "default": "vietnamese",
                            "description": "Language for mindmap output"
                        },
                        "max_nodes": {
                            "type": "integer",
                            "default": 50,
                            "description": "Maximum number of nodes in mindmap"
                        },
                        "max_depth": {
                            "type": "integer",
                            "default": 4,
                            "description": "Maximum depth levels in mindmap"
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["json", "markdown", "mermaid", "html"],
                            "default": "json",
                            "description": "Output format for mindmap"
                        },
                        "temperature": {
                            "type": "number",
                            "default": 0.5,
                            "description": "LLM temperature for generation"
                        }
                    },
                    "required": ["text"]
                }
            })
        
        return tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool by name.
        
        Args:
            name: Tool name to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            if name == "summarize_text":
                return await self._call_summarize(arguments)
            elif name == "create_mindmap":
                return await self._call_mindmap(arguments)
            else:
                return {
                    "error": True,
                    "message": f"Unknown tool: {name}",
                    "available_tools": [tool["name"] for tool in self.list_tools()]
                }
        except Exception as e:
            logger.error("Error calling tool %s: %s", name, e, exc_info=True)
            return {
                "error": True,
                "message": str(e),
                "tool": name
            }
    
    async def _call_summarize(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to call summarize tool."""
        if not self.summary_tool:
            return {
                "error": True,
                "message": "Summary tool not initialized"
            }
        
        # Validate text is provided
        if "text" not in arguments or not arguments["text"]:
            return {
                "error": True,
                "message": "Text is required"
            }
        
        # Prepare input_data
        input_data = {"text": arguments["text"]}
        
        # Prepare kwargs
        kwargs = {}
        if "summary_type" in arguments:
            kwargs["summary_type"] = arguments["summary_type"]
        if "language" in arguments:
            kwargs["language"] = arguments["language"]
        if "max_length" in arguments:
            kwargs["max_length"] = arguments["max_length"]
        
        # Call tool
        result = await self.summary_tool.process(input_data, **kwargs)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2)
                }
            ],
            "isError": False
        }
    
    async def _call_mindmap(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Internal method to call mindmap tool."""
        if not self.mindmap_tool:
            return {
                "error": True,
                "message": "Mindmap tool not initialized"
            }
        
        # Validate text is provided
        if "text" not in arguments or not arguments["text"]:
            return {
                "error": True,
                "message": "Text is required"
            }
        
        # Prepare input_data
        input_data = {"text": arguments["text"]}
        
        # Prepare kwargs
        kwargs = {}
        if "language" in arguments:
            kwargs["language"] = arguments["language"]
        if "max_nodes" in arguments:
            kwargs["max_nodes"] = arguments["max_nodes"]
        if "max_depth" in arguments:
            kwargs["max_depth"] = arguments["max_depth"]
        if "output_format" in arguments:
            kwargs["output_format"] = arguments["output_format"]
        if "temperature" in arguments:
            kwargs["temperature"] = arguments["temperature"]
        
        # Call tool
        result = await self.mindmap_tool.process(input_data, **kwargs)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False, indent=2)
                }
            ],
            "isError": False
        }
    
    # ==================== MCP Resources Interface ====================
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """
        List all available MCP resources.
        
        Returns:
            List of resource definitions
        """
        return [
            {
                "uri": "prompt://summary/abstractive/vietnamese",
                "name": "Summary Abstractive Prompt (Vietnamese)",
                "description": "Prompt template for abstractive summarization in Vietnamese",
                "mimeType": "text/plain"
            },
            {
                "uri": "prompt://summary/extractive/vietnamese",
                "name": "Summary Extractive Prompt (Vietnamese)",
                "description": "Prompt template for extractive summarization in Vietnamese",
                "mimeType": "text/plain"
            },
            {
                "uri": "prompt://summary/abstractive/english",
                "name": "Summary Abstractive Prompt (English)",
                "description": "Prompt template for abstractive summarization in English",
                "mimeType": "text/plain"
            },
            {
                "uri": "prompt://summary/extractive/english",
                "name": "Summary Extractive Prompt (English)",
                "description": "Prompt template for extractive summarization in English",
                "mimeType": "text/plain"
            },
            {
                "uri": "prompt://mindmap/vietnamese",
                "name": "Mindmap Prompt (Vietnamese)",
                "description": "Prompt template for mindmap generation in Vietnamese",
                "mimeType": "text/plain"
            },
            {
                "uri": "prompt://mindmap/english",
                "name": "Mindmap Prompt (English)",
                "description": "Prompt template for mindmap generation in English",
                "mimeType": "text/plain"
            },
            {
                "uri": "prompt://mindmap/markdown/vietnamese",
                "name": "Mindmap Markdown Prompt (Vietnamese)",
                "description": "Prompt template for markdown mindmap generation in Vietnamese",
                "mimeType": "text/plain"
            },
            {
                "uri": "prompt://mindmap/markdown/english",
                "name": "Mindmap Markdown Prompt (English)",
                "description": "Prompt template for markdown mindmap generation in English",
                "mimeType": "text/plain"
            }
        ]
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read an MCP resource by URI.
        
        Args:
            uri: Resource URI to read
            
        Returns:
            Resource content
        """
        try:
            # Parse URI: prompt://type/subtype/language
            if uri.startswith("prompt://"):
                parts = uri.replace("prompt://", "").split("/")
                
                if len(parts) >= 2:
                    prompt_type = parts[0]
                    language = parts[-1]
                    subtype = parts[1] if len(parts) > 2 else None
                    
                    # Handle markdown mindmap special case
                    if prompt_type == "mindmap" and "markdown" in parts:
                        prompt_type = "markdown_mindmap"
                    
                    # Get prompt
                    prompt_text = get_prompt(
                        prompt_type,
                        language,
                        subtype,
                        max_length=2000
                    )
                    
                    return {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/plain",
                                "text": prompt_text
                            }
                        ]
                    }
            
            return {
                "error": True,
                "message": f"Unknown resource URI: {uri}"
            }
        except Exception as e:
            logger.error("Error reading resource %s: %s", uri, e, exc_info=True)
            return {
                "error": True,
                "message": str(e)
            }
    
    # ==================== MCP Prompts Interface ====================
    
    def list_prompts(self) -> List[Dict[str, Any]]:
        """
        List all available MCP prompts.
        
        Returns:
            List of prompt templates
        """
        return [
            {
                "name": "summarize_abstractive",
                "description": "Generate abstractive summary from text",
                "arguments": [
                    {
                        "name": "text",
                        "description": "Text content to summarize",
                        "required": True
                    },
                    {
                        "name": "language",
                        "description": "Output language (vietnamese/english)",
                        "required": False
                    },
                    {
                        "name": "max_length",
                        "description": "Maximum summary length in words",
                        "required": False
                    }
                ]
            },
            {
                "name": "summarize_extractive",
                "description": "Generate extractive summary from text",
                "arguments": [
                    {
                        "name": "text",
                        "description": "Text content to summarize",
                        "required": True
                    },
                    {
                        "name": "language",
                        "description": "Output language (vietnamese/english)",
                        "required": False
                    },
                    {
                        "name": "max_length",
                        "description": "Maximum summary length in words",
                        "required": False
                    }
                ]
            },
            {
                "name": "create_mindmap_json",
                "description": "Create mindmap in JSON format",
                "arguments": [
                    {
                        "name": "text",
                        "description": "Text content to create mindmap from",
                        "required": True
                    },
                    {
                        "name": "language",
                        "description": "Output language (vietnamese/english)",
                        "required": False
                    },
                    {
                        "name": "max_nodes",
                        "description": "Maximum number of nodes",
                        "required": False
                    },
                    {
                        "name": "max_depth",
                        "description": "Maximum depth levels",
                        "required": False
                    }
                ]
            },
            {
                "name": "create_mindmap_markdown",
                "description": "Create mindmap in Markdown format",
                "arguments": [
                    {
                        "name": "text",
                        "description": "Text content to create mindmap from",
                        "required": True
                    },
                    {
                        "name": "language",
                        "description": "Output language (vietnamese/english)",
                        "required": False
                    },
                    {
                        "name": "max_nodes",
                        "description": "Maximum number of nodes",
                        "required": False
                    },
                    {
                        "name": "max_depth",
                        "description": "Maximum depth levels",
                        "required": False
                    }
                ]
            }
        ]
    
    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get a prompt template with arguments filled in.
        
        Args:
            name: Prompt name
            arguments: Optional arguments to fill in
            
        Returns:
            Prompt template with filled arguments
        """
        arguments = arguments or {}
        
        try:
            if name == "summarize_abstractive":
                language = arguments.get("language", "vietnamese")
                max_length = arguments.get("max_length", 2000)
                text = arguments.get("text", "")
                
                system_prompt = get_prompt("summary", language, "abstractive", max_length=max_length)
                user_prompt = f"Tóm tắt văn bản sau:\n\n{text}" if language == "vietnamese" else f"Summarize the following text:\n\n{text}"
                
                return {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                }
            
            elif name == "summarize_extractive":
                language = arguments.get("language", "vietnamese")
                max_length = arguments.get("max_length", 2000)
                text = arguments.get("text", "")
                
                system_prompt = get_prompt("summary", language, "extractive", max_length=max_length)
                user_prompt = f"Trích xuất các ý chính từ văn bản sau:\n\n{text}" if language == "vietnamese" else f"Extract key points from the following text:\n\n{text}"
                
                return {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                }
            
            elif name == "create_mindmap_json":
                language = arguments.get("language", "vietnamese")
                text = arguments.get("text", "")
                max_nodes = arguments.get("max_nodes", 50)
                max_depth = arguments.get("max_depth", 4)
                
                system_prompt = get_prompt("mindmap", language)
                user_prompt = f"Tạo sơ đồ tư duy từ văn bản sau (tối đa {max_nodes} nút, {max_depth} cấp độ):\n\n{text}" if language == "vietnamese" else f"Create mindmap from the following text (max {max_nodes} nodes, {max_depth} levels):\n\n{text}"
                
                return {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                }
            
            elif name == "create_mindmap_markdown":
                language = arguments.get("language", "vietnamese")
                text = arguments.get("text", "")
                max_nodes = arguments.get("max_nodes", 50)
                max_depth = arguments.get("max_depth", 4)
                
                system_prompt = get_prompt("markdown_mindmap", language)
                user_prompt = f"Tạo sơ đồ tư duy dạng markdown từ văn bản sau (tối đa {max_nodes} nút, {max_depth} cấp độ):\n\n{text}" if language == "vietnamese" else f"Create markdown mindmap from the following text (max {max_nodes} nodes, {max_depth} levels):\n\n{text}"
                
                return {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                }
            
            return {
                "error": True,
                "message": f"Unknown prompt: {name}"
            }
        except Exception as e:
            logger.error("Error getting prompt %s: %s", name, e, exc_info=True)
            return {
                "error": True,
                "message": str(e)
            }
    
    # ==================== Server Info ====================
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get MCP server information.
        
        Returns:
            Server metadata
        """
        return {
            "name": "intramind-tools",
            "version": "1.0.0",
            "description": "MCP Server for IntraMind document processing tools",
            "capabilities": {
                "tools": {
                    "listChanged": True
                },
                "resources": {
                    "subscribe": True,
                    "listChanged": True
                },
                "prompts": {
                    "listChanged": True
                }
            },
            "tools": [tool["name"] for tool in self.list_tools()],
            "resources": [resource["uri"] for resource in self.list_resources()],
            "prompts": [prompt["name"] for prompt in self.list_prompts()]
        }


# ==================== Convenience Functions ====================

def create_mcp_server_from_settings(settings) -> MCPServer:
    """
    Create MCP server from application settings.
    
    Args:
        settings: Application settings object
        
    Returns:
        Configured MCP server instance
    """
    # Get LLM configuration
    provider = getattr(settings, 'llm_provider', 'ollama')
    model_name = getattr(settings, f'{provider}_model', 'unknown')
    
    llm_config = {
        "enabled": getattr(settings, 'enable_llm', True),
        "provider": provider,
        "model_name": model_name,
        "api_key": getattr(settings, 'openai_api_key', None),
        "base_url": getattr(settings, 'openai_base_url', None),
        "temperature": getattr(settings, 'llm_temperature', 0.7),
        "max_tokens": getattr(settings, 'llm_max_tokens', 1000)
    }
    
    # Get tool configurations
    tool_configs = {
        "summary": {
            "summary_type": getattr(settings, 'summary_default_type', 'abstractive'),
            "language": getattr(settings, 'tools_language', 'vietnamese'),
            "max_length": getattr(settings, 'summary_max_tokens', 1000)
        },
        "mindmap": {
            "language": getattr(settings, 'tools_language', 'vietnamese'),
            "max_nodes": getattr(settings, 'mindmap_default_nodes', 50),
            "max_depth": getattr(settings, 'mindmap_default_depth', 4)
        }
    }
    
    return MCPServer(llm_config, tool_configs)
