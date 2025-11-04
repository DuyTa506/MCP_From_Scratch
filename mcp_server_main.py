"""
Standalone MCP Server runner using stdio transport.
This file can be executed directly to run the MCP server.

Usage:
    python mcp_server_main.py

Or with custom config:
    python mcp_server_main.py --config config.json
"""

import asyncio
import json
import sys
import logging
from typing import Dict, Any

from mcp_server import MCPServer, create_mcp_server_from_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StdioMCPServer:
    """
    MCP Server implementation using stdio transport.
    Handles JSON-RPC messages over stdin/stdout.
    """
    
    def __init__(self, mcp_server: MCPServer):
        """
        Initialize stdio MCP server.
        
        Args:
            mcp_server: MCPServer instance
        """
        self.mcp_server = mcp_server
        self.request_id = 0
    
    async def read_request(self) -> Dict[str, Any]:
        """Read JSON-RPC request from stdin."""
        # Read bytes from stdin to handle UTF-8 properly on Windows
        line_bytes = await asyncio.get_event_loop().run_in_executor(
            None, sys.stdin.buffer.readline
        )
        if not line_bytes:
            return None
        # Decode from UTF-8
        line = line_bytes.decode('utf-8').strip()
        return json.loads(line)
    
    def write_response(self, response: Dict[str, Any]):
        """Write JSON-RPC response to stdout."""
        # Encode to UTF-8 bytes to avoid Windows encoding issues
        response_str = json.dumps(response, ensure_ascii=False)
        response_bytes = response_str.encode('utf-8')
        sys.stdout.buffer.write(response_bytes)
        sys.stdout.buffer.write(b'\n')
        sys.stdout.buffer.flush()
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming JSON-RPC request."""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {},
                            "resources": {},
                            "prompts": {}
                        },
                        "serverInfo": self.mcp_server.get_server_info()
                    }
                }
            
            elif method == "tools/list":
                tools = self.mcp_server.list_tools()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": tools
                    }
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
            
            elif method == "resources/list":
                resources = self.mcp_server.list_resources()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "resources": resources
                    }
                }
            
            elif method == "resources/read":
                uri = params.get("uri")
                result = await self.mcp_server.read_resource(uri)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            
            elif method == "prompts/list":
                prompts = self.mcp_server.list_prompts()
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "prompts": prompts
                    }
                }
            
            elif method == "prompts/get":
                prompt_name = params.get("name")
                arguments = params.get("arguments", {})
                result = await self.mcp_server.get_prompt(prompt_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result
                }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
        
        except Exception as e:
            logger.error("Error handling request %s: %s", method, e, exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
    
    async def run(self):
        """Run the stdio MCP server."""
        logger.info("Starting MCP Server (stdio transport)...")
        
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
            
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON: %s", e)
                continue
            except Exception as e:
                logger.error("Unexpected error: %s", e, exc_info=True)
                continue


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        "llm_config": {
            "enabled": True,
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


async def main():
    """Main entry point for MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="IntraMind Tools MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--settings-module",
        type=str,
        help="Python module path to settings (e.g., 'app.settings')"
    )
    
    args = parser.parse_args()
    
    # Initialize MCP server
    if args.settings_module:
        # Load from settings module
        try:
            import importlib
            settings_module = importlib.import_module(args.settings_module)
            mcp_server = create_mcp_server_from_settings(settings_module)
        except Exception as e:
            logger.error("Failed to load settings from module: %s", e)
            sys.exit(1)
    elif args.config:
        # Load from config file
        try:
            config = load_config_from_file(args.config)
            mcp_server = MCPServer(
                config.get("llm_config", {}),
                config.get("tool_configs", {})
            )
        except Exception as e:
            logger.error("Failed to load config from file: %s", e)
            sys.exit(1)
    else:
        # Use default config
        config = create_default_config()
        mcp_server = MCPServer(
            config["llm_config"],
            config["tool_configs"]
        )
    
    # Create and run stdio server
    stdio_server = StdioMCPServer(mcp_server)
    await stdio_server.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
