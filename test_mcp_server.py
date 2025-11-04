"""
Test script for MCP Server.
Tests the MCP server by sending JSON-RPC messages via stdio.
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_mcp_server():
    """Test MCP server by sending JSON-RPC messages."""
    
    # Path to mcp_server_main.py
    server_script = Path(__file__).parent / "mcp_server_main.py"
    
    # Check if config file exists
    config_file = Path(__file__).parent / "config.json"
    if not config_file.exists():
        print("WARNING: config.json not found. Creating default config...")
        create_default_config(config_file)
    
    # Start server process
    print("Starting MCP server...")
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(server_script),
        '--config', str(config_file),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    # Start reading stderr in background for debugging
    async def read_stderr():
        """Read stderr in background to see server logs."""
        while True:
            line = await process.stderr.readline()
            if not line:
                break
            print(f"   [SERVER LOG] {line.decode().strip()}")
    
    # Start stderr reader for debugging
    stderr_task = asyncio.create_task(read_stderr())
    
    try:
        # 1. Initialize
        print("\nStep 1: Initialize MCP server...")
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
        
        # MCP server sends a notification first, then waits for initialize request
        # So we need to read the notification first
        print("   Waiting for server notification...")
        notification = await read_response(process, timeout=5.0)
        if notification:
            if notification.get("method") == "notifications/initialized":
                print("   Received initialized notification (from server)")
            else:
                print(f"   WARNING: Unexpected message: {notification.get('method', 'unknown')}")
        
        # Now send initialize request
        await send_request(process, init_request)
        # Read response (should have id matching our request)
        response = await read_response(process, timeout=5.0)
        
        # Filter out notifications, only get responses with matching id
        while response and response.get("id") != init_request["id"]:
            if response.get("method") == "notifications/initialized":
                print("   Skipping duplicate notification")
            response = await read_response(process, timeout=5.0)
        
        if response and "result" in response:
            print("Initialize successful!")
            print(f"   Server info: {response['result'].get('serverInfo', {})}")
        else:
            print("Initialize failed!")
            print(f"   Response: {response}")
            return
        
        # Send initialized notification (required by MCP protocol)
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await send_request(process, initialized_notification)
        
        # 2. List tools
        print("\nStep 2: List available tools...")
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        
        await send_request(process, list_tools_request)
        response = await read_response(process, timeout=5.0)
        # Ensure we got the right response (matching id)
        while response and response.get("id") != list_tools_request["id"]:
            response = await read_response(process, timeout=5.0)
        
        if response and "result" in response:
            tools = response["result"].get("tools", [])
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"   - {tool.get('name')}: {tool.get('description', '')}")
        else:
            print("Failed to list tools!")
            print(f"   Response: {response}")
            return
        
        # 3. Test summarize_text tool
        print("\nStep 3: Test summarize_text tool...")
        summarize_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "summarize_text",
                "arguments": {
                    "text": """Artificial Intelligence (AI) has revolutionized many industries in recent years. 
                    Machine learning algorithms can now process vast amounts of data and make predictions with remarkable accuracy. 
                    Natural language processing enables computers to understand and generate human language. 
                    Computer vision allows machines to interpret visual information from the world around them. 
                    These technologies are being applied in healthcare, finance, transportation, and many other sectors. 
                    As AI continues to advance, it promises to transform how we work, live, and interact with technology."""
                }
            }
        }
        
        await send_request(process, summarize_request)
        print("   Waiting for response (this may take a while, up to 120s)...")
        
        # Read response with longer timeout for summarize operation
        response = None
        max_attempts = 12  # 12 attempts x 10s = 120s max
        attempt = 0
        
        while attempt < max_attempts:
            try:
                response = await read_response(process, timeout=10.0)
                if response and response.get("id") == summarize_request["id"]:
                    break
                elif response and "id" in response:
                    print(f"   WARNING: Received response with id={response.get('id')}, waiting for id={summarize_request['id']}...")
                elif response and response.get("method") == "notifications/initialized":
                    print("   Skipping notification, waiting for response...")
                else:
                    print(f"   WARNING: Unexpected response: {response.get('method', 'unknown') if response else 'None'}")
            except asyncio.TimeoutError:
                attempt += 1
                if attempt < max_attempts:
                    print(f"   Still waiting... ({attempt}/{max_attempts})")
                else:
                    print("   Timeout waiting for response!")
                    break
        
        if response and "result" in response:
            result = response["result"]
            if "content" in result:
                content = result["content"][0]["text"]
                try:
                    summary_data = json.loads(content)
                    print("Summarize successful!")
                    print(f"   Summary: {summary_data.get('summary', '')[:200]}...")
                    print(f"   Metadata: {summary_data.get('metadata', {})}")
                except json.JSONDecodeError:
                    print("Summarize successful!")
                    print(f"   Content: {content[:200]}...")
            else:
                print("WARNING: Response received but no content:")
                print(f"   {json.dumps(result, indent=2)}")
        else:
            print("Summarize failed!")
            if "error" in response:
                print(f"   Error: {response['error']}")
            else:
                print(f"   Response: {response}")
        
        # 4. Test create_mindmap tool
        print("\nStep 4: Test create_mindmap tool...")
        mindmap_request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "create_mindmap",
                "arguments": {
                    "text": """Python is a high-level programming language known for its simplicity and readability.
                    It supports multiple programming paradigms including object-oriented, imperative, functional, and procedural styles.
                    Python has a large standard library and a vibrant ecosystem of third-party packages.
                    It is widely used in web development, data science, artificial intelligence, and automation."""
                }
            }
        }
        
        await send_request(process, mindmap_request)
        print("   Waiting for response (this may take a while, up to 120s)...")
        
        # Read response with longer timeout for mindmap operation
        response = None
        max_attempts = 12  # 12 attempts x 10s = 120s max
        attempt = 0
        
        while attempt < max_attempts:
            try:
                response = await read_response(process, timeout=10.0)
                if response and response.get("id") == mindmap_request["id"]:
                    break
                elif response and "id" in response:
                    print(f"   WARNING: Received response with id={response.get('id')}, waiting for id={mindmap_request['id']}...")
                elif response and response.get("method") == "notifications/initialized":
                    print("   Skipping notification, waiting for response...")
                else:
                    print(f"   WARNING: Unexpected response: {response.get('method', 'unknown') if response else 'None'}")
            except asyncio.TimeoutError:
                attempt += 1
                if attempt < max_attempts:
                    print(f"   Still waiting... ({attempt}/{max_attempts})")
                else:
                    print("   Timeout waiting for response!")
                    break
        
        if response and "result" in response:
            result = response["result"]
            if "content" in result:
                content = result["content"][0]["text"]
                try:
                    mindmap_data = json.loads(content)
                    print("Mindmap creation successful!")
                    print(f"   Format: {mindmap_data.get('metadata', {}).get('output_format', 'unknown')}")
                    print(f"   Nodes: {mindmap_data.get('metadata', {}).get('node_count', 'unknown')}")
                    print(f"   Depth: {mindmap_data.get('metadata', {}).get('max_depth', 'unknown')}")
                except json.JSONDecodeError:
                    print("Mindmap creation successful!")
                    print(f"   Content preview: {content[:200]}...")
            else:
                print("WARNING: Response received but no content:")
                print(f"   {json.dumps(result, indent=2)}")
        else:
            print("Mindmap creation failed!")
            if "error" in response:
                print(f"   Error: {response['error']}")
            else:
                print(f"   Response: {response}")
        
        # 5. List resources
        print("\nStep 5: List available resources...")
        list_resources_request = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "resources/list"
        }
        
        await send_request(process, list_resources_request)
        response = await read_response(process, timeout=5.0)
        # Ensure we got the right response (matching id)
        while response and response.get("id") != list_resources_request["id"]:
            response = await read_response(process, timeout=5.0)
        
        if response and "result" in response:
            resources = response["result"].get("resources", [])
            print(f"Found {len(resources)} resources:")
            for resource in resources[:5]:  # Show first 5
                print(f"   - {resource.get('uri', '')}")
        else:
            print("Failed to list resources!")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except asyncio.TimeoutError:
        print("\nTimeout waiting for response!")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\nUnexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cancel stderr reader
        try:
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
        except (NameError, Exception):
            pass  # Ignore errors when canceling
        
        print("\nStopping MCP server...")
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
        print("Server stopped")


async def send_request(process, request):
    """Send JSON-RPC request to server."""
    request_str = json.dumps(request, ensure_ascii=False) + "\n"
    process.stdin.write(request_str.encode())
    await process.stdin.drain()


async def read_response(process, timeout=10.0, ignore_notifications=False):
    """
    Read JSON-RPC response from server.
    
    Args:
        process: Server subprocess
        timeout: Timeout in seconds
        ignore_notifications: If True, skip notifications (messages without 'id')
    
    Returns:
        Response dict or None
    """
    try:
        while True:
            response_line = await asyncio.wait_for(
                process.stdout.readline(),
                timeout=timeout
            )
            if not response_line:
                return None
            
            response = json.loads(response_line.decode().strip())
            
            # Skip notifications if requested
            if ignore_notifications and "id" not in response:
                continue
            
            return response
            
    except json.JSONDecodeError as e:
        print(f"WARNING: JSON decode error: {e}")
        if 'response_line' in locals():
            print(f"   Raw response: {response_line.decode()[:200]}")
        return None


def create_default_config(config_file: Path):
    """Create default config.json file."""
    default_config = {
        "llm_config": {
            "enabled": True,
            "provider": "openai",
            "model_name": os.getenv("OPENAI_MODEL", ""),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
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
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    print(f"Created default config at {config_file}")
    print("WARNING: Please update OPENAI_API_KEY in .env file or config.json")


if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("WARNING: python-dotenv not installed. Skipping .env loading.")
    
    print("=" * 60)
    print("MCP Server Test Script")
    print("=" * 60)
    print()
    
    asyncio.run(test_mcp_server())

