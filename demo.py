"""
Demo script for using MCP Server with OpenAI LLM.
This demonstrates how to use the SummaryTool and MindmapTool via MCP Server.
"""

import asyncio
import os
from dotenv import load_dotenv
from mcp_server import MCPServer

# Load environment variables from .env file
load_dotenv()


async def demo_summarize(mcp_server=None):
    """Demo: Summarize text using SummaryTool."""
    print("=" * 60)
    print("Demo: Text Summarization")
    print("=" * 60)
    
    # Create MCP server if not provided
    if mcp_server is None:
        # Configure LLM
        llm_config = {
            "enabled": True,
            "provider": "openai",
            "model_name": os.getenv("OPENAI_MODEL", ""),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
            "default_temperature": 0.7,
            "default_max_tokens": 2000
        }
        
        # Configure tools
        tool_configs = {
            "summary": {
                "summary_type": os.getenv("SUMMARY_DEFAULT_TYPE", "abstractive"),
                "language": os.getenv("TOOLS_LANGUAGE", "vietnamese"),
                "max_length": int(os.getenv("SUMMARY_MAX_TOKENS", "2000"))
            }
        }
        
        # Create MCP server
        mcp_server = MCPServer(llm_config, tool_configs)
    
    # Sample text to summarize
    sample_text = """
    Artificial Intelligence (AI) has revolutionized many industries in recent years. 
    Machine learning algorithms can now process vast amounts of data and make predictions 
    with remarkable accuracy. Natural language processing enables computers to understand 
    and generate human language. Computer vision allows machines to interpret visual 
    information from the world around them. These technologies are being applied in 
    healthcare, finance, transportation, and many other sectors. As AI continues to 
    advance, it promises to transform how we work, live, and interact with technology.
    
    Deep learning, a subset of machine learning, uses neural networks with multiple 
    layers to learn complex patterns in data. These networks are inspired by the 
    structure of the human brain. Convolutional neural networks excel at image 
    recognition tasks, while recurrent neural networks are well-suited for sequential 
    data like text and speech. Transformer models have become the state-of-the-art 
    for many natural language processing tasks.
    """
    
    print("\nInput text length:", len(sample_text), "characters")
    print("\nGenerating summary...")
    
    # Call summarize tool
    result = await mcp_server.call_tool("summarize_text", {
        "text": sample_text,
        "summary_type": "abstractive",
        "language": "english",
        "max_length": 200
    })
    
    if result.get("isError") or result.get("error"):
        print(f"Error: {result.get('message', 'Unknown error')}")
        return
    
    # Check if result has content
    if "content" not in result or not result["content"]:
        print("Error: No content in result")
        print("Result:", result)
        return
    
    import json
    try:
        result_data = json.loads(result["content"][0]["text"])
        print("\n" + "-" * 60)
        print("Summary:")
        print("-" * 60)
        print(result_data.get("summary", ""))
        print("\n" + "-" * 60)
        print("Metadata:")
        print("-" * 60)
        print(json.dumps(result_data.get("metadata", {}), indent=2))
        print()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing result: {e}")
        print("Raw result:", result)


async def demo_mindmap(mcp_server=None):
    """Demo: Create mindmap using MindmapTool."""
    print("=" * 60)
    print("Demo: Mindmap Generation")
    print("=" * 60)
    
    # Create MCP server if not provided
    if mcp_server is None:
        # Configure LLM
        llm_config = {
            "enabled": True,
            "provider": "openai",
            "model_name": os.getenv("OPENAI_MODEL", ""),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
            "default_temperature": 0.5,
            "default_max_tokens": 2000
        }
        
        # Configure tools
        tool_configs = {
            "mindmap": {
                "language": os.getenv("TOOLS_LANGUAGE", "vietnamese"),
                "max_nodes": int(os.getenv("MINDMAP_DEFAULT_NODES", "50")),
                "max_depth": int(os.getenv("MINDMAP_DEFAULT_DEPTH", "4"))
            }
        }
        
        # Create MCP server
        mcp_server = MCPServer(llm_config, tool_configs)
    
    # Sample text for mindmap
    sample_text = """
    Python is a high-level programming language known for its simplicity and readability.
    It supports multiple programming paradigms including object-oriented, imperative, 
    and functional programming. Python has a large standard library and an active 
    community that contributes to thousands of third-party packages.
    
    Key features of Python include dynamic typing, automatic memory management, 
    and support for multiple platforms. It's widely used in web development, 
    data science, machine learning, automation, and scientific computing.
    
    Popular Python frameworks include Django and Flask for web development, 
    NumPy and Pandas for data analysis, and TensorFlow and PyTorch for machine learning.
    """
    
    print("\nInput text length:", len(sample_text), "characters")
    print("\nGenerating mindmap...")
    
    # Call mindmap tool
    result = await mcp_server.call_tool("create_mindmap", {
        "text": sample_text,
        "language": "english",
        "max_nodes": 30,
        "max_depth": 3,
        "output_format": "json"
    })
    
    if result.get("isError") or result.get("error"):
        print(f"Error: {result.get('message', 'Unknown error')}")
        return
    
    # Check if result has content
    if "content" not in result or not result["content"]:
        print("Error: No content in result")
        print("Result:", result)
        return
    
    import json
    try:
        result_data = json.loads(result["content"][0]["text"])
        print("\n" + "-" * 60)
        print("Mindmap (JSON):")
        print("-" * 60)
        print(json.dumps(result_data.get("mindmap", {}), indent=2))
        print("\n" + "-" * 60)
        print("Metadata:")
        print("-" * 60)
        print(json.dumps(result_data.get("metadata", {}), indent=2))
        print()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing result: {e}")
        print("Raw result:", result)


async def demo_mindmap_markdown(mcp_server=None):
    """Demo: Create mindmap in Markdown format."""
    print("=" * 60)
    print("Demo: Mindmap Generation (Markdown Format)")
    print("=" * 60)
    
    # Create MCP server if not provided
    if mcp_server is None:
        # Configure LLM
        llm_config = {
            "enabled": True,
            "provider": "openai",
            "model_name": os.getenv("OPENAI_MODEL", ""),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
            "default_temperature": 0.5,
            "default_max_tokens": 2000
        }
        
        # Configure tools
        tool_configs = {
            "mindmap": {
                "language": "vietnamese",
                "max_nodes": 30,
                "max_depth": 3
            }
        }
        
        # Create MCP server
        mcp_server = MCPServer(llm_config, tool_configs)
    
    # Sample text in Vietnamese
    sample_text = """
    Trí tuệ nhân tạo (AI) đang thay đổi cách chúng ta làm việc và sống. 
    Machine Learning cho phép máy tính học từ dữ liệu mà không cần lập trình 
    cụ thể. Deep Learning sử dụng mạng neural nhiều lớp để xử lý thông tin phức tạp.
    
    Các ứng dụng của AI bao gồm: nhận dạng hình ảnh, xử lý ngôn ngữ tự nhiên, 
    xe tự lái, và chatbot thông minh. AI đang được sử dụng trong y tế để chẩn đoán 
    bệnh, trong tài chính để phát hiện gian lận, và trong giáo dục để cá nhân hóa 
    việc học tập.
    """
    
    print("\nInput text length:", len(sample_text), "characters")
    print("\nGenerating markdown mindmap...")
    
    # Call mindmap tool with markdown format
    result = await mcp_server.call_tool("create_mindmap", {
        "text": sample_text,
        "language": "vietnamese",
        "max_nodes": 25,
        "max_depth": 3,
        "output_format": "markdown"
    })
    
    if result.get("isError") or result.get("error"):
        print(f"Error: {result.get('message', 'Unknown error')}")
        return
    
    # Check if result has content
    if "content" not in result or not result["content"]:
        print("Error: No content in result")
        print("Result:", result)
        return
    
    import json
    try:
        result_data = json.loads(result["content"][0]["text"])
        print("\n" + "-" * 60)
        print("Mindmap (Markdown):")
        print("-" * 60)
        print(result_data.get("formats", {}).get("markdown", ""))
        print("\n" + "-" * 60)
        print("Metadata:")
        print("-" * 60)
        print(json.dumps(result_data.get("metadata", {}), indent=2))
        print()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing result: {e}")
        print("Raw result:", result)


async def cleanup_async_clients(mcp_server):
    """Cleanup async clients to avoid Windows event loop warnings."""
    try:
        # Close async clients if they exist
        if mcp_server.summary_tool and hasattr(mcp_server.summary_tool, 'llm'):
            llm = mcp_server.summary_tool.llm
            if hasattr(llm, 'async_client') and llm.async_client:
                await llm.async_client.close()
        
        if mcp_server.mindmap_tool and hasattr(mcp_server.mindmap_tool, 'llm'):
            llm = mcp_server.mindmap_tool.llm
            if hasattr(llm, 'async_client') and llm.async_client:
                await llm.async_client.close()
    except Exception:
        # Ignore cleanup errors
        pass


async def main():
    """Main demo function."""
    print("\n" + "=" * 60)
    print("MCP Server Demo with OpenAI LLM")
    print("=" * 60)
    print("\nMake sure you have:")
    print("1. Created a .env file with your API credentials")
    print("2. Installed required packages: pip install python-dotenv")
    print()
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("WARNING: .env file not found!")
        print("Please create a .env file with your API credentials.")
        print("You can copy .env.example to .env and update the values.")
        print()
        return
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env file!")
        return
    
    print(f"✓ Using model: {os.getenv('OPENAI_MODEL', '')}")
    print(f"✓ Using base URL: {os.getenv('OPENAI_BASE_URL', '')}")
    print()
    
    # Create shared MCP server with both tools for all demos
    llm_config = {
        "enabled": True,
        "provider": "openai",
        "model_name": os.getenv("OPENAI_MODEL", ""),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL", ""),
        "default_temperature": 0.7,
        "default_max_tokens": 2000
    }
    
    tool_configs = {
        "summary": {
            "summary_type": os.getenv("SUMMARY_DEFAULT_TYPE", "abstractive"),
            "language": os.getenv("TOOLS_LANGUAGE", "vietnamese"),
            "max_length": int(os.getenv("SUMMARY_MAX_TOKENS", "2000"))
        },
        "mindmap": {
            "language": os.getenv("TOOLS_LANGUAGE", "vietnamese"),
            "max_nodes": int(os.getenv("MINDMAP_DEFAULT_NODES", "50")),
            "max_depth": int(os.getenv("MINDMAP_DEFAULT_DEPTH", "4"))
        }
    }
    
    shared_mcp_server = MCPServer(llm_config, tool_configs)
    
    try:
        # Run demos with shared server
        await demo_summarize(shared_mcp_server)
        await asyncio.sleep(0.5)
        
        await demo_mindmap(shared_mcp_server)
        await asyncio.sleep(0.5)
        
        await demo_mindmap_markdown(shared_mcp_server)
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback as tb
        tb.print_exc()
    
    finally:
        # Cleanup async clients to avoid Windows event loop warnings
        if shared_mcp_server:
            await cleanup_async_clients(shared_mcp_server)
        # Small delay to allow cleanup to complete
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    # Fix Windows event loop policy to avoid cleanup warnings
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    import traceback
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
