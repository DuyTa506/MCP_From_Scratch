"""
Factory for creating and managing tool instances.
Provides centralized tool management and configuration.
"""

from typing import Dict, Any, Optional, Type, List
import logging

from .base import BaseTool
from .summary import SummaryTool
from .mindmap import MindmapTool

logger = logging.getLogger(__name__)


class ToolFactory:
    """
    Factory class for creating and managing tool instances.
    
    Supports:
    - Dynamic tool creation
    - Configuration-based instantiation
    - Tool registration
    - Provider validation
    """
    
    # Registry of available tools
    _tools = {
        "summary": SummaryTool,
        "mindmap": MindmapTool,
    }
    
    @classmethod
    def create_tool(
        cls,
        tool_name: str,
        llm_config: Dict[str, Any],
        tool_config: Optional[Dict[str, Any]] = None
    ) -> BaseTool:
        """
        Create tool instance with LLM integration.
        
        Args:
            tool_name: Name of tool to create
            llm_config: LLM configuration dictionary
            tool_config: Tool-specific configuration
            
        Returns:
            Configured tool instance
            
        Raises:
            ValueError: If tool is not supported
            ImportError: If required dependencies are missing
        """
        tool_name = tool_name.lower()
        
        if tool_name not in cls._tools:
            available = ", ".join(cls._tools.keys())
            raise ValueError(
                f"Unsupported tool: '{tool_name}'. "
                f"Available tools: {available}"
            )
        
        try:
            # Create LLM instance using existing factory
            from llms.factory import LLMFactory
            llm = LLMFactory.create_from_config(llm_config)
            
            if llm is None:
                raise ValueError(f"Failed to create LLM from config: {llm_config}")
            
            # Merge tool config with defaults
            final_config = tool_config or {}
            
            # Create tool instance
            tool_class = cls._tools[tool_name]
            tool_instance = tool_class(llm=llm, config=final_config)
            
            logger.info(f"✓ Created {tool_name} tool with LLM provider")
            return tool_instance
            
        except ImportError as e:
            logger.error(f"Missing dependencies for {tool_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to create {tool_name} tool: {e}")
            raise
    
    @classmethod
    def create_from_settings(
        cls,
        tool_name: str,
        settings
    ) -> Optional[BaseTool]:
        """
        Create tool from application settings.
        
        Args:
            tool_name: Name of tool to create
            settings: Application settings object
            
        Returns:
            Configured tool instance or None if disabled
        """
        # Check if tools are enabled
        if not getattr(settings, 'enable_tools', True):
            logger.info("Tools disabled in settings")
            return None
        
        # Get tool-specific provider and model if available
        tool_provider = getattr(settings, f'{tool_name}_provider', None)
        tool_model = getattr(settings, f'{tool_name}_model', None)
        
        # Use tool-specific settings if available, otherwise inherit from main LLM settings
        provider = tool_provider or getattr(settings, 'llm_provider', 'ollama')
        model_name = tool_model or getattr(settings, f'{provider}_model', 'unknown')
        
        # Get LLM configuration from settings
        llm_config = {
            "enabled": getattr(settings, 'enable_llm', True),
            "provider": provider,
            "model_name": model_name,
            "api_key": getattr(settings, 'openai_api_key', None),
            "base_url": getattr(settings, 'openai_base_url', None),
            "temperature": getattr(settings, 'llm_temperature', 0.7),
            "max_tokens": getattr(settings, 'llm_max_tokens', 1000)
        }
        
        # Get tool-specific configuration
        tool_config = {}
        if tool_name == "summary":
            tool_config = {
                "summary_type": getattr(settings, 'summary_default_type', 'abstractive'),
                "language": getattr(settings, 'tools_language', 'vietnamese'),
                "max_length": getattr(settings, 'summary_max_tokens', 1000)
            }
        elif tool_name == "mindmap":
            tool_config = {
                "language": getattr(settings, 'tools_language', 'vietnamese'),
                "max_nodes": getattr(settings, 'mindmap_default_nodes', 50),
                "max_depth": getattr(settings, 'mindmap_default_depth', 4)
            }
        
        try:
            return cls.create_tool(tool_name, llm_config, tool_config)
        except Exception as e:
            logger.warning(f"Failed to create {tool_name} from settings: {e}")
            return None
    
    @classmethod
    def get_available_tools(cls) -> List[str]:
        """
        Get list of available tool names.
        
        Returns:
            List of tool names
        """
        return list(cls._tools.keys())
    
    @classmethod
    def register_tool(
        cls,
        name: str,
        tool_class: Type[BaseTool]
    ) -> None:
        """
        Register a new tool.
        
        Args:
            name: Tool name
            tool_class: Tool class that inherits from BaseTool
        """
        if not issubclass(tool_class, BaseTool):
            raise ValueError("Tool class must inherit from BaseTool")
        
        cls._tools[name] = tool_class
        logger.info(f"Registered new tool: {name}")
    
    @classmethod
    def validate_tool(cls, tool_name: str) -> bool:
        """
        Check if tool is available and dependencies are installed.
        
        Args:
            tool_name: Tool name to validate
            
        Returns:
            True if tool is available and ready
        """
        if tool_name not in cls._tools:
            return False
        
        try:
            # Try to import the tool class
            tool_class = cls._tools[tool_name]
            
            # Basic validation - check if class can be instantiated
            # (without actually creating instance)
            if hasattr(tool_class, 'process') and hasattr(tool_class, 'validate_input'):
                return True
                
            return False
            
        except Exception:
            return False
    
    @classmethod
    def get_tool_info(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Tool name to get info for
            
        Returns:
            Tool information dictionary or None if not found
        """
        if tool_name not in cls._tools:
            return None
        
        try:
            # Create temporary instance to get info
            temp_llm = type('TempLLM', (), {
                'get_model_info': lambda: {"provider": "temp", "model_name": "temp"}
            })()
            
            # Create minimal config to avoid serialization issues
            tool_config = {}
            if tool_name == "summary":
                tool_config = {
                    "summary_type": "abstractive",
                    "language": "vietnamese",
                    "max_length": 1000
                }
            elif tool_name == "mindmap":
                tool_config = {
                    "language": "vietnamese",
                    "max_nodes": 50,
                    "max_depth": 4
                }
            
            tool_instance = cls._tools[tool_name](llm=temp_llm, config=tool_config)
            return tool_instance.get_tool_info()
            
        except Exception as e:
            logger.error(f"Error getting tool info for {tool_name}: {e}")
            return {
                "name": tool_name,
                "error": str(e),
                "available": False
            }
    
    @classmethod
    def get_all_tools_info(cls) -> List[Dict[str, Any]]:
        """
        Get information about all available tools.
        
        Returns:
            List of tool information dictionaries
        """
        tools_info = []
        
        for tool_name in cls._tools.keys():
            info = cls.get_tool_info(tool_name)
            if info:
                tools_info.append(info)
        
        return tools_info


# Convenience functions for common use cases
def create_summary_tool(settings) -> Optional[SummaryTool]:
    """
    Create summary tool from settings.
    
    Args:
        settings: Application settings object
        
    Returns:
        SummaryTool instance or None
    """
    return ToolFactory.create_from_settings("summary", settings)


def create_mindmap_tool(settings) -> Optional[MindmapTool]:
    """
    Create mindmap tool from settings.
    
    Args:
        settings: Application settings object
        
    Returns:
        MindmapTool instance or None
    """
    return ToolFactory.create_from_settings("mindmap", settings)


def create_all_tools(settings) -> Dict[str, BaseTool]:
    """
    Create all available tools from settings.
    
    Args:
        settings: Application settings object
        
    Returns:
        Dictionary of tool instances
    """
    tools = {}
    
    for tool_name in ToolFactory.get_available_tools():
        tool = ToolFactory.create_from_settings(tool_name, settings)
        if tool:
            tools[tool_name] = tool
    
    return tools