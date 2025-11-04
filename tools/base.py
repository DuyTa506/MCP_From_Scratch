"""
Base class for all tools in the system.
Provides common interface and LLM integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """
    Abstract base class for all tools in the system.
    
    All tools must inherit from this class and implement:
    - process(): Main processing method
    - validate_input(): Input validation method
    """
    
    def __init__(self, llm, config: Dict[str, Any]):
        """
        Initialize tool with LLM instance and configuration.
        
        Args:
            llm: LLM instance for processing
            config: Tool-specific configuration
        """
        self.llm = llm
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize tool-specific settings
        self._initialize_tool_settings()
    
    @abstractmethod
    async def process(
        self, 
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process input data and return structured result.
        
        Args:
            input_data: Input data containing text or document_id
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary with processed result and metadata
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before processing.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get tool information and capabilities.
        
        Returns:
            Dictionary with tool metadata
        """
        # Filter config to only include serializable values
        serializable_config = {}
        if self.config:
            for key, value in self.config.items():
                # Skip non-serializable objects
                if not hasattr(value, '__dict__') and not callable(value):
                    try:
                        # Test if value is JSON serializable
                        import json
                        json.dumps(value)
                        serializable_config[key] = value
                    except (TypeError, ValueError):
                        # Skip non-serializable values
                        pass
        
        return {
            "name": self.__class__.__name__,
            "version": "1.0.0",
            "description": self.__doc__ or "",
            "config": serializable_config,
            "capabilities": self._get_capabilities(),
            "supported_languages": self._get_supported_languages(),
            "created_at": datetime.now().isoformat()
        }
    
    def _initialize_tool_settings(self):
        """
        Initialize tool-specific settings from config.
        Override in subclasses for custom initialization.
        """
        pass
    
    def _get_capabilities(self) -> List[str]:
        """
        Get list of tool capabilities.
        Override in subclasses to specify capabilities.
        """
        return []
    
    def _get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        Override in subclasses to specify languages.
        """
        return ["vietnamese", "english"]
    
    async def _get_document_text(self, document_id: str) -> str:
        """
        Retrieve document text from storage.
        
        Args:
            document_id: ID of document to retrieve
            
        Returns:
            Document text content
        """
        try:
            # This will be implemented based on existing storage systems
            # For now, return placeholder
            self.logger.warning(f"Document retrieval not implemented for ID: {document_id}")
            return f"Document content for {document_id}"
        except Exception as e:
            self.logger.error(f"Error retrieving document {document_id}: {e}")
            raise
    
    def _truncate_text(self, text: str, max_length: int = 8000) -> str:
        """
        Truncate text to maximum length with ellipsis.
        
        Args:
            text: Text to truncate
            max_length: Maximum allowed length
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from LLM response.
        
        Args:
            response: LLM response containing JSON
            
        Returns:
            Parsed JSON dictionary
        """
        import json
        import re
        
        try:
            # Try to parse entire response as JSON
            return json.loads(response.strip())
        except json.JSONDecodeError:
            try:
                # Try to extract JSON using regex
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
            
            # Fallback: return basic structure
            return {
                "error": "Could not parse JSON response",
                "raw_response": response[:500]  # First 500 chars
            }