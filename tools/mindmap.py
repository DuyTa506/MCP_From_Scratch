"""
Production-Ready Mindmap Tool – Refactored Version.
Optimized for creating mindmaps with logical structure chunking, preserving hierarchy and relationships.
Refactored into smaller, more maintainable modules.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime


from .base import BaseTool
from .mindmap_utils import MindmapChunker, MindmapFormatter, MindmapProcessor

logger = logging.getLogger(__name__)


class MindmapTool(BaseTool):
    """
    Refactored Mindmap generation tool using modular components.
    - Split text by logical structure (headings, sections, lists)
    - Preserve hierarchical metadata (level, parent, title)
    - Process each level separately to preserve relationships
    - Support Vietnamese and English languages
    - Output multiple formats (JSON, Mermaid, Markdown)
    """

    def __init__(self, llm, config: Dict[str, Any]):
        super().__init__(llm, config)
        self.default_language = self.config.get("language", "vietnamese")
        self.max_nodes = self.config.get("max_nodes", 50)
        self.max_depth = self.config.get("max_depth", 4)
        self.default_temperature = self.config.get("temperature", 0.5)
        self.context_window = self.config.get("context_window", 16000)
        self.default_format = self.config.get("default_format", "json")  # json or markdown
        
        # Initialize modular components
        self.chunker = MindmapChunker(config)
        self.formatter = MindmapFormatter()
        self.processor = MindmapProcessor(llm, config)
        
        # Get model info from LLM instance first
        try:
            llm_info = self.llm.get_model_info()
            self.model_name = llm_info.get("model_name", "gpt-4")
            self.provider = llm_info.get("provider", "openai")
        except Exception:
            self.model_name = "gpt-4"
            self.provider = "openai"
        
        # Initialize tokenizer
        self._init_tokenizer()
        self.chunker.set_tokenizer(self.tokenizer, self.tokenizer_type)

    def _init_tokenizer(self):
        """Initialize tokenizer based on model type."""
        # Qwen models - use transformers
        if "qwen" in self.model_name.lower():
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen2.5-14B-Instruct",
                    trust_remote_code=True
                )
                self.tokenizer_type = "transformers"
                logger.info(f"✓ Loaded Qwen tokenizer from transformers")
                return
            except Exception as e:
                logger.warning(f"Failed to load transformers tokenizer: {e}")
        
        # Try tiktoken for OpenAI models
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            self.tokenizer_type = "tiktoken"
            logger.info(f"✓ Loaded tiktoken")
        except Exception:
            # Fallback to o200k_base
            try:
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("o200k_base")
                self.tokenizer_type = "tiktoken"
                logger.info(f"✓ Loaded tiktoken o200k_base fallback")
            except Exception as e:
                logger.error(f"✗ Tokenizer loading failed: {e}")
                self.tokenizer = None
                self.tokenizer_type = None

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Main processing method using refactored components."""
        if not self.validate_input(input_data):
            raise ValueError("Input must have 'text'")
        
        start_time = datetime.now()
        self.processor.reset_stats()
        
        text = input_data.get("text", "")
        language = kwargs.get("language", self.default_language)
        max_nodes = kwargs.get("max_nodes", self.max_nodes)
        max_depth = kwargs.get("max_depth", self.max_depth)
        temperature = kwargs.get("temperature", self.default_temperature)
        output_format = kwargs.get("output_format", self.default_format)
        
        if not text or len(text.strip()) < 100:
            raise ValueError("Text too short to create mindmap (>100 characters)")
        
        # Step 1: Chunking using improved method
        chunks = self.chunker.split_text_into_chunks(text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Step 2: Direct processing if single chunk
        if len(chunks) == 1:
            mindmap_data = await self.processor.process_single_chunk(
                chunks[0], language, max_nodes, max_depth, temperature, output_format
            )
        else:
            # Step 3: Map-Reduce processing for multiple chunks
            mindmap_data = await self.processor.process_chunks(
                chunks, language, max_nodes, max_depth, temperature, output_format
            )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        # Step 4: Create multiple output formats
        if output_format == "markdown":
            # For markdown format, mindmap_data is already a markdown string
            formats = {
                "markdown": mindmap_data,
                "json": self.formatter.markdown_to_json(mindmap_data) if isinstance(mindmap_data, str) else mindmap_data,
                "mermaid": self.formatter.markdown_to_mermaid(mindmap_data) if isinstance(mindmap_data, str) else self.formatter.json_to_mermaid(mindmap_data),
                "html": self.formatter.markdown_to_html(mindmap_data) if isinstance(mindmap_data, str) else self.formatter.json_to_html(mindmap_data)
            }
        else:
            # For JSON format, keep existing behavior
            formats = {
                "json": mindmap_data,
                "mermaid": self.formatter.json_to_mermaid(mindmap_data),
                "markdown": self.formatter.json_to_markdown(mindmap_data),
                "html": self.formatter.json_to_html(mindmap_data)
            }
        
        # Metadata - keep only essential information
        metadata = {
            "language": language,
            "output_format": output_format,
            "node_count": self.formatter.count_nodes(mindmap_data) if output_format == "json" else self.formatter.count_markdown_nodes(mindmap_data),
            "max_depth": self.formatter.get_actual_depth(mindmap_data) if output_format == "json" else self.formatter.get_markdown_depth(mindmap_data),
            "processing_time_seconds": round(duration, 2)
        }
        
        return {
            "mindmap": mindmap_data,
            "formats": formats,
            "metadata": metadata
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        return "text" in input_data and input_data["text"] and len(input_data["text"].strip()) > 0

    def _get_capabilities(self) -> List[str]:
        """Get list of tool capabilities."""
        return [
            "hierarchical_structure",
            "concept_extraction",
            "relationship_mapping",
            "multilingual_support",
            "json_output",
            "mermaid_output",
            "markdown_output",
            "html_output",
            "markdown_generation",
            "markmap_compatible",
            "structure_aware_chunking",
            "token_based_chunking",
            "hierarchical_processing",
            "recursive_collapse",
            "map_reduce_pipeline",
            "parallel_processing",
            "context_window_optimization",
            "duplicate_removal",
            "enhanced_json_validation",
            "parent_child_matching",
            "qwen_transformers_support",
            "tiktoken_support"
        ]

    def _get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ["vietnamese", "english"]
    
    def get_root_topic(self) -> str:
        """Get the root topic for proper export formatting."""
        return getattr(self, '_root_topic', 'Mindmap')
    
    def set_root_topic(self, topic: str):
        """Set the root topic for proper export formatting."""
        self._root_topic = topic