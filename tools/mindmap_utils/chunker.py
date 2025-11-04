"""
Mindmap chunking module for structure-aware text splitting.
Handles document structure detection and intelligent chunking strategies.
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MindmapChunker:
    """
    Handles text chunking for mindmap generation with structure-aware strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize chunker with configuration.
        
        Args:
            config: Configuration dictionary containing chunking parameters
        """
        self.context_window = config.get("context_window", 16000)
        self.chunk_size_tokens = config.get("chunk_size_tokens", 2048)
        self.chunk_overlap_tokens = config.get("chunk_overlap_tokens", 200)
        self.tokenizer = None
        self.tokenizer_type = None
        
    def set_tokenizer(self, tokenizer, tokenizer_type: str):
        """Set the tokenizer for token counting."""
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using configured tokenizer."""
        if self.tokenizer:
            try:
                if self.tokenizer_type == "transformers":
                    return len(self.tokenizer.encode(text))
                else:  # tiktoken
                    return len(self.tokenizer.encode(text))
            except Exception:
                return len(text) // 4
        return len(text) // 4
    
    def detect_document_structure(self, text: str) -> List[Dict]:
        """
        Detect document structure with headings, sections, lists.
        Returns list of sections with metadata (level, title, content).
        """
        sections = []
        
        # Regex patterns for different heading types
        heading_patterns = [
            (r'^# (.+)$', 1),      # H1 with space
            (r'^## (.+)$', 2),     # H2 with space
            (r'^### (.+)$', 3),   # H3 with space
            (r'^#### (.+)$', 4),    # H4 with space
            (r'^#(.+)$', 1),       # H1 without space (PDF converted)
            (r'^##(.+)$', 2),      # H2 without space (PDF converted)
            (r'^###(.+)$', 3),     # H3 without space (PDF converted)
            (r'^####(.+)$', 4),    # H4 without space (PDF converted)
            (r'^(CHƯƠNG|PHẦN|MỤC|BÀI|PART|CHAPTER)\s*(.+)$', 1),  # Vietnamese sections
            (r'^(Chapter|Section|Part)\s*(.+)$', 1),  # English sections
        ]
        
        # Split by headings
        lines = text.split('\n')
        current_section = {"level": 0, "title": "Introduction", "content": [], "parent": None}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for headings
            matched = False
            for pattern, level in heading_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous section
                    if current_section["content"]:
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        "level": level,
                        "title": match.group(1).strip(),
                        "content": [],
                        "parent": self._find_parent_section(sections, level)
                    }
                    matched = True
                    break
            
            if not matched:
                # Add to current section content
                current_section["content"].append(line)
        
        # Add last section
        if current_section["content"]:
            sections.append(current_section)
            
        return sections
    
    def _find_parent_section(self, sections: List[Dict], current_level: int) -> Optional[str]:
        """Find appropriate parent section based on level"""
        for section in reversed(sections):
            if section["level"] < current_level:
                return section["title"]
        return None
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks using structure-aware strategy.
        
        Priority:
        1. If text fits in context window → no chunking
        2. Try section/heading-based splitting
        3. Fallback to token-based chunking
        """
        # Use 80% of context window for safety (leave room for prompts)
        safe_chunk_size = int(self.context_window * 0.8)
        
        if self.count_tokens(text) <= safe_chunk_size:
            return [text]
        
        # Try structure-aware splitting
        section_pattern = r"(?m)^(#{1,3}\s+.+$|CHƯƠNG|PHẦN|CHAPTER|SECTION|MỤC|BÀI|PART|\n[A-Z\s]{5,}\n)"
        sections = [s.strip() for s in re.split(section_pattern, text) if len(s.strip()) > 50]
        
        chunks = []
        for section in sections:
            section_tokens = self.count_tokens(section)
            if section_tokens > safe_chunk_size:
                # Section too large, split by tokens
                chunks.extend(self._token_chunk(section, max_size=safe_chunk_size))
            else:
                chunks.append(section)
        
        # Validate all chunks
        valid_chunks = []
        for chunk in chunks:
            chunk_tokens = self.count_tokens(chunk)
            if chunk_tokens <= safe_chunk_size:
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Chunk still too large ({chunk_tokens} tokens), re-chunking")
                valid_chunks.extend(self._token_chunk(chunk, max_size=safe_chunk_size))
        
        if not valid_chunks:
            logger.warning("Structure splitting failed, using token chunking")
            return self._token_chunk(text, max_size=safe_chunk_size)
        
        return valid_chunks
    
    def _token_chunk(self, text: str, max_size: Optional[int] = None) -> List[str]:
        """Fallback: split text into token-based chunks with overlap."""
        if not self.tokenizer:
            return [text]
        try:
            chunk_size = max_size if max_size is not None else self.chunk_size_tokens
            
            if self.tokenizer_type == "transformers":
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
            else:
                tokens = self.tokenizer.encode(text)
            
            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_tokens = tokens[start:end]
                
                if self.tokenizer_type == "transformers":
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                else:
                    chunk_text = self.tokenizer.decode(chunk_tokens)
                
                chunks.append(chunk_text)
                start = end - self.chunk_overlap_tokens if end < len(tokens) else end
            
            if len(chunks) > 1:
                return chunks
        except Exception as e:
            logger.warning(f"Token chunking failed: {e}")

        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=(max_size or self.chunk_size_tokens) * 2.5,
                chunk_overlap=self.chunk_overlap_tokens * 2.5
            )
            return splitter.split_text(text)
        except Exception as e:
            logger.error(f"Character splitter failed: {e}")

        # Fallback cuối cùng
        return [text]
    
    def mindmap_chunk(self, text: str) -> List[Dict]:
        """
        Chunk text using structure-aware approach for mindmap.
        Each chunk is a complete section with metadata.
        """
        # Check if text fits in context window
        if self.count_tokens(text) <= int(self.context_window * 0.8):
            return [{
                "content": text,
                "level": 0,
                "title": "Main Topic",
                "parent": None,
                "tokens": self.count_tokens(text)
            }]
        
        # Detect document structure
        sections = self.detect_document_structure(text)
        
        chunks = []
        for section in sections:
            section_text = '\n'.join(section["content"])
            token_count = self.count_tokens(section_text)
            
            if token_count <= int(self.context_window * 0.8):
                # Section fits in one chunk
                chunks.append({
                    "content": section_text,
                    "level": section["level"],
                    "title": section["title"],
                    "parent": section["parent"],
                    "tokens": token_count
                })
            else:
                # Section too long, split further
                sub_chunks = self._split_large_section(section)
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _split_large_section(self, section: Dict) -> List[Dict]:
        """
        Split large section into smaller chunks while preserving metadata.
        """
        content = '\n'.join(section["content"])
        tokens = self.count_tokens(content)
        
        if not self.tokenizer:
            return [{
                "content": content,
                "level": section["level"],
                "title": section["title"],
                "parent": section["parent"],
                "tokens": tokens
            }]
        
        # Split by tokens
        if self.tokenizer_type == "transformers":
            token_list = self.tokenizer.encode(content, add_special_tokens=False)
        else:
            token_list = self.tokenizer.encode(content)
        
        chunks = []
        
        start = 0
        chunk_index = 0
        while start < len(token_list):
            end = min(start + self.chunk_size_tokens, len(token_list))
            chunk_tokens = token_list[start:end]
            
            if self.tokenizer_type == "transformers":
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            else:
                chunk_text = self.tokenizer.decode(chunk_tokens)
            
            chunks.append({
                "content": chunk_text,
                "level": section["level"],
                "title": f"{section['title']} (Part {chunk_index + 1})",
                "parent": section["parent"],
                "tokens": len(chunk_tokens)
            })
            
            start = end - self.chunk_overlap_tokens if end < len(token_list) else end
            chunk_index += 1
        
        return chunks