"""
Utility functions for tools module.
Contains helper functions for text processing and validation.
"""

import re
import json
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()


def truncate_text_intelligently(text: str, max_length: int, respect_sentences: bool = True) -> str:
    """
    Intelligently truncate text while preserving sentence boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        respect_sentences: Whether to try to end at sentence boundary
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    if not respect_sentences:
        return text[:max_length] + "..."
    
    # Try to find sentence boundary near max_length
    truncated = text[:max_length]
    
    # Look for sentence endings
    sentence_endings = ['.', '!', '?', '。', '！', '？']
    best_pos = -1
    
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > best_pos and pos > max_length * 0.8:  # At least 80% of max_length
            best_pos = pos
    
    if best_pos > 0:
        return text[:best_pos + 1]
    else:
        # No good sentence boundary, truncate at word boundary
        words = truncated.split()
        result = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_length - 3:  # Leave room for "..."
                result.append(word)
                current_length += len(word) + 1
            else:
                break
        
        return ' '.join(result) + "..."
    
    return text[:max_length] + "..."


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text using simple heuristics.
    
    Args:
        text: Text to analyze
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        List of key phrases
    """
    # Simple heuristic: look for noun phrases and important terms
    # This is a basic implementation - could be enhanced with NLP
    
    # Split into sentences
    sentences = re.split(r'[.!?。！？]', text)
    
    # Extract potential phrases (2-4 word sequences)
    phrases = []
    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) < 2:
            continue
            
        # Look for 2-4 word sequences
        for i in range(len(words) - 1):
            for length in range(2, min(5, len(words) - i + 1)):
                phrase = ' '.join(words[i:i + length])
                if len(phrase) > 20:  # Skip very long phrases
                    continue
                    
                # Simple filtering: start with capital or contain important words
                if (phrase[0].isupper() or 
                    any(word in phrase.lower() for word in 
                        ['quan trọng', 'important', 'chính', 'main', 'cốt lõi', 'core'])):
                    phrases.append(phrase)
    
    # Remove duplicates and limit
    unique_phrases = list(dict.fromkeys(phrases))  # Preserve order while removing duplicates
    return unique_phrases[:max_phrases]


def validate_json_structure(data: Any, max_depth: int = 5, current_depth: int = 0) -> bool:
    """
    Validate JSON structure for mindmap compatibility.
    
    Args:
        data: Data to validate
        max_depth: Maximum allowed depth
        current_depth: Current depth level
        
    Returns:
        True if valid structure
    """
    if current_depth > max_depth:
        return False
    
    if isinstance(data, dict):
        # Check required fields for mindmap nodes
        if "id" in data or "text" in data or "children" in data:
            # Validate children if present
            if "children" in data:
                if not isinstance(data["children"], list):
                    return False
                for child in data["children"]:
                    if not validate_json_structure(child, max_depth, current_depth + 1):
                        return False
        return True
    
    elif isinstance(data, list):
        for item in data:
            if not validate_json_structure(item, max_depth, current_depth):
                return False
        return True
    
    return True  # Other types are valid at leaf level


def estimate_reading_time(text: str, language: str = "vietnamese") -> float:
    """
    Estimate reading time in minutes.
    
    Args:
        text: Text to analyze
        language: Language code
        
    Returns:
        Estimated reading time in minutes
    """
    if not text:
        return 0.0
    
    # Words per minute by language
    wpm_rates = {
        "vietnamese": 180,  # Vietnamese reading speed
        "english": 220,     # English reading speed
    }
    
    wpm = wpm_rates.get(language, 200)
    
    # Count words (simple split by whitespace)
    word_count = len(text.split())
    
    return max(0.5, word_count / wpm)  # Minimum 30 seconds


def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns.
    
    Args:
        text: Text to analyze
        
    Returns:
        Language code ('vietnamese', 'english', or 'unknown')
    """
    if not text or len(text) < 50:
        return "unknown"
    
    # Count Vietnamese-specific characters
    vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]', text, re.IGNORECASE))
    
    # Count total characters (excluding spaces and punctuation)
    total_chars = len(re.sub(r'[\s\W]', '', text))
    
    if total_chars == 0:
        return "unknown"
    
    vietnamese_ratio = vietnamese_chars / total_chars
    
    # If more than 5% Vietnamese characters, assume Vietnamese
    if vietnamese_ratio > 0.05:
        return "vietnamese"
    
    # Additional heuristics for English
    english_words = len(re.findall(r'\b[a-zA-Z]+\b', text))
    total_words = len(text.split())
    
    if total_words > 0:
        english_ratio = english_words / total_words
        if english_ratio > 0.8:
            return "english"
    
    return "unknown"


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def create_error_response(error_message: str, tool_name: str) -> Dict[str, Any]:
    """
    Create standardized error response for tools.
    
    Args:
        error_message: Error description
        tool_name: Name of the tool
        
    Returns:
        Error response dictionary
    """
    return {
        "error": True,
        "error_message": error_message,
        "tool": tool_name,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
    
    return filename.strip()


class TextAnalyzer:
    """
    Utility class for text analysis operations.
    """
    
    @staticmethod
    def get_text_statistics(text: str) -> Dict[str, Any]:
        """
        Get comprehensive text statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "paragraph_count": 0,
                "avg_words_per_sentence": 0,
                "reading_time": 0.0
            }
        
        # Basic counts
        char_count = len(text)
        word_count = len(text.split())
        
        # Sentence count (simple heuristic)
        sentence_endings = r'[.!?。！？]+'
        sentences = re.split(sentence_endings, text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Paragraph count
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Average words per sentence
        avg_words = word_count / sentence_count if sentence_count > 0 else 0
        
        # Reading time
        language = detect_language(text)
        reading_time = estimate_reading_time(text, language)
        
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_words_per_sentence": round(avg_words, 1),
            "reading_time_minutes": round(reading_time, 1),
            "detected_language": language
        }
    
    @staticmethod
    def extract_structure(text: str) -> Dict[str, Any]:
        """
        Extract document structure (headings, lists, etc.).
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with structure information
        """
        structure = {
            "headings": [],
            "lists": [],
            "links": [],
            "emphasis": []
        }
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Headings (markdown-style)
            if line.strip().startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                content = line.strip('#').strip()
                structure["headings"].append({
                    "line": i + 1,
                    "level": level,
                    "content": content
                })
            
            # Lists (bullet points, numbered lists)
            if re.match(r'^\s*[-*+]\s', line) or re.match(r'^\s*\d+\.\s', line):
                structure["lists"].append({
                    "line": i + 1,
                    "content": line.strip()
                })
            
            # Links (URLs)
            urls = re.findall(r'https?://[^\s]+', line)
            for url in urls:
                structure["links"].append({
                    "line": i + 1,
                    "url": url
                })
            
            # Emphasis (bold, italic)
            if '**' in line or '*' in line or '_' in line:
                structure["emphasis"].append({
                    "line": i + 1,
                    "content": line.strip()
                })
        
        return structure