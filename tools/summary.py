"""
Production-Ready Summary Tool – Map-Reduce with Recursive Collapse.
Implements full map-reduce pipeline: Map → Recursive Collapse → Reduce
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import asyncio
import re

from .base import BaseTool
from .prompts import get_prompt

logger = logging.getLogger(__name__)


class SummaryTool(BaseTool):
    """
    Document summarization tool using map-reduce with recursive collapse.
    
    Features:
    - Structure-aware chunking (headings/sections)
    - Token-based chunking fallback
    - **Recursive collapse of intermediate summaries**
    - Configurable summary length and type
    - Comprehensive metadata tracking
    """

    def __init__(self, llm, config: Dict[str, Any]):
        super().__init__(llm, config)
        self.default_summary_type = config.get("summary_type", "abstractive")
        self.default_language = config.get("language", "vietnamese")
        self.max_summary_length = config.get("max_length", 2000)
        self.context_window = config.get("context_window", 16000)
        
        # Token limits for collapse logic
        self.chunk_size_tokens = config.get("chunk_size_tokens", 2048)
        self.chunk_overlap_tokens = config.get("chunk_overlap_tokens", 200)
        self.token_max = config.get("token_max", 3000)  # Max tokens before collapse
        
        # Get model info from LLM instance instead of config
        try:
            llm_info = self.llm.get_model_info()
            self.model_name = llm_info.get("model_name", "gpt-4")
            self.provider = llm_info.get("provider", "openai")
        except Exception:
            self.model_name = "gpt-4"
            self.provider = "openai"
        self._init_tokenizer()
        
        # Tracking for metadata
        self._collapse_iterations = 0
        self._total_chunks_processed = 0

    def _init_tokenizer(self):
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
        
        else :
            try:
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("o200k_base")
                self.tokenizer_type = "tiktoken"
                logger.info(f"✓ Loaded tiktoken")
            except Exception as e:
                logger.error(f"✗ Tokenizer loading failed: {e}")
                self.tokenizer = None
                self.tokenizer_type = None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            try:
                if self.tokenizer_type == "transformers":
                    return len(self.tokenizer.encode(text))
                else:  # tiktoken
                    return len(self.tokenizer.encode(text))
            except Exception:
                return len(text) // 4
        return len(text) // 4

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks using structure-aware strategy.
        
        Priority:
        1. If text fits in context window → no chunking
        2. Try section/heading-based splitting
        3. Fallback to token-based chunking
        """
        # Use 80% of context window for safety (leave room for prompts)
        safe_chunk_size = int(self.context_window * 0.8)
        
        if self._count_tokens(text) <= safe_chunk_size:
            return [text]
        
        # Try structure-aware splitting
        section_pattern = r"(?m)^(#{1,3}\s+.+$|CHƯƠNG|PHẦN|CHAPTER|SECTION|MỤC|BÀI|PART|\n[A-Z\s]{5,}\n)"
        sections = [s.strip() for s in re.split(section_pattern, text) if len(s.strip()) > 50]
        
        chunks = []
        for section in sections:
            section_tokens = self._count_tokens(section)
            if section_tokens > safe_chunk_size:
                # Section too large, split by tokens
                chunks.extend(self._token_chunk(section, max_size=safe_chunk_size))
            else:
                chunks.append(section)
        
        # Validate all chunks
        valid_chunks = []
        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk)
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
        try :
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
            logger.warning(f"LangChain splitter failed: {e}")


        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=(max_size or self.chunk_size_tokens) * 2.5,
                chunk_overlap=self.chunk_overlap_tokens * 2.5
            )
            return splitter.split_text(text)
        except Exception as e:
            logger.error(f"Character splitter failed: {e}")

        # Final fallback
        return [text]
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Main processing pipeline with map-reduce + recursive collapse."""
        if not self.validate_input(input_data):
            raise ValueError("Input must have 'text'")
        
        start_time = datetime.now()
        self._collapse_iterations = 0
        self._total_chunks_processed = 0

        # Extract parameters
        text = input_data.get("text", "")
        summary_type = kwargs.get("summary_type", self.default_summary_type)
        language = kwargs.get("language", self.default_language)
        max_length = kwargs.get("max_length", self.max_summary_length)

        if not text or len(text.strip()) < 50:
            raise ValueError("Text too short to summarize (>50 characters)")

        # Step 1: Chunking
        chunks = self._split_text_into_chunks(text)
        self._total_chunks_processed = len(chunks)
        logger.info("Split text into %d chunks", len(chunks))

        # Step 2: Direct summarization if single chunk
        if len(chunks) == 1:
            final_summary = await self._summarize_chunk(
                text, summary_type, language, max_length
            )
        else:
            # Step 3: Map - Summarize each chunk in parallel
            logger.info("Map: Summarizing %d chunks in parallel...", len(chunks))
            chunk_summaries = await asyncio.gather(*[
                self._summarize_chunk(
                    chunk, 
                    summary_type, 
                    language, 
                    max_length // len(chunks)
                )
                for chunk in chunks
            ])
            
            # Step 4: Recursive Collapse if needed
            collapsed_summaries = await self._recursive_collapse(
                chunk_summaries, 
                summary_type, 
                language, 
                max_length
            )
            
            # Step 5: Final Reduce
            logger.info("Reduce: Merging final summary...")
            final_summary = await self._final_reduce(
                collapsed_summaries, 
                summary_type, 
                language, 
                max_length
            )

        duration = (datetime.now() - start_time).total_seconds()

        # Metadata - keep only essential information
        metadata = {
            "summary_type": summary_type,
            "language": language,
            "processing_time_seconds": round(duration, 2)
        }
        
        return {
            "summary": final_summary.strip(), 
            "metadata": metadata
        }

    async def _recursive_collapse(
        self, 
        summaries: List[str], 
        summary_type: str, 
        language: str,
        max_length: int
    ) -> List[str]:
        """
        Recursively collapse summaries if they exceed token_max.
        This is the KEY component of map-reduce summarization.
        
        Args:
            summaries: List of summary strings to collapse
            summary_type: Type of summary (abstractive/extractive)
            language: Target language
            max_length: Maximum length for final summary
            
        Returns:
            List of collapsed summaries that fit within token_max
        """
        # Calculate total tokens in all summaries
        total_tokens = sum(self._count_tokens(s) for s in summaries)
        logger.info(f"Collapse check: {total_tokens} tokens in {len(summaries)} summaries")
        
        # If within limit, no collapse needed
        if total_tokens <= self.token_max:
            logger.info(f"✓ Summaries fit within token_max ({self.token_max}), no collapse needed")
            return summaries
        
        # Need to collapse
        self._collapse_iterations += 1
        logger.info(f"Collapse iteration {self._collapse_iterations}: {total_tokens} > {self.token_max}")
        
        # Split summaries into groups that fit within token_max
        groups = self._split_summaries_into_groups(summaries)
        logger.info(f"plit {len(summaries)} summaries into {len(groups)} groups")
        
        # Collapse each group in parallel
        collapsed = await asyncio.gather(*[
            self._collapse_group(group, summary_type, language, max_length)
            for group in groups
        ])
        
        # Recursively collapse if still too large
        return await self._recursive_collapse(collapsed, summary_type, language, max_length)

    def _split_summaries_into_groups(self, summaries: List[str]) -> List[List[str]]:
        """
        Split summaries into groups that each fit within token_max.
        
        Args:
            summaries: List of summary strings
            
        Returns:
            List of summary groups
        """
        groups = []
        current_group = []
        current_tokens = 0
        
        for summary in summaries:
            summary_tokens = self._count_tokens(summary)
            
            # If adding this summary exceeds limit, start new group
            if current_tokens + summary_tokens > self.token_max and current_group:
                groups.append(current_group)
                current_group = [summary]
                current_tokens = summary_tokens
            else:
                current_group.append(summary)
                current_tokens += summary_tokens
        
        # Add remaining group
        if current_group:
            groups.append(current_group)
        
        return groups

    async def _collapse_group(
        self, summaries: List[str], summary_type: str, language: str, max_length: int
    ) -> str:
        combined = "\n\n".join(summaries)
        if language == "vietnamese":
            prompt = (
                "Gộp các danh sách ý chính dưới đây thành MỘT danh sách tổng hợp có cấu trúc rõ ràng.\n\n"
                "YÊU CẦU TRÌNH BÀY:\n"
                "- Sử dụng dấu gạch đầu dòng '-' cho mỗi ý\n"
                "- Nhóm các ý liên quan dưới tiêu đề chủ đề (dùng ###)\n"
                "- Loại bỏ thông tin trùng lặp hoặc tương tự nhau\n"
                "- Ưu tiên giữ lại kiến thức và sự kiện quan trọng nhất\n"
                f"- Tổng cộng không quá {max_length} từ\n\n"
                "CÁC DANH SÁCH CẦN GỘP:\n"
                f"{combined}"
            )
        else:
            
            prompt = (
                "Merge the lists below into ONE consolidated structured list.\n\n"
                "PRESENTATION REQUIREMENTS:\n"
                "- Use dash '-' bullet points for each point\n"
                "- Group related points under topic headers (use ###)\n"
                "- Remove duplicate or similar information\n"
                "- Prioritize keeping most important knowledge and events\n"
                f"- Maximum {max_length} words total\n\n"
                "LISTS TO MERGE:\n"
                f"{combined}"
            )
        system_prompt = get_prompt("summary", language, summary_type, max_length=max_length)
        try:
            collapsed = await self.llm.agenerate(
                user_prompt=prompt,
                system_prompt=system_prompt,
            )
            return self._post_process_summary(collapsed, language)
        except Exception as e:
            logger.error(f"Collapse group failed: {e}")
            return combined[:max_length]


    async def _final_reduce(
        self, summaries: List[str], summary_type: str, language: str, max_length: int
    ) -> str:
        combined = "\n\n".join(summaries)
        if language == "vietnamese":
            prompt = (
                "Tổng hợp các danh sách ý chính dưới đây thành bản tổng hợp tri thức cuối cùng hoàn chỉnh.\n\n"
                "YÊU CẦU:\n"
                "- Sắp xếp theo nhóm chủ đề hoặc mức độ liên quan\n"
                "- Đảm bảo giữ đầy đủ tất cả kiến thức thiết yếu\n"
                "- Trình bày rõ ràng, dễ đọc và dễ tra cứu\n"
                "- Sử dụng dấu gạch đầu dòng và tiêu đề phân nhóm\n"
                f"- Không quá {max_length} từ\n\n"
                "CÁC DANH SÁCH CẦN TỔNG HỢP:\n"
                f"{combined}"
            )
        else:
            prompt = (
                "Consolidate the lists below into final comprehensive knowledge summary.\n\n"
                "REQUIREMENTS:\n"
                "- Organize by topic groups or relevance level\n"
                "- Ensure all essential knowledge is retained\n"
                "- Clear presentation, easy to read and reference\n"
                "- Use bullet points and group headers\n"
                f"- Maximum {max_length} words\n\n"
                "LISTS TO CONSOLIDATE:\n"
                f"{combined}"
            )
        system_prompt = get_prompt("summary", language, summary_type, max_length=max_length)
        try:
            final = await self.llm.agenerate(
                user_prompt=prompt,
                system_prompt=system_prompt
            )
            return self._post_process_summary(final, language)
        except Exception as e:
            logger.error(f"Final reduce failed: {e}")
            return combined


    async def _summarize_chunk(
        self, text: str, summary_type: str, language: str, max_length: int
    ) -> str:
        system_prompt = get_prompt("summary", language, summary_type, max_length=max_length)

        if summary_type == "abstractive":
            if language == "vietnamese":
                user_prompt = (
                    "Trích xuất và tổng hợp các ý chính, sự kiện quan trọng, khái niệm và định nghĩa nổi bật từ đoạn văn dưới đây. "
                    "Trình bày dưới dạng danh sách có gạch đầu dòng hoặc thẻ tri thức, "
                    f"tối đa {max_length} từ, chỉ giữ lại thông tin mới và quan trọng nhất.\n\n{text}"
                )
            else:
                user_prompt = (
                    "Extract and organize main ideas, important events, concepts, and definitions from the passage below. "
                    "Present as bulleted list or knowledge cards, "
                    f"maximum {max_length} words, only retain new and most important information.\n\n{text}"
                )
        else:
            if language == "vietnamese":
                user_prompt = (
                    f"Trích xuất các câu hoặc ý chính, sự kiện, khái niệm quan trọng nhất từ đoạn văn dưới đây. "
                    f"Mỗi ý một dòng riêng để dễ tra cứu, tránh lặp lại, tối đa {max_length} từ.\n\n{text}"
                )
            else:
                user_prompt = (
                    f"Extract important sentences or key ideas, events, concepts from the passage below. "
                    f"Each on separate line for easy reference, avoid repetition, maximum {max_length} words.\n\n{text}"
                )
        try:
            summary = await self.llm.agenerate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
            )
            return self._post_process_summary(summary, language)
        except Exception as e:
            logger.error(f"Chunk summarization failed: {e}")
            return ""

    def _post_process_summary(self, summary: str, language: str) -> str:
        """Clean up summary output while PRESERVING newlines."""
        if not summary:
            return ""
        
        prefixes = (
            ["Tóm tắt:", "Tóm lại:", "Dưới đây là tóm tắt:", "Bản tóm tắt:", "Dưới đây là"]
            if language == "vietnamese"
            else ["Summary:", "In summary:", "Here is a summary:", "The summary:", "Here is"]
        )
        
        processed = summary.strip()
        

        for prefix in prefixes:
            if processed.startswith(prefix):
                processed = processed[len(prefix):].strip()
                break
        
        lines = processed.split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines]
        result_lines = []
        prev_empty = False
        for line in cleaned_lines:
            if not line.strip():
                if not prev_empty:
                    result_lines.append('')
                prev_empty = True
            else:
                result_lines.append(line)
                prev_empty = False
        
        return '\n'.join(result_lines).strip()


    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields."""
        return "text" in input_data and input_data["text"] and len(input_data["text"].strip()) > 0

    def _get_capabilities(self) -> List[str]:
        """Get tool capabilities."""
        return [
            "abstractive_summarization",
            "extractive_summarization",
            "multilingual_support",
            "configurable_length",
            "structure_aware_chunking",
            "token_based_chunking",
            "recursive_collapse",
            "map_reduce_pipeline",
            "parallel_processing"
        ]
    
    def _get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return ["vietnamese", "english"]
