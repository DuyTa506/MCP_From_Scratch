"""
Mindmap processor module for handling core processing logic.
Contains map-reduce pipeline and LLM interaction methods.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from .formatter import MindmapFormatter
from ..prompts import get_prompt

logger = logging.getLogger(__name__)


class MindmapProcessor:
    """
    Handles core mindmap processing logic including map-reduce pipeline.
    """
    
    def __init__(self, llm, config: Dict[str, Any]):
        """
        Initialize processor with LLM instance and configuration.
        
        Args:
            llm: LLM instance for processing
            config: Configuration dictionary
        """
        self.llm = llm
        self.config = config
        self.token_max = config.get("token_max", 3000)
        self.default_temperature = config.get("temperature", 0.5)
        self.formatter = MindmapFormatter()
        
        # Tracking for metadata
        self._collapse_iterations = 0
        self._total_chunks_processed = 0
    
    async def process_single_chunk(
        self, text: str, language: str, max_nodes: int, max_depth: int, 
        temperature: float, output_format: str
    ) -> Any:
        """
        Process a single chunk of text for mindmap generation.
        
        Args:
            text: Text chunk to process
            language: Target language
            max_nodes: Maximum nodes for mindmap
            max_depth: Maximum depth for mindmap
            temperature: LLM temperature
            output_format: Output format (json or markdown)
            
        Returns:
            Processed mindmap in specified format
        """
        if output_format == "markdown":
            return await self._generate_markdown_mindmap_direct(
                text, language, max_nodes, max_depth, temperature
            )
        else:
            return await self._generate_mindmap_direct(
                text, language, max_nodes, max_depth, temperature
            )
    
    async def process_chunks(
        self, chunks: List[str], language: str, max_nodes: int, 
        max_depth: int, temperature: float, output_format: str
    ) -> Any:
        """
        Process multiple chunks using map-reduce pipeline.
        
        Args:
            chunks: List of text chunks
            language: Target language
            max_nodes: Maximum nodes for mindmap
            max_depth: Maximum depth for mindmap
            temperature: LLM temperature
            output_format: Output format (json or markdown)
            
        Returns:
            Processed and merged mindmap
        """
        self._total_chunks_processed = len(chunks)
        
        # Map Phase: Process each chunk in parallel
        logger.info(f"Map: Processing {len(chunks)} chunks in parallel...")
        
        if output_format == "markdown":
            chunk_mindmaps = await asyncio.gather(*[
                self._extract_markdown_mindmap_from_chunk(
                    {"content": chunk, "level": 0, "title": f"Chunk {i+1}", "parent": None, "tokens": self._count_tokens(chunk)},
                    language, max_nodes, max_depth, temperature
                )
                for i, chunk in enumerate(chunks)
            ])
            
            # Recursive Collapse if needed
            collapsed_mindmaps = await self._recursive_collapse(
                chunk_mindmaps, language, max_nodes, max_depth
            )
            
            # Reduce Phase: Final Merge
            logger.info(f"Reduce: Merging final mindmap...")
            mindmap_data = self._merge_markdown_mindmaps(collapsed_mindmaps, max_nodes, max_depth)
        else:
            chunk_mindmaps = await asyncio.gather(*[
                self._extract_mindmap_from_chunk(
                    {"content": chunk, "level": 0, "title": f"Chunk {i+1}", "parent": None, "tokens": self._count_tokens(chunk)},
                    language, max_nodes, max_depth, temperature
                )
                for i, chunk in enumerate(chunks)
            ])
            
            # Convert to markdown for collapse
            markdown_mindmaps = [self.formatter.json_to_markdown(mindmap) for mindmap in chunk_mindmaps]
            
            # Recursive Collapse if needed
            collapsed_mindmaps = await self._recursive_collapse(
                markdown_mindmaps, language, max_nodes, max_depth
            )
            
            # Reduce Phase: Final Merge and convert back to JSON
            logger.info(f"Reduce: Merging final mindmap...")
            merged_markdown = self._merge_markdown_mindmaps(collapsed_mindmaps, max_nodes, max_depth)
            mindmap_data = self.formatter.markdown_to_json(merged_markdown)
        
        return mindmap_data
    
    async def _recursive_collapse(
        self,
        summaries: List[str],
        language: str,
        max_nodes: int,
        max_depth: int
    ) -> List[str]:
        """
        Recursively collapse summaries if they exceed token_max.
        This is KEY component for handling long documents.
        
        Args:
            summaries: List of summary strings to collapse
            language: Target language
            max_nodes: Maximum nodes for mindmap
            max_depth: Maximum depth for mindmap
            
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
        logger.info(f"Split {len(summaries)} summaries into {len(groups)} groups")
        
        # Collapse each group in parallel
        collapsed = await asyncio.gather(*[
            self._collapse_group(group, language, max_nodes, max_depth)
            for group in groups
        ])
        
        # Recursively collapse if still too large
        return await self._recursive_collapse(collapsed, language, max_nodes, max_depth)
    
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
        self, summaries: List[str], language: str, max_nodes: int, max_depth: int
    ) -> str:
        combined = "\n\n".join(summaries)
        if language == "vietnamese":
            prompt = (
                "Gộp các cấu trúc mindmap dưới đây thành MỘT cấu trúc tổng hợp có cấu trúc rõ ràng.\n\n"
                "YÊU CẦU TRÌNH BÀY:\n"
                "- Sử dụng định dạng # cho main topic, ## cho level 1, ### cho level 2\n"
                "- Nhóm các ý liên quan dưới cùng một nhánh\n"
                "- Loại bỏ thông tin trùng lặp hoặc tương tự nhau\n"
                "- Ưu tiên giữ lại cấu trúc quan trọng nhất\n"
                f"- Tối đa {max_nodes} nodes và {max_depth} cấp độ sâu\n\n"
                "CÁC CẤU TRÚC CẦN GỘP:\n"
                f"{combined}"
            )
        else:
            prompt = (
                "Merge mindmap structures below into ONE consolidated structured mindmap.\n\n"
                "PRESENTATION REQUIREMENTS:\n"
                "- Use # for main topic, ## for level 1, ### for level 2\n"
                "- Group related items under same branch\n"
                "- Remove duplicate or similar information\n"
                "- Prioritize keeping most important structure\n"
                f"- Maximum {max_nodes} nodes and {max_depth} depth levels\n\n"
                "STRUCTURES TO MERGE:\n"
                f"{combined}"
            )
        
        system_prompt = get_prompt("markdown_mindmap", language)
        try:
            collapsed = await self.llm.agenerate(
                user_prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=2000
            )
            return self.formatter.extract_markdown_from_response(collapsed)
        except Exception as e:
            logger.error(f"Collapse group failed: {e}")
            return combined[:2000] if len(combined) > 2000 else combined
    
    async def _generate_mindmap_direct(
        self, text: str, language: str, max_nodes: int, max_depth: int, temperature: float
    ) -> Dict[str, Any]:
        """Generate mindmap directly from text (no chunking)"""
        system_prompt = get_prompt("mindmap", language)
        constraints = self._build_constraints_text(max_nodes, max_depth, language)
        
        user_prompt = self._build_user_prompt(text, constraints, language)
        
        try:
            response = await self.llm.agenerate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=2000
            )
            
            mindmap_json = self.formatter.extract_json_from_response(response)
            return self.formatter.validate_and_fix_json(mindmap_json, max_nodes, max_depth)
            
        except Exception as e:
            logger.error(f"Error generating direct mindmap: {e}")
            return self.formatter._create_fallback_json_mindmap(text, language)
    
    async def _generate_markdown_mindmap_direct(
        self, text: str, language: str, max_nodes: int, max_depth: int, temperature: float
    ) -> str:
        """Generate mindmap directly from text in markdown format (no chunking)"""
        system_prompt = get_prompt("markdown_mindmap", language)
        constraints = self._build_markdown_constraints_text(max_nodes, max_depth, language)
        
        user_prompt = self._build_markdown_user_prompt(text, constraints, language)
        
        try:
            response = await self.llm.agenerate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=2000
            )
            
            # Extract markdown from response
            markdown_content = self.formatter.extract_markdown_from_response(response)
            return self.formatter.validate_and_fix_markdown(markdown_content, max_nodes, max_depth, language)
            
        except Exception as e:
            logger.error(f"Error generating direct markdown mindmap: {e}")
            return self.formatter._create_fallback_markdown_mindmap(text, language)
    
    async def _extract_mindmap_from_chunk(
        self, chunk: Dict, language: str, max_nodes: int, max_depth: int, temperature: float
    ) -> Dict[str, Any]:
        """Extract mindmap nodes from chunk - routes to language-specific method"""
        if language == "vietnamese":
            return await self._extract_mindmap_from_chunk_vietnamese(chunk, max_nodes, max_depth, temperature)
        else:
            return await self._extract_mindmap_from_chunk_english(chunk, max_nodes, max_depth, temperature)
    
    async def _extract_mindmap_from_chunk_vietnamese(
        self, chunk: Dict, max_nodes: int, max_depth: int, temperature: float
    ) -> Dict[str, Any]:
        """Extract mindmap from chunk in Vietnamese"""
        system_prompt = get_prompt("mindmap", "vietnamese")
        
        prompt = f"""
    Từ đoạn văn sau (cấp độ {chunk['level']}, tiêu đề: "{chunk['title']}"),
    trích xuất cấu trúc sơ đồ tư duy với các chủ đề và chủ đề con.

    Yêu cầu:
    - Giữ nguyên cấp độ phân cấp ({chunk['level']})
    - Xác định chủ đề chính và chủ đề con rõ ràng
    - Tối đa {max_depth} cấp độ sâu
    - Xuất ra JSON hợp lệ

    Đoạn văn:
    {chunk['content']}

    Định dạng JSON:
    {{
        "main_topic": "...",
        "subtopics": ["...", "..."],
        "details": {{"subtopic1": ["detail1", "detail2"], ...}},
        "level": {chunk['level']},
        "parent_topic": "{chunk.get('parent', 'Không có')}"
    }}

    KẾT QUẢ PHẢI HOÀN TOÀN BẰNG TIẾNG VIỆT
    """
        
        try:
            response = await self.llm.agenerate(
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=1500
            )
            
            mindmap_data = self.formatter.extract_json_from_response(response)
            return self._convert_chunk_to_node(mindmap_data, chunk)
            
        except Exception as e:
            logger.error(f"Error extracting mindmap from chunk: {e}")
            return self._create_chunk_fallback(chunk)
    
    async def _extract_mindmap_from_chunk_english(
        self, chunk: Dict, max_nodes: int, max_depth: int, temperature: float
    ) -> Dict[str, Any]:
        """Extract mindmap from chunk in English"""
        system_prompt = get_prompt("mindmap", "english")
        
        prompt = f"""
    From text section (level {chunk['level']}, title: "{chunk['title']}"),
    extract mindmap structure with topics and subtopics.

    Requirements:
    - Preserve hierarchy level ({chunk['level']})
    - Identify main topic and subtopics clearly
    - Maximum {max_depth} depth levels
    - Output valid JSON

    Text section:
    {chunk['content']}

    JSON format:
    {{
        "main_topic": "...",
        "subtopics": ["...", "..."],
        "details": {{"subtopic1": ["detail1", "detail2"], ...}},
        "level": {chunk['level']},
        "parent_topic": "{chunk.get('parent', 'None')}"
    }}

    RESULT MUST BE ENTIRELY IN ENGLISH
    """
        
        try:
            response = await self.llm.agenerate(
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=1500
            )
            
            mindmap_data = self.formatter.extract_json_from_response(response)
            return self._convert_chunk_to_node(mindmap_data, chunk)
            
        except Exception as e:
            logger.error(f"Error extracting mindmap from chunk: {e}")
            return self._create_chunk_fallback(chunk)
    
    async def _extract_markdown_mindmap_from_chunk(
        self, chunk: Dict, language: str, max_nodes: int, max_depth: int, temperature: float
    ) -> str:
        """Extract markdown mindmap from chunk - routes to language-specific method"""
        if language == "vietnamese":
            return await self._extract_markdown_from_chunk_vietnamese(chunk, max_nodes, max_depth, temperature)
        else:
            return await self._extract_markdown_from_chunk_english(chunk, max_nodes, max_depth, temperature)
    
    async def _extract_markdown_from_chunk_vietnamese(
        self, chunk: Dict, max_nodes: int, max_depth: int, temperature: float
    ) -> str:
        """Extract markdown mindmap from chunk in Vietnamese"""
        system_prompt = get_prompt("markdown_mindmap", "vietnamese")
        
        prompt = f"""
    Từ đoạn văn sau (cấp độ {chunk['level']}, tiêu đề: "{chunk['title']}"),
    tạo cấu trúc sơ đồ tư duy dạng markdown tương thích với markmap.

    Yêu cầu:
    - Giữ nguyên cấp độ phân cấp ({chunk['level']})
    - Xác định chủ đề chính và chủ đề con rõ ràng
    - Tối đa {max_depth} cấp độ sâu
    - Dùng # cho chủ đề chính, ## cho cấp 1, ### cho cấp 2, #### cho cấp 3
    - Mỗi nút trên một dòng riêng biệt
    - Không dùng dấu gạch đầu dòng

    Đoạn văn:
    {chunk['content']}

    Định dạng markdown mẫu:
    # Chủ đề chính
    ## Chủ đề con 1
    ### Chi tiết 1.1
    #### Chi tiết 1.1.1
    ## Chủ đề con 2

    KẾT QUẢ PHẢI HOÀN TOÀN BẰNG TIẾNG VIỆT
    """
        
        try:
            response = await self.llm.agenerate(
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=1500
            )
            
            markdown_content = self.formatter.extract_markdown_from_response(response)
            return self.formatter.validate_and_fix_markdown(markdown_content, max_nodes, max_depth, "vietnamese")
            
        except Exception as e:
            logger.error(f"Error extracting markdown mindmap from chunk: {e}")
            return self._create_chunk_fallback_markdown(chunk)
    
    async def _extract_markdown_from_chunk_english(
        self, chunk: Dict, max_nodes: int, max_depth: int, temperature: float
    ) -> str:
        """Extract markdown mindmap from chunk in English"""
        system_prompt = get_prompt("markdown_mindmap", "english")
        
        prompt = f"""
    From text section (level {chunk['level']}, title: "{chunk['title']}"),
    create mindmap structure in markdown format compatible with markmap.

    Requirements:
    - Preserve hierarchy level ({chunk['level']})
    - Identify main topic and subtopics clearly
    - Maximum {max_depth} depth levels
    - Use # for main topic, ## for level 1, ### for level 2, #### for level 3
    - Each node on a separate line
    - Do not use bullet points

    Text section:
    {chunk['content']}

    Markdown format example:
    # Main Topic
    ## Subtopic 1
    ### Detail 1.1
    #### Detail 1.1.1
    ## Subtopic 2

    RESULT MUST BE ENTIRELY IN ENGLISH
    """
        
        try:
            response = await self.llm.agenerate(
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=1500
            )
            
            markdown_content = self.formatter.extract_markdown_from_response(response)
            return self.formatter.validate_and_fix_markdown(markdown_content, max_nodes, max_depth, "english")
            
        except Exception as e:
            logger.error(f"Error extracting markdown mindmap from chunk: {e}")
            return self._create_chunk_fallback_markdown(chunk)
    
    def _convert_chunk_to_node(self, mindmap_data: Dict, chunk: Dict) -> Dict[str, Any]:
        """Convert chunk mindmap data to node format"""
        if not isinstance(mindmap_data, dict):
            return self._create_chunk_fallback(chunk)
        
        return {
            "id": f"node_{uuid.uuid4().hex[:8]}",
            "text": chunk["title"],
            "level": chunk["level"],
            "parent": chunk.get("parent"),
            "main_topic": mindmap_data.get("main_topic", chunk["title"]),
            "subtopics": mindmap_data.get("subtopics", []),
            "details": mindmap_data.get("details", {}),
            "children": []
        }
    
    def _merge_markdown_mindmaps(self, chunk_markdowns: List[str], max_nodes: int, max_depth: int) -> str:
        """Merge markdown mindmaps from chunks"""
        if not chunk_markdowns:
            return self.formatter._create_fallback_markdown_mindmap("", "vietnamese")
        
        # Extract main topics and merge
        merged_lines = ["# Mindmap"]
        node_count = 1  # Count root
        
        for markdown in chunk_markdowns:
            lines = markdown.split('\n')
            for line in lines[1:]:  # Skip the first line (usually # title)
                line = line.strip()
                if line.startswith('#') and node_count < max_nodes:
                    merged_lines.append(line)
                    node_count += 1
        
        return '\n'.join(merged_lines)
    
    def _create_chunk_fallback(self, chunk: Dict) -> Dict[str, Any]:
        """Create fallback node for chunk when extraction fails"""
        return {
            "id": f"node_{uuid.uuid4().hex[:8]}",
            "text": chunk["title"],
            "level": chunk["level"],
            "parent": chunk.get("parent"),
            "children": []
        }
    
    def _create_chunk_fallback_markdown(self, chunk: Dict) -> str:
        """Create fallback markdown for chunk when extraction fails"""
        level_prefix = '#' * min(chunk["level"] + 1, 4)
        return f"""{level_prefix} {chunk["title"]}
## Could not extract details
"""
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using simple fallback."""
        return len(text) // 4
    
    def _build_constraints_text(self, max_nodes: int, max_depth: int, language: str) -> str:
        """Build constraints text for prompt - routes to language-specific method"""
        if language == "vietnamese":
            return self._build_constraints_vietnamese(max_nodes, max_depth)
        else:
            return self._build_constraints_english(max_nodes, max_depth)
    
    def _build_constraints_vietnamese(self, max_nodes: int, max_depth: int) -> str:
        """Build constraints text in Vietnamese"""
        return f"""
    Ràng buộc:
    - Tối đa {max_nodes} nút tổng cộng
    - Độ sâu tối đa {max_depth} cấp
    - Phải xuất ra JSON hợp lệ
    - Sử dụng định dạng JSON chính xác
    - Nội dung phải độc nhất, tránh trùng lặp
    - Phân cấp rõ ràng, không chỉ liệt kê phẳng
    - Các trường bắt buộc: id, text, children (mảng)
    """
    
    def _build_constraints_english(self, max_nodes: int, max_depth: int) -> str:
        """Build constraints text in English"""
        return f"""
    Constraints:
    - Maximum {max_nodes} nodes total
    - Maximum depth of {max_depth} levels
    - Must output valid JSON
    - Use exact JSON format
    - Content must be unique, avoid duplicates
    - Clear hierarchy, not just flat listing
    - Required fields: id, text, children (array)
    """
    
    def _build_user_prompt(self, text: str, constraints: str, language: str) -> str:
        """Build user prompt for mindmap generation - routes to language-specific method"""
        if language == "vietnamese":
            return self._build_user_prompt_vietnamese(text, constraints)
        else:
            return self._build_user_prompt_english(text, constraints)
    
    def _build_user_prompt_vietnamese(self, text: str, constraints: str) -> str:
        """Build user prompt in Vietnamese for JSON mindmap"""
        return f"""
    Tạo cấu trúc sơ đồ tư duy từ văn bản sau:

    {constraints}

    Văn bản cần phân tích:
    {text}

    Định dạng JSON yêu cầu (PHẢI tuân thủ chính xác):
    {{
        "id": "root",
        "text": "Chủ đề chính",
        "children": [
            {{
                "id": "node1",
                "text": "Chủ đề con 1",
                "children": [
                    {{
                        "id": "node1_1",
                        "text": "Chi tiết 1",
                        "children": []
                    }}
                ]
            }}
        ]
    }}

    LƯU Ý QUAN TRỌNG:
    - id: chuỗi định danh duy nhất cho mỗi nút
    - text: nội dung hiển thị của nút
    - children: mảng các nút con (có thể rỗng [])
    - KHÔNG được thiếu trường bắt buộc
    - KHÔNG được thêm trường ngoài định dạng
    - Đảm bảo JSON hợp lệ với dấu ngoặc đúng
    - KẾT QUẢ PHẢI HOÀN TOÀN BẰNG TIẾNG VIỆT
    """
    
    def _build_user_prompt_english(self, text: str, constraints: str) -> str:
        """Build user prompt in English for JSON mindmap"""
        return f"""
    Create mindmap structure from following text:

    {constraints}

    Text to analyze:
    {text}

    Required JSON format (MUST follow exactly):
    {{
        "id": "root",
        "text": "Main Topic",
        "children": [
            {{
                "id": "node1",
                "text": "Subtopic 1",
                "children": [
                    {{
                        "id": "node1_1",
                        "text": "Detail 1",
                        "children": []
                    }}
                ]
            }}
        ]
    }}

    IMPORTANT NOTES:
    - id: unique identifier string for each node
    - text: display content of node
    - children: array of child nodes (can be empty [])
    - DO NOT miss required fields
    - DO NOT add extra fields beyond format
    - Ensure valid JSON with proper brackets
    - RESULT MUST BE ENTIRELY IN ENGLISH
    """
    
    def _build_markdown_constraints_text(self, max_nodes: int, max_depth: int, language: str) -> str:
        """Build constraints text for markdown prompt - routes to language-specific method"""
        if language == "vietnamese":
            return self._build_markdown_constraints_vietnamese(max_nodes, max_depth)
        else:
            return self._build_markdown_constraints_english(max_nodes, max_depth)
    
    def _build_markdown_constraints_vietnamese(self, max_nodes: int, max_depth: int) -> str:
        """Build constraints for markdown in Vietnamese"""
        return f"""
    Ràng buộc:
    - Tối đa {max_nodes} nút tổng cộng
    - Độ sâu tối đa {max_depth} cấp
    - Sử dụng định dạng markdown chính xác
    - Nội dung phải độc nhất, tránh trùng lặp
    - Phân cấp rõ ràng, không chỉ liệt kê phẳng
    - Mỗi nút trên một dòng riêng biệt
    - Chỉ dùng ký hiệu # để chỉ cấp độ
    - Kết quả phải bằng tiếng Việt
    """
    
    def _build_markdown_constraints_english(self, max_nodes: int, max_depth: int) -> str:
        """Build constraints for markdown in English"""
        return f"""
    Constraints:
    - Maximum {max_nodes} nodes total
    - Maximum depth of {max_depth} levels
    - Use exact markdown format
    - Content must be unique, avoid duplicates
    - Clear hierarchy, not just flat listing
    - Each node on a separate line
    - Only use # symbols to indicate levels
    - Result must be in English
    """
    
    def _build_markdown_user_prompt(self, text: str, constraints: str, language: str) -> str:
        """Build user prompt for markdown mindmap generation - routes to language-specific method"""
        if language == "vietnamese":
            return self._build_markdown_prompt_vietnamese(text, constraints)
        else:
            return self._build_markdown_prompt_english(text, constraints)
    
    def _build_markdown_prompt_vietnamese(self, text: str, constraints: str) -> str:
        """Build Vietnamese prompt for markdown mindmap generation"""
        return f"""
    Tạo cấu trúc sơ đồ tư duy dạng markdown từ văn bản sau:

    {constraints}

    Văn bản cần phân tích:
    {text}

    YÊU CẦU VỀ ĐỊNH DẠNG MARKDOWN (BẮT BUỘC TUÂN THỦ):
    # Chủ đề chính
    ## Nhánh cấp 1 - Thứ nhất
    ### Nhánh cấp 2 - Mục 1.1
    #### Nhánh cấp 3 - Chi tiết 1.1.1
    ### Nhánh cấp 2 - Mục 1.2
    ## Nhánh cấp 1 - Thứ hai
    ### Nhánh cấp 2 - Mục 2.1

    LƯU Ý QUAN TRỌNG:
    - Dùng # cho chủ đề chính (chỉ một dòng)
    - Dùng ## cho các nhánh cấp 1
    - Dùng ### cho các nhánh cấp 2
    - Dùng #### cho các nhánh cấp 3 (chi tiết)
    - Mỗi nút trên một dòng riêng biệt
    - TUYỆT ĐỐI KHÔNG dùng dấu gạch đầu dòng (-, *, +)
    - Đảm bảo cú pháp markdown hợp lệ
    - Mỗi nhánh phải ngắn gọn, cô đọng (không quá 10 từ)
    - Tổ chức theo thứ bậc logic rõ ràng
    - KẾT QUẢ PHẢI HOÀN TOÀN BẰNG TIẾNG VIỆT
    """
    
    def _build_markdown_prompt_english(self, text: str, constraints: str) -> str:
        """Build English prompt for markdown mindmap generation"""
        return f"""
    Create mindmap structure in markdown format from the following text:

    {constraints}

    Text to analyze:
    {text}

    REQUIRED MARKDOWN FORMAT (MUST FOLLOW EXACTLY):
    # Main Topic
    ## Level 1 Branch - First
    ### Level 2 Branch - Item 1.1
    #### Level 3 Branch - Detail 1.1.1
    ### Level 2 Branch - Item 1.2
    ## Level 1 Branch - Second
    ### Level 2 Branch - Item 2.1

    IMPORTANT NOTES:
    - Use # for main topic (only one line)
    - Use ## for level 1 branches
    - Use ### for level 2 branches
    - Use #### for level 3 branches (details)
    - Each node on a separate line
    - ABSOLUTELY DO NOT use bullet points (-, *, +)
    - Ensure valid markdown syntax
    - Each branch must be concise (no more than 10 words)
    - Organize with clear logical hierarchy
    - RESULT MUST BE ENTIRELY IN ENGLISH
    """
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics for metadata."""
        return {
            "collapse_iterations": self._collapse_iterations,
            "total_chunks_processed": self._total_chunks_processed
        }
    
    def reset_stats(self):
        """Reset processing statistics."""
        self._collapse_iterations = 0
        self._total_chunks_processed = 0