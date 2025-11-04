"""
Prompt templates for tools module.
Contains multi-language prompts for different tool types.
"""

SUMMARY_PROMPTS = {
    "vietnamese": {
        "abstractive": """
Bạn là chuyên gia tóm tắt tri thức. Trích xuất và tổ chức nội dung thành danh sách các ý chính rõ ràng.

YÊU CẦU VỀ ĐỊNH DẠNG (BẮT BUỘC):
- Sử dụng dấu gạch đầu dòng "-" cho mỗi ý chính
- Mỗi ý chính là một dòng riêng biệt
- Nếu có nhiều nhóm chủ đề, dùng ### để phân loại
- Với định nghĩa hoặc thuật ngữ: **Tên thuật ngữ**: Giải thích ngắn gọn
- Tổng cộng không quá {max_length} từ


YÊU CẦU VỀ NỘI DUNG:
- Chỉ giữ lại thông tin mới, kiến thức cốt lõi
- Không lặp lại hoặc nêu nội dung lạc đề
- Ưu tiên các sự kiện, số liệu, định nghĩa quan trọng
- Ngôn ngữ súc tích, rõ ràng, dễ tra cứu sau này
        """,

        "extractive": """
Trích xuất các ý chính, câu quan trọng nhất, định nghĩa và khái niệm nổi bật từ văn bản. Trình bày dưới dạng danh sách có gạch đầu dòng.

YÊU CẦU VỀ ĐỊNH DẠNG:
- Mỗi ý chính là một dòng bắt đầu bằng dấu "-"
- Nếu có các nhóm logic khác nhau, dùng ### để tạo tiêu đề nhóm
- Giữ nguyên câu gốc quan trọng từ văn bản
- Không thêm thông tin từ bên ngoài văn bản gốc
- Sắp xếp theo thứ tự logic hoặc mức độ quan trọng


        """
    },
    "english": {
        "abstractive": """
You are a knowledge summarization expert. Extract and organize content into clear, structured key points.

FORMAT REQUIREMENTS (MANDATORY):
- Use dash "-" bullet points for each key point
- Each key point on a separate line
- Use ### for topic grouping if multiple themes exist
- For definitions or terms: **Term Name**: Brief explanation
- Maximum {max_length} words total


CONTENT REQUIREMENTS:
- Only retain new information and core knowledge
- No repetition or off-topic content
- Prioritize important events, metrics, and definitions
- Concise, clear language, easy to reference later
        """,

        "extractive": """
Extract main ideas, most important sentences, definitions, and notable concepts from text. Present as bulleted list.

FORMAT REQUIREMENTS:
- Each key point on one line starting with dash "-"
- Use ### to create group headers for different logical groups
- Keep original critical sentences from text
- Do not add information from outside the source text
- Arrange by logical order or importance level
        """
    }
}


MINDMAP_PROMPTS = {
    "vietnamese": """
    Bạn là một chuyên gia tạo sơ đồ tư duy. Phân tích văn bản và tạo cấu trúc mindmap 
    phân cấp với các chủ đề chính và chủ đề con.
    
    Hướng dẫn:
    - Xác định chủ đề chính làm gốc của sơ đồ
    - Phân chia thành các nhánh con theo thứ tự quan trọng
    - Mỗi nhánh có thể có các nhánh con nhỏ hơn
    - Sử dụng từ ngữ ngắn gọn, rõ ràng cho mỗi node
    - Đảm bảo cấu trúc hợp lý và dễ hiểu
    - Tối đa 4 cấp độ sâu
    - Sử dụng định dạng JSON chính xác
    
    Định dạng JSON yêu cầu:
    {
        "id": "root",
        "text": "Chủ đề chính",
        "children": [
            {
                "id": "node1",
                "text": "Chủ đề con 1",
                "children": [
                    {
                        "id": "node1_1",
                        "text": "Chi tiết 1",
                        "children": []
                    }
                ]
            }
        ]
    }
    """,
    
    "english": """
    You are an expert mindmap creator. Analyze the text and create a hierarchical 
    mindmap structure with main topics and subtopics.
    
    Guidelines:
    - Identify the main topic as the root
    - Divide into branches in order of importance
    - Each branch can have smaller sub-branches
    - Use concise, clear wording for each node
    - Ensure logical and understandable structure
    - Maximum 4 levels deep
    - Use exact JSON format
    
    Required JSON format:
    {
        "id": "root",
        "text": "Main Topic",
        "children": [
            {
                "id": "node1",
                "text": "Subtopic 1",
                "children": [
                    {
                        "id": "node1_1",
                        "text": "Detail 1",
                        "children": []
                    }
                ]
            }
        ]
    }
    """
}

MARKDOWN_MINDMAP_PROMPTS = {
    "vietnamese": """
    You are an expert mindmap creator. Analyze the text and create a mindmap structure
    in markdown format compatible with markmap.
    
    Guidelines:
    - Use # for main topic
    - Use ## for level 1 branches
    - Use ### for level 2 branches
    - Use #### for level 3 branches
    - Each line is a node in the mindmap
    - Use indentation to show hierarchical structure
    - Maximum 4 levels deep
    - Keep content concise and clear for each node
    
    Required markdown format:
    # Main Topic
    ## Main Branch 1
    ### Sub-branch 1.1
    #### Detail 1.1.1
    ### Sub-branch 1.2
    ## Main Branch 2
    ### Sub-branch 2.1
    
    NOTES:
    - Each node on a separate line
    - Do not use bullet points (-, *, +)
    - Only use # symbols to indicate levels
    - Do not add other special characters
    - CRITICAL: ALWAYS RETURN RESULT IN VIETNAMESE
    """,
    
    "english": """
    You are an expert mindmap creator. Analyze the text and create a mindmap structure
    in markdown format compatible with markmap.
    
    Guidelines:
    - Use # for main topic
    - Use ## for level 1 branches
    - Use ### for level 2 branches
    - Use #### for level 3 branches
    - Each line is a node in the mindmap
    - Use indentation to show hierarchical structure
    - Maximum 4 levels deep
    - Keep content concise and clear for each node
    
    Required markdown format:
    # Main Topic
    ## Main Branch 1
    ### Sub-branch 1.1
    #### Detail 1.1.1
    ### Sub-branch 1.2
    ## Main Branch 2
    ### Sub-branch 2.1
    
    NOTES:
    - Each node on a separate line
    - Do not use bullet points (-, *, +)
    - Only use # symbols to indicate levels
    - Do not add other special characters
    - CRITICAL: ALWAYS RETURN RESULT IN ENGLISH
    """
}

# Additional prompts for different use cases
SPECIALIZED_PROMPTS = {
    "technical_summary": {
        "vietnamese": """
        Tóm tắt tài liệu kỹ thuật, tập trung vào:
        - Các khái niệm kỹ thuật quan trọng
        - Quy trình và phương pháp
        - Kết quả và kết luận
        - Ứng dụng thực tiễn
        """,
        "english": """
        Summarize technical document, focusing on:
        - Important technical concepts
        - Processes and methodologies
        - Results and conclusions
        - Practical applications
        """
    },
    "academic_summary": {
        "vietnamese": """
        Tóm tắt tài liệu học thuật, bao gồm:
        - Mục tiêu nghiên cứu
        - Phương pháp luận
        - Kết quả chính
        - Đóng góp khoa học
        """,
        "english": """
        Summarize academic document, including:
        - Research objectives
        - Methodology
        - Key results
        - Scientific contributions
        """
    }
}

def get_prompt(prompt_type: str, language: str = "vietnamese", subtype: str = None, **kwargs) -> str:
    """
    Get prompt template by type and language with dynamic parameters.
    
    Args:
        prompt_type: Type of prompt (summary, mindmap, etc.)
        language: Language code (vietnamese, english)
        subtype: Subtype of prompt (abstractive, extractive, etc.)
        **kwargs: Additional parameters for template substitution
        
    Returns:
        Prompt string
    """
    if prompt_type == "summary":
        if subtype and subtype in SUMMARY_PROMPTS[language]:
            prompt = SUMMARY_PROMPTS[language][subtype]
        else:
            prompt = SUMMARY_PROMPTS[language]["abstractive"]
        
        # Replace dynamic parameters
        max_length = kwargs.get("max_length", 200)
        return prompt.format(max_length=max_length)
        
    elif prompt_type == "mindmap":
        return MINDMAP_PROMPTS[language]
    elif prompt_type == "markdown_mindmap":
        return MARKDOWN_MINDMAP_PROMPTS[language]
    elif prompt_type in SPECIALIZED_PROMPTS:
        if subtype and subtype in SPECIALIZED_PROMPTS[prompt_type][language]:
            return SPECIALIZED_PROMPTS[prompt_type][language][subtype]
        return SPECIALIZED_PROMPTS[prompt_type][language]
    else:
        return "Please process the provided text."