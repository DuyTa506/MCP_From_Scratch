"""
Mindmap formatter module for converting between different output formats.
Handles JSON, Markdown, Mermaid, and HTML conversions.
"""

import json
import uuid
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class MindmapFormatter:
    """
    Handles conversion between different mindmap output formats.
    """
    
    def __init__(self):
        """Initialize formatter."""
        pass
    
    def json_to_mermaid(self, mindmap: Dict) -> str:
        """Convert mindmap JSON to Mermaid format."""
        def node_to_mermaid(node, level=0):
            if not node.get("children"):
                return f'{"  " * level}{node["text"]}'
            
            children_mermaid = []
            for child in node["children"]:
                children_mermaid.append(node_to_mermaid(child, level + 1))
            
            return f'{"  " * level}{node["text"]}\n' + '\n'.join(children_mermaid)
        
        return f"mindmap\n  {node_to_mermaid(mindmap)}"
    
    def json_to_markdown(self, mindmap: Dict) -> str:
        """Convert mindmap JSON to Markdown outline."""
        def node_to_markdown(node, level=0):
            prefix = "#" * (level + 1)
            result = f'{prefix} {node["text"]}\n'
            
            if node.get("children"):
                for child in node["children"]:
                    result += node_to_markdown(child, level + 1)
            
            return result
        
        return node_to_markdown(mindmap)
    
    def json_to_html(self, mindmap: Dict) -> str:
        """Convert mindmap JSON to HTML nested list format."""
        def node_to_html(node, level=0):
            indent = "  " * level
            if not node.get("children") or len(node["children"]) == 0:
                return f'{indent}<li>{node["text"]}</li>'
            
            children_html = ""
            for child in node["children"]:
                children_html += node_to_html(child, level + 2)
            
            return f'{indent}<li>{node["text"]}\n{indent}  <ul>\n{children_html}\n{indent}  </ul>\n{indent}</li>'
        
        return f'<ul>\n{node_to_html(mindmap)}\n</ul>'
    
    def markdown_to_json(self, markdown: str) -> Dict[str, Any]:
        """Convert markdown mindmap to JSON format."""
        if not isinstance(markdown, str):
            return {"id": "root", "text": "Error", "children": []}
        
        lines = markdown.split('\n')
        root = {"id": "root", "text": "Mindmap", "children": []}
        stack = [root]  # Stack to track parent nodes
        
        for line in lines:
            line = line.strip()
            if not line.startswith('#'):
                continue
            
            # Count the level
            level = 0
            for char in line:
                if char == '#':
                    level += 1
                else:
                    break
            
            # Extract text
            text = line[level:].strip()
            
            # Create node
            node = {
                "id": f"node_{uuid.uuid4().hex[:8]}",
                "text": text,
                "children": []
            }
            
            # Find parent (level-1 in stack)
            if level <= len(stack):
                parent = stack[level - 1]
                parent["children"].append(node)
                
                # Update stack
                if level == len(stack):
                    stack.append(node)
                else:
                    stack[level] = node
        
        return root
    
    def markdown_to_mermaid(self, markdown: str) -> str:
        """Convert markdown mindmap to Mermaid format."""
        if not isinstance(markdown, str):
            return "mindmap\n  Root"
        
        lines = markdown.split('\n')
        mermaid_lines = ["mindmap"]
        
        for line in lines:
            line = line.strip()
            if not line.startswith('#'):
                continue
            
            # Count the level
            level = 0
            for char in line:
                if char == '#':
                    level += 1
                else:
                    break
            
            # Extract text
            text = line[level:].strip()
            
            # Add to mermaid with proper indentation
            indent = "  " * (level - 1)
            mermaid_lines.append(f"{indent}{text}")
        
        return '\n'.join(mermaid_lines)
    
    def markdown_to_html(self, markdown: str) -> str:
        """Convert markdown mindmap to HTML nested list format."""
        if not isinstance(markdown, str):
            return "<ul><li>Error</li></ul>"
        
        lines = markdown.split('\n')
        root = {"text": "Mindmap", "children": [], "level": 0}
        stack = [{"node": root, "level": 0}]
        
        for line in lines:
            line = line.strip()
            if not line.startswith('#'):
                continue
            
            # Count the level
            level = 0
            for char in line:
                if char == '#':
                    level += 1
                else:
                    break
            
            # Extract text
            text = line[level:].strip()
            
            # Create node
            node = {"text": text, "children": [], "level": level}
            
            # Find parent
            while len(stack) > 1 and stack[-1]["level"] >= level:
                stack.pop()
            
            parent = stack[-1]["node"]
            parent["children"].append(node)
            stack.append({"node": node, "level": level})
        
        # Convert to HTML
        def node_to_html(node, level=0):
            indent = "  " * level
            if not node.get("children"):
                return f'{indent}<li>{node["text"]}</li>'
            
            children_html = ""
            for child in node["children"]:
                children_html += node_to_html(child, level + 2)
            
            return f'{indent}<li>{node["text"]}\n{indent}  <ul>\n{children_html}\n{indent}  </ul>\n{indent}</li>'
        
        return f'<ul>\n{node_to_html(root)}\n</ul>'
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with enhanced error handling."""
        try:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                
                # Try to parse JSON directly first
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to fix common issues
                    json_str = self._fix_common_json_issues(json_str)
                    return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
        
        # Return fallback if JSON extraction fails
        return {"error": "Could not extract JSON"}
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Remove trailing commas before closing brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix missing quotes around keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Simple approach: just ensure brackets are balanced
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        
        return json_str
    
    def extract_markdown_from_response(self, response: str) -> str:
        """Extract markdown content from LLM response."""
        # Find the first # and extract from there
        lines = response.split('\n')
        markdown_lines = []
        in_markdown = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                in_markdown = True
                markdown_lines.append(line)
            elif in_markdown and line:
                # Continue collecting non-empty lines after we found markdown
                markdown_lines.append(line)
            elif in_markdown and not line:
                # Stop at empty line after markdown content
                break
        
        return '\n'.join(markdown_lines) if markdown_lines else response.strip()
    
    def validate_and_fix_json(self, mindmap: Dict, max_nodes: int, max_depth: int) -> Dict[str, Any]:
        """Validate and fix JSON mindmap structure with enhanced validation."""
        if not isinstance(mindmap, dict):
            return self._create_fallback_json_mindmap("", "vietnamese")
        
        # Ensure required fields
        if "id" not in mindmap:
            mindmap["id"] = "root"
        if "text" not in mindmap:
            mindmap["text"] = "Mindmap"
        if "children" not in mindmap:
            mindmap["children"] = []
        
        # Validate children is a list
        if not isinstance(mindmap["children"], list):
            logger.warning("Children field is not a list, converting to empty list")
            mindmap["children"] = []
        
        # Remove duplicate nodes at same level
        mindmap["children"] = self._remove_duplicate_nodes(mindmap["children"])
        
        # Fix structure recursively
        return self._fix_node_structure(mindmap, max_nodes, max_depth, current_depth=0, node_count=1)
    
    def _remove_duplicate_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """Remove duplicate nodes based on text content."""
        seen_texts = set()
        unique_nodes = []
        
        for node in nodes:
            node_text = node.get("text", "").lower().strip()
            if node_text and node_text not in seen_texts:
                seen_texts.add(node_text)
                unique_nodes.append(node)
        
        return unique_nodes
    
    def _fix_node_structure(
        self, node: Dict, max_nodes: int, max_depth: int, current_depth: int, node_count: int
    ) -> Dict[str, Any]:
        """Recursively fix node structure and enforce constraints with enhanced validation."""
        # Check depth limit
        if current_depth >= max_depth:
            node["children"] = []
            return node
        
        # Ensure required fields
        if "id" not in node:
            node["id"] = f"node_{uuid.uuid4().hex[:8]}"
        if "text" not in node:
            node["text"] = "Untitled"
        if "children" not in node:
            node["children"] = []
        
        # Validate children is a list
        if not isinstance(node["children"], list):
            logger.warning(f"Children field in node {node['id']} is not a list, converting to empty list")
            node["children"] = []
        
        # Remove duplicate children
        node["children"] = self._remove_duplicate_nodes(node["children"])
        
        # Process children
        fixed_children = []
        for child in node["children"]:
            if node_count >= max_nodes:
                break  # Node limit reached
            
            if isinstance(child, dict):
                fixed_child = self._fix_node_structure(
                    child, max_nodes, max_depth, current_depth + 1, node_count + 1
                )
                fixed_children.append(fixed_child)
                node_count += 1
        
        node["children"] = fixed_children
        return node
    
    def validate_and_fix_markdown(self, markdown: str, max_nodes: int, max_depth: int, language: str) -> str:
        """Validate and fix markdown structure."""
        if not markdown:
            return self._create_fallback_markdown_mindmap("", language)
        
        lines = markdown.split('\n')
        fixed_lines = []
        node_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with # (markdown heading)
            if line.startswith('#'):
                # Count the level
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                
                # Limit depth
                if level > max_depth:
                    # Reduce to max_depth
                    line = '#' * max_depth + line[max_depth:]
                
                # Count nodes
                node_count += 1
                if node_count > max_nodes:
                    break
                    
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines) if fixed_lines else self._create_fallback_markdown_mindmap("", language)
    
    def _create_fallback_json_mindmap(self, text: str, language: str) -> Dict[str, Any]:
        """Create fallback JSON mindmap when generation fails."""
        if language == "vietnamese":
            root_text = "Mindmap"
            error_text = "Could not create detailed structure"
        else:
            root_text = "Mindmap"
            error_text = "Could not create detailed structure"
        
        return {
            "id": "root",
            "text": root_text,
            "children": [
                {
                    "id": "error",
                    "text": error_text,
                    "children": []
                }
            ]
        }
    
    def _create_fallback_markdown_mindmap(self, text: str, language: str) -> str:
        """Create fallback markdown mindmap when generation fails."""
        if language == "vietnamese":
            return """# Mindmap
## Could not create detailed structure
## Please try again
"""
        else:
            return """# Mindmap
## Could not create detailed structure
## Please try again
"""
    
    def count_nodes(self, mindmap: Dict) -> int:
        """Count total nodes in JSON mindmap."""
        if not isinstance(mindmap, dict) or "children" not in mindmap:
            return 0
        
        count = 1  # Count root
        for child in mindmap["children"]:
            count += self.count_nodes(child)
        
        return count
    
    def get_actual_depth(self, mindmap: Dict) -> int:
        """Get actual maximum depth of JSON mindmap."""
        if not isinstance(mindmap, dict) or "children" not in mindmap:
            return 0
        
        if not mindmap["children"]:
            return 1
        
        max_child_depth = 0
        for child in mindmap["children"]:
            child_depth = self.get_actual_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth
    
    def count_markdown_nodes(self, markdown: str) -> int:
        """Count total nodes in markdown mindmap."""
        if not isinstance(markdown, str):
            return 0
        
        lines = markdown.split('\n')
        count = 0
        for line in lines:
            if line.strip().startswith('#'):
                count += 1
        
        return count
    
    def get_markdown_depth(self, markdown: str) -> int:
        """Get actual maximum depth of markdown mindmap."""
        if not isinstance(markdown, str):
            return 0
        
        lines = markdown.split('\n')
        max_depth = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                level = 0
                for char in line:
                    if char == '#':
                        level += 1
                    else:
                        break
                max_depth = max(max_depth, level)
        
        return max_depth