"""
Base class for LLM provider implementations.
Supports both sync and async methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator, AsyncIterator
import logging
import asyncio

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    All LLM providers must implement:
    - generate(): Generate a complete response
    - stream(): Stream response token by token (optional)
    """

    @abstractmethod
    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            user_prompt: The user's prompt, including any context.
            system_prompt: System prompt to guide LLM behavior
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated response as string
        """
        pass

    def stream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream response token by token (optional).

        Args:
            Same as generate()

        Yields:
            Response tokens as they are generated
        """
        # Default implementation: yield complete response at once
        response = self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        yield response

    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt for RAG.

        Returns:
            Default system prompt
        """
        return """You are a helpful AI assistant. Answer the user's question based on the provided context. 
If the context doesn't contain relevant information, say so. Be concise and accurate.

Guidelines:
- Use information from the context to answer
- If unsure, acknowledge uncertainty
- Cite sources when possible
- Be concise but comprehensive"""

    def _get_default_vietnamese_system_prompt(self) -> str:
        """
        Get default Vietnamese system prompt.

        Returns:
            Vietnamese system prompt
        """
        return """Bạn là một trợ lý AI hữu ích. Trả lời câu hỏi của người dùng dựa trên ngữ cảnh được cung cấp.
Nếu ngữ cảnh không chứa thông tin liên quan, hãy nói rõ. Hãy ngắn gọn và chính xác.

Hướng dẫn:
- Sử dụng thông tin từ ngữ cảnh để trả lời
- Nếu không chắc chắn, hãy thừa nhận điều đó
- Trích dẫn nguồn khi có thể
- Ngắn gọn nhưng đầy đủ"""

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.

        Returns:
            Dictionary with model information
        """
        pass

    # ==================== ASYNC METHODS ====================

    async def agenerate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Async version of generate.
        Default implementation wraps sync generate in thread executor.
        
        Override this method for native async implementation.

        Args:
            Same as generate()

        Returns:
            Generated response as string
        """
        return await asyncio.to_thread(
            self.generate,
            user_prompt,
            system_prompt,
            temperature,
            max_tokens,
            **kwargs
        )

    async def astream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Async stream response token by token.
        Default implementation wraps sync stream.
        
        Override this method for native async streaming.

        Args:
            Same as generate()

        Yields:
            Response tokens as they are generated
        """
        # Default: yield from sync stream in thread
        for token in self.stream(
            user_prompt, system_prompt, temperature, max_tokens, **kwargs
        ):
            yield token