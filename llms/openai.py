"""
OpenAI LLM provider implementation.
Supports both sync and async with native async client.
"""

from typing import Optional, Dict, Any, Iterator, AsyncIterator
from .base import BaseLLM
import logging

logger = logging.getLogger(__name__)

class OpenAILLM(BaseLLM):
    """
    LLM provider for OpenAI comparative API.

    Features:
    Advantages:
    - ✅ Fast API
    - ✅ Streaming support
    - ✅ Function calling support
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: Optional[int] = None,
    ):
        """
        Initialize OpenAI LLM provider.

        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
            base_url: Custom API base URL (for Azure OpenAI, etc.)
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens to generate
        """
        try:
            from openai import OpenAI, AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai is required for OpenAILLM. "
                "Install it with: pip install openai"
            )

        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

        # Initialize OpenAI clients (both sync and async)
        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if organization:
            client_kwargs["organization"] = organization
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.async_client = AsyncOpenAI(**client_kwargs)

        logger.info(f"✓ Initialized OpenAI clients (sync + async) with model '{model_name}'")

    def _inject_disable_thinking(self, kwargs):
        # Always inject enable_thinking=False if not explicitly set
        kwargs = dict(kwargs)  # defensive copy
        extra_body = kwargs.get("extra_body", {})
        chat_kwargs = extra_body.get("chat_template_kwargs", {})
        # Only inject if user didn't manually change
        if "enable_thinking" not in chat_kwargs:
            chat_kwargs["enable_thinking"] = False
        extra_body["chat_template_kwargs"] = chat_kwargs
        kwargs["extra_body"] = extra_body
        return kwargs

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response using OpenAI API.

        Args:
            user_prompt: The user's prompt, including any context.
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional OpenAI parameters (top_p, presence_penalty, etc.)

        Returns:
            Generated response
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        # Build messages
        messages = self._build_messages(user_prompt, system_prompt)
        kwargs = self._inject_disable_thinking(kwargs)

        try:
            logger.info(f"Generating response with OpenAI model '{self.model_name}'")
            # Call OpenAI API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            answer = completion.choices[0].message.content

            logger.info(
                f"Generated {len(answer)} characters. "
                f"Tokens: {completion.usage.total_tokens}"
            )

            return answer

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    def stream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream response from OpenAI API.

        Args:
            Same as generate()

        Yields:
            Response tokens as they are generated
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        # Build messages
        messages = self._build_messages(user_prompt, system_prompt)
        kwargs = self._inject_disable_thinking(kwargs)

        try:
            logger.info(f"Streaming response with OpenAI model '{self.model_name}'")
            # Stream from OpenAI
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error streaming from OpenAI: {e}")
            raise

    def _build_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None
    ) -> list:
        """
        Build OpenAI messages format.

        Args:
            user_prompt: The user's prompt, including any context.
            system_prompt: System instructions

        Returns:
            List of message dicts
        """
        messages = []

        # System message
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        messages.append({
            "role": "system",
            "content": system_prompt
        })

        messages.append({
            "role": "user",
            "content": user_prompt
        })

        return messages

    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        return {
            "provider": "openai",
            "model_name": self.model_name,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
        }

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate).
        
        For accurate counting, use tiktoken library.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count
        """
        try:
            import tiktoken

            if "gpt-4" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.model_name:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
            
        except ImportError:
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text) // 4

    # ==================== ASYNC METHODS ====================

    async def agenerate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Async generate response using OpenAI API (native async client).

        Args:
            Same as generate()

        Returns:
            Generated response
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        # Build messages
        messages = self._build_messages(user_prompt, system_prompt)
        kwargs = self._inject_disable_thinking(kwargs)

        try:
            logger.debug(f"Async generating response with OpenAI model '{self.model_name}'")
            # Call OpenAI API asynchronously
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            answer = completion.choices[0].message.content

            logger.debug(
                f"Generated {len(answer)} characters. "
                f"Tokens: {completion.usage.total_tokens}"
            )

            return answer

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise

    async def astream(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Async stream response from OpenAI API (native async streaming).

        Args:
            Same as generate()

        Yields:
            Response tokens as they are generated
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        # Build messages
        messages = self._build_messages(user_prompt, system_prompt)
        kwargs = self._inject_disable_thinking(kwargs)

        try:
            logger.info(f"Async streaming response with OpenAI model '{self.model_name}'")
            # Stream from OpenAI asynchronously
            stream = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error streaming from OpenAI: {e}")
            raise
