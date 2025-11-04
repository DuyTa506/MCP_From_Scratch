"""
Ollama LLM provider implementation.
Supports both sync and async with native AsyncClient.
"""

from typing import Optional, Dict, Any, Iterator, AsyncIterator
from .base import BaseLLM
import logging

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """
    LLM provider using Ollama for local inference.
    
    Ollama provides easy access to local LLMs like:
    - llama3, llama3.1
    - mistral, mixtral
    - gemma, gemma2
    - qwen2, phi3
    - And many more...
    """

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        default_temperature: float = 0.7,
        default_max_tokens: Optional[int] = None,
    ):
        """
        Initialize Ollama LLM provider.

        Args:
            model_name: Ollama model name (e.g., 'llama3', 'mistral', 'gemma2')
            base_url: Ollama API base URL
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens to generate
        """
        try:
            import requests
            from ollama import AsyncClient
        except ImportError:
            raise ImportError(
                "requests and ollama are required for OllamaLLM. "
                "Install them with: pip install requests ollama"
            )

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.requests = requests
        
        # Initialize both sync and async clients
        self.async_client = AsyncClient(host=base_url)

        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"✓ Connected to Ollama at {self.base_url}")
        except Exception as e:
            logger.warning(
                f"Could not connect to Ollama: {e}. "
                f"Make sure Ollama is running with: ollama serve"
            )

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response using Ollama.

        Args:
            user_prompt: The user's prompt, including any context.
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional Ollama parameters

        Returns:
            Generated response
        """
        temperature = temperature if temperature is not None else self.default_temperature
        
        # Prepare Ollama request
        payload = {
            "model": self.model_name,
            "prompt": user_prompt,
            "system": system_prompt or self._get_default_vietnamese_system_prompt(),
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }

        # Add max_tokens if specified
        if max_tokens or self.default_max_tokens:
            payload["options"]["num_predict"] = max_tokens or self.default_max_tokens

        # Add any additional options
        if kwargs:
            payload["options"].update(kwargs)

        try:
            logger.info(f"Generating response with Ollama model '{self.model_name}'")
            
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "")
            
            logger.info(f"Generated {len(answer)} characters")
            return answer

        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
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
        Stream response token by token from Ollama.

        Args:
            Same as generate()

        Yields:
            Response tokens as they are generated
        """
        temperature = temperature if temperature is not None else self.default_temperature
        
        # Prepare request with streaming
        payload = {
            "model": self.model_name,
            "prompt": user_prompt,
            "system": system_prompt or self._get_default_vietnamese_system_prompt(),
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }

        if max_tokens or self.default_max_tokens:
            payload["options"]["num_predict"] = max_tokens or self.default_max_tokens

        if kwargs:
            payload["options"].update(kwargs)

        try:
            logger.info(f"Streaming response with Ollama model '{self.model_name}'")
            
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            # Stream responses
            import json
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                    
                    # Check if done
                    if chunk.get("done", False):
                        break

        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        try:
            response = self.requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=5,
            )
            response.raise_for_status()
            model_info = response.json()
            
            return {
                "provider": "ollama",
                "model_name": self.model_name,
                "base_url": self.base_url,
                "default_temperature": self.default_temperature,
                "model_details": model_info,
            }
        except Exception as e:
            logger.warning(f"Could not fetch model info: {e}")
            return {
                "provider": "ollama",
                "model_name": self.model_name,
                "base_url": self.base_url,
                "default_temperature": self.default_temperature,
            }

    def list_available_models(self) -> list:
        """
        List all models available in Ollama.

        Returns:
            List of model names
        """
        try:
            response = self.requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            
            models = [model["name"] for model in data.get("models", [])]
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    # ==================== ASYNC METHODS (Native AsyncClient) ====================

    async def agenerate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Async generate response using Ollama AsyncClient (native async).

        Args:
            Same as generate()

        Returns:
            Generated response
        """
        temperature = temperature if temperature is not None else self.default_temperature
        
        # Prepare options
        options = {
            "temperature": temperature,
        }
        
        if max_tokens or self.default_max_tokens:
            options["num_predict"] = max_tokens or self.default_max_tokens
        
        options.update(kwargs)

        try:
            logger.info(f"Async generating response with Ollama model '{self.model_name}'")
            
            # Use Ollama AsyncClient (native async, no thread executor!)
            response = await self.async_client.generate(
                model=self.model_name,
                prompt=user_prompt,
                system=system_prompt or self._get_default_vietnamese_system_prompt(),
                options=options,
                stream=False,
            )

            answer = response.get("response", "")
            
            logger.info(f"Generated {len(answer)} characters")
            return answer

        except Exception as e:
            logger.error(f"Error calling Ollama async API: {e}")
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
        Async stream response using Ollama AsyncClient (native async streaming).

        Args:
            Same as generate()

        Yields:
            Response tokens as they are generated
        """
        temperature = temperature if temperature is not None else self.default_temperature
        
        # Prepare options
        options = {
            "temperature": temperature,
        }
        
        if max_tokens or self.default_max_tokens:
            options["num_predict"] = max_tokens or self.default_max_tokens
        
        options.update(kwargs)

        try:
            logger.info(f"Async streaming response with Ollama model '{self.model_name}'")
            
            # Use Ollama AsyncClient streaming (native async!)
            stream = await self.async_client.generate(
                model=self.model_name,
                prompt=user_prompt,
                system=system_prompt or self._get_default_vietnamese_system_prompt(),
                options=options,
                stream=True,
            )

            # Stream responses
            async for chunk in stream:
                if "response" in chunk:
                    yield chunk["response"]

        except Exception as e:
            logger.error(f"Error async streaming from Ollama: {e}")
            raise