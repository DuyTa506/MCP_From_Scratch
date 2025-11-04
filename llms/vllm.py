"""
vLLM provider implementation for high-performance inference.
Supports both sync and async with AsyncLLMEngine.
"""

from typing import Optional, Dict, Any, Iterator, AsyncIterator
from .base import BaseLLM
import logging
import uuid

logger = logging.getLogger(__name__)


class vLLM(BaseLLM):
    """
    LLM provider using vLLM inference engine.
    
    vLLM provides high-throughput serving for LLMs with:
    - PagedAttention for efficient memory usage
    - Continuous batching for optimal throughput
    - Support for many popular models
    
    Advantages:
    - ✅ Highest throughput
    - ✅ Production-ready
    - ✅ Efficient GPU utilization
    - ✅ Multi-GPU support
    - ✅ Streaming support
    
    Best for:
    - Production deployments
    - High QPS requirements
    - Batch processing
    - Local inference with performance
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-8B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 512,
        trust_remote_code: bool = False,
    ):
        """
        Initialize vLLM provider.

        Args:
            model_name: HuggingFace model identifier
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory utilization (0-1)
            max_model_len: Maximum model context length
            default_temperature: Default sampling temperature
            default_max_tokens: Default max tokens to generate
            trust_remote_code: Whether to trust remote code
        """
        try:
            from vllm import LLM, SamplingParams, AsyncLLMEngine, AsyncEngineArgs
        except ImportError:
            raise ImportError(
                "vllm is required for vLLM provider. "
                "Install it with: pip install vllm"
            )

        self.model_name = model_name
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.SamplingParams = SamplingParams
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code

        logger.info(f"Loading vLLM model: {model_name}")
        logger.info(f"  - Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"  - GPU memory utilization: {gpu_memory_utilization}")

        try:
            # Initialize vLLM sync engine
            self.llm = LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=trust_remote_code,
            )

            # Store AsyncEngineArgs for lazy async engine initialization
            self.AsyncEngineArgs = AsyncEngineArgs
            self._async_engine = None

            logger.info(f"✓ vLLM model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            raise

    def generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate response using vLLM.

        Args:
            user_prompt: The user's prompt, including any context.
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional vLLM sampling parameters

        Returns:
            Generated response
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        # Create sampling params
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        try:
            logger.info(f"Generating response with vLLM model '{self.model_name}'")
            
            # Generate with vLLM
            outputs = self.llm.generate([user_prompt], sampling_params)
            
            # Extract response
            answer = outputs[0].outputs[0].text
            
            logger.info(f"Generated {len(answer)} characters")
            return answer

        except Exception as e:
            logger.error(f"Error generating with vLLM: {e}")
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
        Stream response from vLLM.
        
        Note: vLLM doesn't support native streaming in offline mode.
        This method simulates streaming by yielding the complete response.

        Args:
            Same as generate()

        Yields:
            Complete response (not true streaming)
        """
        # vLLM offline inference doesn't support true streaming
        # Yield complete response
        response = self.generate(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Simulate streaming by yielding in chunks
        chunk_size = 50
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]

    def batch_generate(
        self,
        user_prompts: list,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> list:
        """
        Generate responses for multiple queries in batch.
        This is where vLLM really shines!

        Args:
            user_prompts: List of user prompts
            system_prompt: System prompt (same for all)
            temperature: Sampling temperature
            max_tokens: Max tokens per response
            **kwargs: Additional sampling parameters

        Returns:
            List of generated responses
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        # Create sampling params
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        try:
            logger.info(
                f"Batch generating for {len(user_prompts)} queries with vLLM"
            )
            
            # Batch generation with vLLM (very efficient!)
            outputs = self.llm.generate(user_prompts, sampling_params)
            
            # Extract responses
            answers = [output.outputs[0].text for output in outputs]
            
            logger.info(f"Generated {len(answers)} responses")
            return answers

        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            raise

    async def _get_async_engine(self):
        """
        Lazy initialization of AsyncLLMEngine.
        Only creates engine when first async method is called.
        """
        if self._async_engine is None:
            logger.info("Initializing AsyncLLMEngine...")
            
            from vllm import AsyncLLMEngine
            
            engine_args = self.AsyncEngineArgs(
                model=self.model_name,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                trust_remote_code=self.trust_remote_code,
            )
            
            self._async_engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("✓ AsyncLLMEngine initialized")
        
        return self._async_engine

    async def agenerate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Async generate response using vLLM AsyncLLMEngine (native async).

        Args:
            Same as generate()

        Returns:
            Generated response
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        # Create sampling params
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        try:
            logger.info(f"Async generating response with vLLM model '{self.model_name}'")
            
            # Get async engine
            engine = await self._get_async_engine()
            
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            
            # Generate with AsyncLLMEngine
            results_generator = engine.generate(
                prompt=user_prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            # Get final output
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            
            answer = final_output.outputs[0].text if final_output else ""
            
            logger.info(f"Generated {len(answer)} characters")
            return answer

        except Exception as e:
            logger.error(f"Error generating with vLLM async: {e}")
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
        Async stream response using vLLM AsyncLLMEngine (native async streaming).

        Args:
            Same as generate()

        Yields:
            Response tokens as they are generated
        """
        temperature = temperature if temperature is not None else self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens

        # Create sampling params
        sampling_params = self.SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        try:
            logger.info(f"Async streaming response with vLLM model '{self.model_name}'")
            
            # Get async engine
            engine = await self._get_async_engine()
            
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            
            # Stream with AsyncLLMEngine
            results_generator = engine.generate(
                prompt=user_prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            # Stream delta tokens
            async for request_output in results_generator:
                # Yield only the new generated text (delta)
                yield request_output.outputs[0].text

        except Exception as e:
            logger.error(f"Error streaming with vLLM async: {e}")
            raise

    async def abort_request(self, request_id: str):
        """
        Abort an ongoing async request.
        Useful when client disconnects or request is cancelled.

        Args:
            request_id: The request ID to abort
        """
        if self._async_engine:
            await self._async_engine.abort(request_id)
            logger.info(f"Aborted request: {request_id}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM model information."""
        return {
            "provider": "vllm",
            "model_name": self.model_name,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "tensor_parallel_size": self.tensor_parallel_size,
        }