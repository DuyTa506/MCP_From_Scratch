"""
LLM Factory for centralized provider management.
Supports dynamic provider selection and configuration.
"""

from typing import Optional, Dict, Any
from .base import BaseLLM
from .openai import OpenAILLM
from .ollama import OllamaLLM
from .vllm import vLLM
import logging

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Factory class for creating LLM instances based on configuration.
    
    Supports:
    - Dynamic provider selection
    - Configuration-based instantiation
    - Provider validation
    - Fallback mechanisms
    """
    
    # Registry of available providers
    _providers = {
        "openai": OpenAILLM,
        "ollama": OllamaLLM,
        "vllm": vLLM,
    }
    
    @classmethod
    def create_llm(
        self,
        provider: str,
        **kwargs
    ) -> BaseLLM:
        """
        Create LLM instance based on provider name.
        
        Args:
            provider: Provider name ("openai", "ollama", "vllm")
            **kwargs: Provider-specific configuration
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If provider is not supported
            ImportError: If required dependencies are missing
        """
        if provider is None:
            logger.warning("Provider is None, using default 'ollama'")
            provider = "ollama"
        else:
            provider = provider.lower()
        
        if provider not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Available providers: {available}"
            )
        
        try:
            llm_class = self._providers[provider]
            logger.info("Creating %s LLM with config: %s", provider, kwargs)
            
            # Create instance with provided configuration
            llm_instance = llm_class(**kwargs)
            
            logger.info("✓ %s LLM created successfully", provider)
            return llm_instance
            
        except ImportError as e:
            logger.error("Missing dependencies for %s: %s", provider, e)
            raise
        except Exception as e:
            logger.error("Failed to create %s LLM: %s", provider, e)
            raise
    
    @classmethod
    def create_from_config(
        self,
        config: Dict[str, Any]
    ) -> Optional[BaseLLM]:
        """
        Create LLM from configuration dictionary.
        
        Args:
            config: Configuration dict with provider and settings
            
        Returns:
            Configured LLM instance or None if disabled
            
        Example config:
        {
            "enabled": True,
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": "sk-...",
            "temperature": 0.7
        }
        """
        # Check if LLM is enabled
        if not config.get("enabled", True):
            logger.info("LLM disabled in configuration")
            return None
        
        provider = config.get("provider", "ollama")
        
        # Handle None provider - use default instead of returning None
        if provider is None:
            logger.warning("Provider is None in LLM config, using default 'ollama': %s", config)
            provider = "ollama"
        
        # Remove non-provider config keys
        llm_config = {k: v for k, v in config.items()
                     if k not in ["enabled", "provider"]}
        
        try:
            return self.create_llm(provider, **llm_config)
        except Exception as e:
            logger.warning("Failed to create LLM from config: %s", e)
            return None
    
    @classmethod
    def create_openai(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ) -> OpenAILLM:
        """Create OpenAI LLM with common parameters."""
        return self.create_llm(
            "openai",
            model_name=model_name,
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            **kwargs
        )
    
    @classmethod
    def create_ollama(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        **kwargs
    ) -> OllamaLLM:
        """Create Ollama LLM with common parameters."""
        return self.create_llm(
            "ollama",
            model_name=model_name,
            base_url=base_url,
            **kwargs
        )
    
    @classmethod
    def create_vllm(
        self,
        model_name: str = "meta-llama/Llama-3-8B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs
    ) -> vLLM:
        """Create vLLM instance with common parameters."""
        return self.create_llm(
            "vllm",
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs
        )
    
    @classmethod
    def get_available_providers(self) -> list:
        """Get list of available provider names."""
        return list(self._providers.keys())
    
    @classmethod
    def register_provider(
        self,
        name: str,
        llm_class: type
    ) -> None:
        """
        Register a new LLM provider.
        
        Args:
            name: Provider name
            llm_class: LLM class that inherits from BaseLLM
        """
        if not issubclass(llm_class, BaseLLM):
            raise ValueError("LLM class must inherit from BaseLLM")
        
        self._providers[name] = llm_class
        logger.info("Registered new LLM provider: %s", name)
    
    @classmethod
    def validate_provider(self, provider: str) -> bool:
        """
        Check if provider is available and dependencies are installed.
        
        Args:
            provider: Provider name to validate
            
        Returns:
            True if provider is available and ready
        """
        if provider not in self._providers:
            return False
        
        try:
            # Try to import the provider class
            llm_class = self._providers[provider]
            
            # For some providers, we can check dependencies
            if provider == "openai":
                import openai  # noqa: F401
            elif provider == "ollama":
                import ollama  # noqa: F401
            elif provider == "vllm":
                import vllm  # noqa: F401
                
            return True
            
        except ImportError:
            return False
        except Exception:
            return False


# Convenience functions for common use cases
def create_llm_from_settings(settings) -> Optional[BaseLLM]:
    """
    Create LLM from application settings.
    
    Args:
        settings: Application settings object
        
    Returns:
        Configured LLM instance or None
    """
    if not getattr(settings, 'enable_llm', True):
        return None
    
    provider = getattr(settings, 'llm_provider', 'ollama')
    
    try:
        if provider == "openai":
            return LLMFactory.create_openai(
                model_name=getattr(settings, 'openai_model', 'gpt-4o-mini'),
                api_key=getattr(settings, 'openai_api_key', None),
                base_url=getattr(settings, 'openai_base_url', None)
            )
        elif provider == "ollama":
            return LLMFactory.create_ollama(
                model_name=getattr(settings, 'ollama_model', 'llama3'),
                base_url=getattr(settings, 'ollama_base_url', 'http://localhost:11434')
            )
        elif provider == "vllm":
            return LLMFactory.create_vllm(
                model_name=getattr(settings, 'vllm_model', 'meta-llama/Llama-3-8B-Instruct')
            )
        else:
            logger.warning("Unknown LLM provider: %s", provider)
            return None
            
    except Exception as e:
        logger.warning("Failed to create LLM: %s", e)
        return None


def create_llm_with_fallback(
    primary_provider: str,
    fallback_provider: str = "ollama",
    **config
) -> Optional[BaseLLM]:
    """
    Create LLM with fallback mechanism.
    
    Args:
        primary_provider: Primary provider to try
        fallback_provider: Fallback provider if primary fails
        **config: Configuration parameters
        
    Returns:
        LLM instance or None if both fail
    """
    # Try primary provider
    try:
        if LLMFactory.validate_provider(primary_provider):
            return LLMFactory.create_llm(primary_provider, **config)
    except Exception as e:
        logger.warning("Primary provider %s failed: %s", primary_provider, e)
    
    # Try fallback provider
    try:
        if LLMFactory.validate_provider(fallback_provider):
            logger.info("Falling back to %s", fallback_provider)
            return LLMFactory.create_llm(fallback_provider, **config)
    except Exception as e:
        logger.error("Fallback provider %s also failed: %s", fallback_provider, e)
    
    return None
