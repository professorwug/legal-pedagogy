"""
LLM model interface wrappers for pedadog.

This module provides standardized interfaces for different LLM providers
to work with the thermometer and belief measurement systems.
"""

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from anthropic import Anthropic


class BaseLLM(ABC):
    """Base class for LLM model wrappers."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
    
    @abstractmethod
    def prompt(self, text: str) -> str:
        """
        Send a prompt to the model and return the response.
        
        Args:
            text: Input prompt text
            
        Returns:
            Model response as string
        """
        pass


class AnthropicModel(BaseLLM):
    """Wrapper for Anthropic Claude models."""
    
    def __init__(
        self,
        name: str = "claude-3-haiku-20240307",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        # Initialize Anthropic client
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided or set as environment variable")
        
        self.client = Anthropic(api_key=self.api_key)
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def prompt(self, text: str) -> str:
        """Send prompt to Claude and return response."""
        try:
            response = self.client.messages.create(
                model=self.name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": text}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            raise RuntimeError(f"Error calling Anthropic API: {str(e)}")


class MockLLM(BaseLLM):
    """Mock LLM for testing purposes."""
    
    def __init__(
        self,
        name: str = "mock-llm",
        response_pattern: str = "0.5",
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.response_pattern = response_pattern
        self.call_count = 0
    
    def prompt(self, text: str) -> str:
        """Return mock response."""
        self.call_count += 1
        
        # Can provide different responses based on call count or input
        if "judge" in text.lower():
            responses = ["0.7", "0.8", "0.6", "0.9", "0.5"]
        elif "appellant" in text.lower():
            responses = ["0.3", "0.4", "0.2", "0.5", "0.1"]
        else:
            responses = ["0.5", "0.6", "0.4", "0.7", "0.3"]
        
        return responses[self.call_count % len(responses)]


# Factory functions for easy model creation

def create_anthropic_model(
    model_name: str = "claude-3-haiku-20240307",
    api_key: Optional[str] = None,
    **kwargs
) -> AnthropicModel:
    """Create an Anthropic model instance."""
    return AnthropicModel(
        name=model_name,
        api_key=api_key,
        **kwargs
    )


def create_mock_model(
    name: str = "mock-llm",
    **kwargs
) -> MockLLM:
    """Create a mock model instance for testing."""
    return MockLLM(name=name, **kwargs)


def create_model_from_config(config: Dict[str, Any]) -> BaseLLM:
    """
    Create a model instance from configuration dictionary.
    
    Args:
        config: Configuration dict with 'provider' and model parameters
        
    Returns:
        Configured model instance
    """
    provider = config.get("provider", "anthropic").lower()
    
    if provider == "anthropic":
        return create_anthropic_model(
            model_name=config.get("model_name", "claude-3-haiku-20240307"),
            api_key=config.get("api_key"),
            max_tokens=config.get("max_tokens", 1024),
            temperature=config.get("temperature", 0.7)
        )
    elif provider == "mock":
        return create_mock_model(
            name=config.get("name", "mock-llm"),
            response_pattern=config.get("response_pattern", "0.5")
        )
    else:
        raise ValueError(f"Unsupported model provider: {provider}")


# Convenience function to create default models
def get_default_model() -> BaseLLM:
    """Get the default model based on environment."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return create_anthropic_model()
    else:
        print("Warning: No ANTHROPIC_API_KEY found, using mock model")
        return create_mock_model()


# Export commonly used models
DEFAULT_LLM = None  # Will be set by application configuration


def set_default_llm(model: BaseLLM) -> None:
    """Set the global default LLM."""
    global DEFAULT_LLM
    DEFAULT_LLM = model
    
    # Also set it in generate_belief_vector module
    from . import generate_belief_vector
    generate_belief_vector.DEFAULT_LLM = model