"""
LLM model interface wrappers for pedadog using Princeton AI Sandbox.

This module provides standardized interfaces for different LLM providers
to work with the thermometer and belief measurement systems.
"""

import os
import time
import random
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Princeton AI Sandbox configuration
SANDBOX_API_KEY = os.environ.get('AI_SANDBOX_KEY')
SANDBOX_ENDPOINT = "https://api-ai-sandbox.princeton.edu/"
SANDBOX_API_VERSION = "2025-03-01-preview"

# Available models in the AI Sandbox
AVAILABLE_MODELS = [
    "o3-mini",
    "gpt-4o-mini", 
    "gpt-4o",
    "gpt-35-turbo-16k",
    "Meta-Llama-3-1-70B-Instruct-htzs",
    "Meta-Llama-3-1-8B-Instruct-nwxcg",
    "Mistral-small-zgjes"
]

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class AISandboxModel(BaseLLM):
    """Wrapper for Princeton AI Sandbox models."""
    
    def __init__(
        self,
        name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system_prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.5,
        max_tokens: int = 1024,
        max_retries: int = 5,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        # Validate model name
        if name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {name} not available. Choose from: {AVAILABLE_MODELS}")
        
        # Initialize AI Sandbox client
        self.api_key = api_key or SANDBOX_API_KEY
        if not self.api_key:
            raise ValueError("AI_SANDBOX_KEY must be provided or set as environment variable")
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=SANDBOX_ENDPOINT,
            api_version=SANDBOX_API_VERSION
        )
        
        # Model parameters
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_retries = max_retries
    
    def prompt(self, text: str) -> str:
        """Send prompt to AI Sandbox and return response with retry logic."""
        base_delay = 1  # Base delay in seconds
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text},
                    ]
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                # Calculate backoff delay with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts failed. Last error: {str(e)}")
                    raise RuntimeError(f"Failed to get response after {self.max_retries} attempts") from e


class O3MiniModel(BaseLLM):
    """Specialized wrapper for O3-Mini model with optimized settings."""
    
    def __init__(
        self,
        name: str = "o3-mini",
        api_key: Optional[str] = None,
        system_prompt: str = "",
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        # Initialize AI Sandbox client
        self.api_key = api_key or SANDBOX_API_KEY
        if not self.api_key:
            raise ValueError("AI_SANDBOX_KEY must be provided or set as environment variable")
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=SANDBOX_ENDPOINT,
            api_version=SANDBOX_API_VERSION
        )
        
        self.system_prompt = system_prompt
        self.max_retries = 5
    
    def prompt(self, text: str) -> str:
        """Send prompt to O3-Mini with specialized retry logic."""
        base_delay = 1
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text},
                    ]
                )
                return response.choices[0].message.content
                
            except Exception as e:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                
                if attempt < self.max_retries - 1:
                    logger.warning(f"O3-Mini attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"O3-Mini failed after {self.max_retries} attempts: {str(e)}")
                    raise RuntimeError(f"O3-Mini failed after {self.max_retries} attempts") from e


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

def create_ai_sandbox_model(
    model_name: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    system_prompt: str = "",
    **kwargs
) -> AISandboxModel:
    """Create an AI Sandbox model instance."""
    return AISandboxModel(
        name=model_name,
        api_key=api_key,
        system_prompt=system_prompt,
        **kwargs
    )


def create_o3_mini_model(
    api_key: Optional[str] = None,
    system_prompt: str = "",
    **kwargs
) -> O3MiniModel:
    """Create an O3-Mini model instance."""
    return O3MiniModel(
        api_key=api_key,
        system_prompt=system_prompt,
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
    provider = config.get("provider", "ai_sandbox").lower()
    
    if provider == "ai_sandbox":
        return create_ai_sandbox_model(
            model_name=config.get("model_name", "gpt-4o-mini"),
            api_key=config.get("api_key"),
            system_prompt=config.get("system_prompt", ""),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.5),
            max_tokens=config.get("max_tokens", 1024)
        )
    elif provider == "o3_mini":
        return create_o3_mini_model(
            api_key=config.get("api_key"),
            system_prompt=config.get("system_prompt", "")
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
    if os.getenv("AI_SANDBOX_KEY"):
        return create_ai_sandbox_model()
    else:
        logger.warning("No AI_SANDBOX_KEY found, using mock model")
        return create_mock_model()


# Legacy function for backward compatibility with original sandbox API
def sandbox_llm(
    prompt: str, 
    system_prompt: str = "You are a helpful assistant.", 
    temperature: float = 0.7, 
    top_p: float = 0.5,  
    max_tokens: int = 4096, 
    model_to_be_used: str = 'gpt-4o'
) -> str:
    """
    Legacy function compatible with original ai_sandbox.py interface.
    
    This maintains backward compatibility while using the new model structure.
    """
    model = create_ai_sandbox_model(
        model_name=model_to_be_used,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    return model.prompt(prompt)


def prompt_o3_mini(
    prompt: str, 
    system_prompt: str = "", 
    model_to_be_used: str = "o3-mini"
) -> str:
    """
    Legacy function for O3-Mini compatible with original ai_sandbox.py interface.
    """
    model = create_o3_mini_model(system_prompt=system_prompt)
    return model.prompt(prompt)


# Export commonly used models
DEFAULT_LLM = None  # Will be set by application configuration


def set_default_llm(model: BaseLLM) -> None:
    """Set the global default LLM."""
    global DEFAULT_LLM
    DEFAULT_LLM = model
    
    # Also set it in generate_belief_vector module
    try:
        from . import generate_belief_vector
        generate_belief_vector.DEFAULT_LLM = model
    except ImportError:
        # Module may not be loaded yet
        pass


# Initialize default model if API key is available
if SANDBOX_API_KEY:
    try:
        DEFAULT_LLM = get_default_model()
    except Exception as e:
        logger.warning(f"Could not initialize default model: {e}")
        DEFAULT_LLM = create_mock_model()
else:
    logger.info("AI_SANDBOX_KEY not found, DEFAULT_LLM will use mock model")
    DEFAULT_LLM = create_mock_model()