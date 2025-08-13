"""
Simple model wrappers for the minimum viable experiment.

This module provides appellant and judge model instances with
appropriate prompting for the Supreme Court context.
"""

import yaml
from pathlib import Path
from pedadog.models import create_model_from_config, BaseLLM


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent.parent / "pedadog" / "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


class AppellantModel:
    """Wrapper for appellant (lawyer) model with specific prompting."""
    
    def __init__(self, base_model: BaseLLM, config: dict):
        self.base_model = base_model
        self.behavior_prompt = config['prompts']['appellant_behavior']
        self.name = f"appellant-{base_model.name}"
    
    def prompt(self, text: str) -> str:
        """Send prompt with appellant behavior context."""
        full_prompt = f"{self.behavior_prompt}\n\n{text}"
        return self.base_model.prompt(full_prompt)


class JudgeModel:
    """Wrapper for judge model with specific prompting."""
    
    def __init__(self, base_model: BaseLLM, config: dict):
        self.base_model = base_model
        self.behavior_prompt = config['prompts']['judge_behavior']
        self.name = f"judge-{base_model.name}"
    
    def prompt(self, text: str) -> str:
        """Send prompt with judge behavior context."""
        full_prompt = f"{self.behavior_prompt}\n\n{text}"
        return self.base_model.prompt(full_prompt)


def create_appellant_model(config: dict = None) -> AppellantModel:
    """
    Create an appellant model instance.
    
    Args:
        config: Configuration dictionary (loads from file if None)
        
    Returns:
        Configured appellant model
    """
    if config is None:
        config = load_config()
    
    # Use appellant-specific config if available, otherwise default
    model_config = config['llms'].get('appellant_model', config['llms']['default'])
    base_model = create_model_from_config(model_config)
    
    return AppellantModel(base_model, config)


def create_judge_model(config: dict = None) -> JudgeModel:
    """
    Create a judge model instance.
    
    Args:
        config: Configuration dictionary (loads from file if None)
        
    Returns:
        Configured judge model
    """
    if config is None:
        config = load_config()
    
    # Use judge-specific config if available, otherwise default
    model_config = config['llms'].get('judge_model', config['llms']['default'])
    base_model = create_model_from_config(model_config)
    
    return JudgeModel(base_model, config)


def get_models() -> tuple[AppellantModel, JudgeModel]:
    """
    Get both appellant and judge models.
    
    Returns:
        Tuple of (appellant_model, judge_model)
    """
    config = load_config()
    
    appellant = create_appellant_model(config)
    judge = create_judge_model(config)
    
    return appellant, judge


if __name__ == "__main__":
    # Test the models
    appellant, judge = get_models()
    
    print("Testing appellant model:")
    test_prompt = "What is your position on this constitutional issue?"
    appellant_response = appellant.prompt(test_prompt)
    print(f"Appellant: {appellant_response[:200]}...")
    
    print("\nTesting judge model:")
    judge_response = judge.prompt(test_prompt)
    print(f"Judge: {judge_response[:200]}...")