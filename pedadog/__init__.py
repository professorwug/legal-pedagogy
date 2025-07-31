"""
Pedadog: Legal Pedagogy Analysis Package

This package provides tools for analyzing case beliefs and character beliefs
in legal pedagogy contexts, including distributional belief measurement
and PDF argument extraction.
"""

__version__ = "0.1.0"
__author__ = "Legal Pedagogy Team"

from .case_beliefs import CaseBeliefAnalyzer
from .character_beliefs import CharacterBeliefAnalyzer
from .thermometer import thermo, monte_carlo_belief_of, BeliefResults, BeliefDistribution
from .generate_belief_vector import (
    extract_arguments_from_pdfs, 
    generate_belief_vector_from_arguments,
    generate_belief_vector_from_pdfs
)
from .make_character_questions import create_character_questions_file, load_character_questions
from .models import (
    BaseLLM, 
    AnthropicModel, 
    MockLLM,
    create_anthropic_model,
    create_mock_model,
    get_default_model,
    set_default_llm
)

__all__ = [
    "CaseBeliefAnalyzer", 
    "CharacterBeliefAnalyzer",
    "thermo",
    "monte_carlo_belief_of", 
    "BeliefResults",
    "BeliefDistribution",
    "extract_arguments_from_pdfs",
    "generate_belief_vector_from_arguments", 
    "generate_belief_vector_from_pdfs",
    "create_character_questions_file",
    "load_character_questions",
    "BaseLLM",
    "AnthropicModel",
    "MockLLM", 
    "create_anthropic_model",
    "create_mock_model",
    "get_default_model",
    "set_default_llm"
]