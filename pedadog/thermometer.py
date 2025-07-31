"""
Thermometer module for distributional belief measurement.

This module provides functions for systematically querying judge models
to generate belief distributions about questions in a given context.
"""

import asyncio
import re
import time
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from rich.progress import Progress, TaskID
from rich.console import Console


@dataclass
class BeliefResponse:
    """Container for a single belief measurement response."""
    raw_response: str
    numeric_value: Optional[float]
    timestamp: float
    runtime_s: float


class BeliefDistribution:
    """Container for belief distribution data with convenience methods."""
    
    def __init__(self, model_name: str, question: str, responses: List[BeliefResponse]):
        self.model_name = model_name
        self.question = question
        self.responses = responses
        self._valid_responses = [r for r in responses if r.numeric_value is not None]
    
    @property
    def values(self) -> List[float]:
        """Get all valid numeric values."""
        return [r.numeric_value for r in self._valid_responses]
    
    @property
    def mean(self) -> float:
        """Calculate mean of valid responses."""
        if not self._valid_responses:
            return 0.0
        return np.mean(self.values)
    
    @property
    def variance(self) -> float:
        """Calculate variance of valid responses."""
        if len(self._valid_responses) < 2:
            return 0.0
        return np.var(self.values)
    
    @property
    def std(self) -> float:
        """Calculate standard deviation of valid responses."""
        return np.sqrt(self.variance)
    
    @property
    def valid_count(self) -> int:
        """Number of valid numeric responses."""
        return len(self._valid_responses)
    
    @property
    def total_count(self) -> int:
        """Total number of responses."""
        return len(self.responses)
    
    @property
    def rejection_rate(self) -> float:
        """Rate of responses that couldn't be parsed as numeric."""
        if not self.responses:
            return 1.0
        return 1.0 - (self.valid_count / self.total_count)


class BeliefResults:
    """Nested dictionary-like container for belief measurement results."""
    
    def __init__(self):
        self._results: Dict[str, Dict[str, BeliefDistribution]] = {}
    
    def add_result(self, model_name: str, question: str, distribution: BeliefDistribution):
        """Add a belief distribution result."""
        if model_name not in self._results:
            self._results[model_name] = {}
        self._results[model_name][question] = distribution
    
    def get(self, model_name: str, question: str) -> Optional[BeliefDistribution]:
        """Get a specific belief distribution."""
        return self._results.get(model_name, {}).get(question)
    
    def get_model_results(self, model_name: str) -> Dict[str, BeliefDistribution]:
        """Get all results for a specific model."""
        return self._results.get(model_name, {})
    
    def get_question_results(self, question: str) -> Dict[str, BeliefDistribution]:
        """Get results for a specific question across all models."""
        return {model: results[question] 
                for model, results in self._results.items() 
                if question in results}
    
    @property
    def model_names(self) -> List[str]:
        """Get all model names."""
        return list(self._results.keys())
    
    @property
    def questions(self) -> List[str]:
        """Get all unique questions."""
        questions = set()
        for model_results in self._results.values():
            questions.update(model_results.keys())
        return list(questions)
    
    def __getitem__(self, key):
        """Dictionary-like access."""
        return self._results[key]
    
    def __contains__(self, key):
        """Dictionary-like membership test."""
        return key in self._results


def extract_numeric_value(response: str, min_val: float = 0.0, max_val: float = 1.0) -> Optional[float]:
    """
    Extract numeric value from model response.
    
    Args:
        response: Raw response string from model
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Numeric value if found and valid, None otherwise
    """
    # Look for decimal numbers between min_val and max_val
    pattern = r'-?(?:0(?:\.\d+)?|1(?:\.0+)?|\d*\.\d+)'
    matches = re.findall(pattern, response.strip())
    
    for match in matches:
        try:
            value = float(match)
            if min_val <= value <= max_val:
                return value
        except ValueError:
            continue
    
    return None


def monte_carlo_belief_of(
    question: str,
    model,
    n_samples: int = 20,
    min_val: float = 0.0,
    max_val: float = 1.0,
    max_workers: int = 8
) -> List[BeliefResponse]:
    """
    Perform Monte Carlo sampling of belief for a single question.
    
    Args:
        question: The question to ask the model
        model: Model object with .prompt(str) method and .name attribute
        n_samples: Number of samples to collect
        min_val: Minimum valid numeric response
        max_val: Maximum valid numeric response
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of BeliefResponse objects
    """
    responses = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(_single_belief_query, question, model, min_val, max_val): i 
            for i in range(n_samples)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_sample):
            try:
                response = future.result()
                responses.append(response)
            except Exception as e:
                # Log error but continue with other samples
                console = Console()
                console.print(f"[red]Error in sample query: {e}[/red]")
    
    return responses


def _single_belief_query(
    question: str, 
    model, 
    min_val: float, 
    max_val: float
) -> BeliefResponse:
    """Execute a single belief query to the model."""
    start_time = time.time()
    
    try:
        raw_response = model.prompt(question)
        numeric_value = extract_numeric_value(raw_response, min_val, max_val)
    except Exception as e:
        raw_response = f"ERROR: {str(e)}"
        numeric_value = None
    
    end_time = time.time()
    
    return BeliefResponse(
        raw_response=raw_response,
        numeric_value=numeric_value,
        timestamp=end_time,
        runtime_s=end_time - start_time
    )


def thermo(
    questions: List[str],
    context: str,
    models: List,
    n_samples: int = 20,
    min_val: float = 0.0,
    max_val: float = 1.0,
    max_workers: int = 8
) -> BeliefResults:
    """
    Main thermometer function for measuring belief distributions.
    
    Args:
        questions: List of questions to ask
        context: Context string to prepend to questions (can be empty)
        models: List of model objects with .prompt() method and .name attribute
        n_samples: Number of Monte Carlo samples per question per model
        min_val: Minimum valid numeric response
        max_val: Maximum valid numeric response
        max_workers: Maximum number of parallel workers
        
    Returns:
        BeliefResults object containing all distributions
    """
    results = BeliefResults()
    console = Console()
    
    # Prepare contextualized questions
    contextualized_questions = []
    for q in questions:
        if context.strip():
            contextualized_q = f"{context.strip()}\n\n{q}"
        else:
            contextualized_q = q
        contextualized_questions.append(contextualized_q)
    
    total_tasks = len(questions) * len(models)
    
    with Progress(console=console) as progress:
        task = progress.add_task(
            f"[cyan]Measuring beliefs...", 
            total=total_tasks
        )
        
        completed_tasks = 0
        start_time = time.time()
        
        for model in models:
            model_name = getattr(model, 'name', str(model))
            
            for i, (question, contextualized_q) in enumerate(zip(questions, contextualized_questions)):
                # Perform Monte Carlo sampling for this question-model pair
                responses = monte_carlo_belief_of(
                    contextualized_q,
                    model,
                    n_samples=n_samples,
                    min_val=min_val,
                    max_val=max_val,
                    max_workers=max_workers
                )
                
                # Create distribution object
                distribution = BeliefDistribution(model_name, question, responses)
                results.add_result(model_name, question, distribution)
                
                # Update progress
                completed_tasks += 1
                elapsed_time = time.time() - start_time
                avg_time_per_task = elapsed_time / completed_tasks
                remaining_tasks = total_tasks - completed_tasks
                eta = avg_time_per_task * remaining_tasks
                
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Beliefs: {completed_tasks}/{total_tasks} "
                                f"(ETA: {eta:.1f}s, {completed_tasks * n_samples} calls made)"
                )
    
    return results