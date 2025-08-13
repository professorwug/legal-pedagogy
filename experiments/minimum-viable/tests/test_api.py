"""
Tests for the pedadog API functions.

This module tests:
1. Argument extraction from PDFs
2. Belief vector generation with arguments and character beliefs
3. Logging of all judge queries
"""

import json
import pytest
from pathlib import Path
import numpy as np
from datetime import datetime

# Add parent directories to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pedadog import extract_arguments, belief_vector
from pedadog.models import MockLLM
from simple_models import JudgeModel, load_config


class LoggingJudgeModel:
    """Judge model wrapper that logs all queries."""
    
    def __init__(self, base_judge_model, log_dir: Path):
        self.base_model = base_judge_model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.queries = []
        self.name = f"logging-{base_judge_model.name}"
        
        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"judge_queries_{timestamp}.json"
    
    def prompt(self, text: str) -> str:
        """Log the query and pass to base model."""
        # Log the query
        query_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": text,
            "query_number": len(self.queries) + 1
        }
        
        # Get response from base model
        response = self.base_model.prompt(text)
        
        # Add response to log entry
        query_entry["response"] = response
        
        # Store in memory
        self.queries.append(query_entry)
        
        # Write to file (append mode)
        self._save_queries()
        
        return response
    
    def _save_queries(self):
        """Save all queries to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.queries, f, indent=2)
    
    def get_query_summary(self):
        """Get summary of logged queries."""
        summary = {
            "total_queries": len(self.queries),
            "log_file": str(self.log_file),
            "query_types": self._categorize_queries()
        }
        return summary
    
    def _categorize_queries(self):
        """Categorize queries by type."""
        categories = {
            "argument_beliefs": 0,
            "character_assessments": 0,
            "other": 0
        }
        
        for query in self.queries:
            text = query["query"].lower()
            if "agree with" in text and ("petitioner" in text or "respondent" in text):
                categories["argument_beliefs"] += 1
            elif "rate the appellant" in text or "based on this legal interaction" in text:
                categories["character_assessments"] += 1
            else:
                categories["other"] += 1
        
        return categories


def test_extract_arguments():
    """Test that arguments are extracted correctly from PDFs."""
    print("\n=== Testing Argument Extraction ===")
    
    # Get paths to PDFs
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    petitioner_pdf = data_dir / "petitioner.pdf"
    respondent_pdf = data_dir / "respondent.pdf"
    
    # Check that PDFs exist
    assert petitioner_pdf.exists(), f"Petitioner PDF not found at {petitioner_pdf}"
    assert respondent_pdf.exists(), f"Respondent PDF not found at {respondent_pdf}"
    
    # Extract arguments
    arguments = extract_arguments(
        pdf_petitioner=petitioner_pdf,
        pdf_respondent=respondent_pdf,
        context_words=3000
    )
    
    # Save extracted arguments for inspection
    output_dir = Path(__file__).parent.parent / "processed_data"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "test_extracted_arguments.json"
    with open(output_file, 'w') as f:
        json.dump(arguments, f, indent=2)
    
    print(f"Extracted arguments saved to: {output_file}")
    
    # Validate structure
    assert isinstance(arguments, list), "Arguments should be a list"
    assert len(arguments) > 0, "Should extract at least one argument"
    
    # Check for both petitioner and respondent arguments
    petitioner_args = [arg for arg in arguments if arg.get("type") == "petitioner"]
    respondent_args = [arg for arg in arguments if arg.get("type") == "respondent"]
    
    assert len(petitioner_args) > 0, "Should have petitioner arguments"
    assert len(respondent_args) > 0, "Should have respondent arguments"
    
    # Validate argument structure
    for arg in arguments:
        assert "argument" in arg, "Each argument should have 'argument' field"
        assert "type" in arg, "Each argument should have 'type' field"
        assert arg["type"] in ["petitioner", "respondent"], "Type should be petitioner or respondent"
        assert isinstance(arg.get("sub_arguments", []), list), "sub_arguments should be a list"
    
    print(f"✓ Extracted {len(petitioner_args)} petitioner arguments")
    print(f"✓ Extracted {len(respondent_args)} respondent arguments")
    print(f"✓ Total arguments: {len(arguments)}")
    
    return arguments


def test_belief_vector_with_logging():
    """Test belief vector generation with query logging."""
    print("\n=== Testing Belief Vector with Logging ===")
    
    # First extract arguments
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    arguments = extract_arguments(
        pdf_petitioner=data_dir / "petitioner.pdf",
        pdf_respondent=data_dir / "respondent.pdf"
    )
    
    # Create a mock context
    context = "This is a Supreme Court oral argument about pharmaceutical regulation and FDA approval processes."
    
    # Create logging judge models
    processed_data_dir = Path(__file__).parent.parent / "processed_data"
    
    # Use MockLLM for testing (faster and deterministic)
    mock_base = MockLLM()
    config = load_config()
    
    # Create judge model with mock base
    judge_model = JudgeModel(mock_base, config)
    
    # Wrap with logging
    logging_judge1 = LoggingJudgeModel(judge_model, processed_data_dir / "judge1_logs")
    logging_judge2 = LoggingJudgeModel(judge_model, processed_data_dir / "judge2_logs")
    
    # Path to character rubric
    rubric_path = data_dir / "moot_rubric.txt"
    
    # Generate belief vectors with small n_samples for testing
    belief_distributions = belief_vector(
        arguments[:3],  # Use only first 3 arguments for faster testing
        context,
        logging_judge1,
        logging_judge2,
        path_to_character_rubric=rubric_path,
        n_samples=3  # Small number for testing
    )
    
    # Validate belief distributions structure
    assert isinstance(belief_distributions, list), "Should return a list of distributions"
    assert len(belief_distributions) == 2, "Should have distributions for 2 judges"
    
    # Check each judge's beliefs
    for i, judge_beliefs in enumerate(belief_distributions):
        assert isinstance(judge_beliefs, dict), f"Judge {i+1} beliefs should be a dict"
        
        # Check that we have beliefs for arguments
        argument_questions = [k for k in judge_beliefs.keys() if "agree with" in k]
        assert len(argument_questions) > 0, f"Judge {i+1} should have argument beliefs"
        
        # Check that we have character assessments
        character_questions = [k for k in judge_beliefs.keys() if "rate the appellant" in k.lower()]
        assert len(character_questions) > 0, f"Judge {i+1} should have character assessments"
        
        # Check that values are numpy arrays
        for question, values in judge_beliefs.items():
            assert isinstance(values, np.ndarray), f"Values should be numpy arrays"
            assert len(values) == 3, f"Should have 3 samples (n_samples=3)"
            # Check that values are in [0, 1] range
            assert np.all(values >= 0) and np.all(values <= 1), f"Values should be in [0, 1]"
    
    # Get and save query summaries
    summary1 = logging_judge1.get_query_summary()
    summary2 = logging_judge2.get_query_summary()
    
    print(f"✓ Judge 1 processed {summary1['total_queries']} queries")
    print(f"  - Argument beliefs: {summary1['query_types']['argument_beliefs']}")
    print(f"  - Character assessments: {summary1['query_types']['character_assessments']}")
    print(f"  - Log saved to: {summary1['log_file']}")
    
    print(f"✓ Judge 2 processed {summary2['total_queries']} queries")
    print(f"  - Argument beliefs: {summary2['query_types']['argument_beliefs']}")
    print(f"  - Character assessments: {summary2['query_types']['character_assessments']}")
    print(f"  - Log saved to: {summary2['log_file']}")
    
    # Save belief distributions for inspection
    output_file = processed_data_dir / "test_belief_distributions.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_beliefs = []
    for judge_beliefs in belief_distributions:
        serializable = {}
        for question, values in judge_beliefs.items():
            serializable[question] = values.tolist()
        serializable_beliefs.append(serializable)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_beliefs, f, indent=2)
    
    print(f"✓ Belief distributions saved to: {output_file}")
    
    return belief_distributions, logging_judge1, logging_judge2


def test_character_beliefs_integration():
    """Test that character beliefs are properly integrated."""
    print("\n=== Testing Character Beliefs Integration ===")
    
    # Create minimal test arguments
    test_arguments = [
        {
            "argument": "The FDA's authority should be limited",
            "type": "petitioner",
            "sub_arguments": []
        }
    ]
    
    context = "Test context for Supreme Court argument"
    rubric_path = Path(__file__).parent.parent.parent.parent / "data" / "moot_rubric.txt"
    
    # Create a logging judge for this test
    processed_data_dir = Path(__file__).parent.parent / "processed_data"
    mock_base = MockLLM()
    config = load_config()
    judge_model = JudgeModel(mock_base, config)
    logging_judge = LoggingJudgeModel(judge_model, processed_data_dir / "character_test_logs")
    
    # Generate beliefs with character rubric
    beliefs = belief_vector(
        test_arguments,
        context,
        logging_judge,
        path_to_character_rubric=rubric_path,
        n_samples=2
    )
    
    # Check that character questions were asked
    character_queries = [
        q for q in logging_judge.queries 
        if "rate the appellant" in q["query"].lower()
    ]
    
    # Count expected character attributes from rubric
    with open(rubric_path, 'r') as f:
        rubric_lines = [line.strip() for line in f.readlines() 
                       if line.strip() and not line.startswith("#")]
    
    expected_character_questions = len(rubric_lines) * 2  # n_samples=2
    
    assert len(character_queries) > 0, "Should have character assessment queries"
    assert len(character_queries) == expected_character_questions, \
        f"Expected {expected_character_questions} character queries, got {len(character_queries)}"
    
    print(f"✓ Character beliefs integrated: {len(character_queries)} queries")
    print(f"✓ Rubric attributes tested: {len(rubric_lines)}")
    
    # Verify specific rubric items are in queries
    sample_attributes = [
        "knowledge of the record",
        "respect and courtesy",
        "correct grammar"
    ]
    
    for attr in sample_attributes:
        matching_queries = [
            q for q in character_queries 
            if attr.lower() in q["query"].lower()
        ]
        assert len(matching_queries) > 0, f"Should have queries for '{attr}'"
        print(f"  ✓ Found queries for: {attr}")
    
    return beliefs


def test_all():
    """Run all tests."""
    print("\n" + "="*50)
    print("RUNNING PEDADOG API TESTS")
    print("="*50)
    
    # Test 1: Argument extraction
    arguments = test_extract_arguments()
    
    # Test 2: Belief vector with logging
    beliefs, judge1, judge2 = test_belief_vector_with_logging()
    
    # Test 3: Character beliefs integration
    character_beliefs = test_character_beliefs_integration()
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED ✓")
    print("="*50)
    
    print("\nCheck the processed_data directory for:")
    print("  - test_extracted_arguments.json")
    print("  - test_belief_distributions.json")
    print("  - judge1_logs/judge_queries_*.json")
    print("  - judge2_logs/judge_queries_*.json")
    print("  - character_test_logs/judge_queries_*.json")


if __name__ == "__main__":
    test_all()