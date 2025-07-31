"""
Character question generation from rubric attributes.

This module processes a moot court rubric file and generates questions
about character attributes for belief measurement.
"""

import json
from typing import List, Dict, Any
from pathlib import Path


def load_rubric_attributes(rubric_path: Path) -> List[str]:
    """
    Load attributes from moot_rubric.txt file.
    
    Args:
        rubric_path: Path to the rubric file (one attribute per line)
        
    Returns:
        List of attribute strings
    """
    try:
        with open(rubric_path, 'r', encoding='utf-8') as f:
            attributes = [line.strip() for line in f if line.strip()]
        return attributes
    except FileNotFoundError:
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
    except Exception as e:
        raise ValueError(f"Error reading rubric file: {str(e)}")


def load_question_template(template_path: Path) -> str:
    """
    Load the question template from character_attribute_question.txt.
    
    Args:
        template_path: Path to the template file
        
    Returns:
        Template string containing ATTRIBUTE_TEXT placeholder
    """
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read().strip()
        return template
    except FileNotFoundError:
        raise FileNotFoundError(f"Template file not found: {template_path}")
    except Exception as e:
        raise ValueError(f"Error reading template file: {str(e)}")


def generate_character_questions(
    attributes: List[str],
    template: str
) -> List[Dict[str, Any]]:
    """
    Generate character questions by substituting attributes into template.
    
    Args:
        attributes: List of attribute strings
        template: Question template with ATTRIBUTE_TEXT placeholder
        
    Returns:
        List of question dictionaries with metadata
    """
    questions = []
    
    for i, attribute in enumerate(attributes):
        # Replace ATTRIBUTE_TEXT with the actual attribute
        question_text = template.replace("ATTRIBUTE_TEXT", attribute)
        
        question_data = {
            "id": i,
            "attribute": attribute,
            "question": question_text,
            "category": "character",
            "type": "appellant_character"
        }
        
        questions.append(question_data)
    
    return questions


def create_character_questions_file(
    rubric_path: Path,
    template_path: Path,
    output_path: Path
) -> List[Dict[str, Any]]:
    """
    Complete pipeline: load rubric and template, generate questions, save to file.
    
    Args:
        rubric_path: Path to moot_rubric.txt
        template_path: Path to character_attribute_question.txt
        output_path: Path to save generated questions JSON
        
    Returns:
        List of generated question dictionaries
    """
    print(f"Loading attributes from {rubric_path}")
    attributes = load_rubric_attributes(rubric_path)
    print(f"Loaded {len(attributes)} attributes")
    
    print(f"Loading question template from {template_path}")
    template = load_question_template(template_path)
    print("Template loaded successfully")
    
    print("Generating character questions...")
    questions = generate_character_questions(attributes, template)
    print(f"Generated {len(questions)} questions")
    
    # Save to output file
    print(f"Saving questions to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2)
    
    print("Character questions generated successfully")
    return questions


def load_character_questions(questions_path: Path) -> List[Dict[str, Any]]:
    """
    Load previously generated character questions from JSON file.
    
    Args:
        questions_path: Path to the questions JSON file
        
    Returns:
        List of question dictionaries
    """
    try:
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions
    except FileNotFoundError:
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in questions file: {str(e)}")


def get_question_texts(questions: List[Dict[str, Any]]) -> List[str]:
    """
    Extract just the question text strings from question dictionaries.
    
    Args:
        questions: List of question dictionaries
        
    Returns:
        List of question text strings
    """
    return [q["question"] for q in questions]


# CLI function for Snakemake integration
def main():
    """Command-line interface for character question generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate character questions from rubric attributes"
    )
    parser.add_argument(
        "--rubric", 
        required=True, 
        type=Path,
        help="Path to moot_rubric.txt file"
    )
    parser.add_argument(
        "--template", 
        required=True, 
        type=Path,
        help="Path to character_attribute_question.txt template"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        type=Path,
        help="Output path for generated questions JSON"
    )
    
    args = parser.parse_args()
    
    try:
        questions = create_character_questions_file(
            args.rubric,
            args.template,
            args.output
        )
        
        print(f"\nGenerated {len(questions)} character questions:")
        for q in questions[:3]:  # Show first 3 as examples
            print(f"  - {q['attribute']}: {q['question'][:100]}...")
        
        if len(questions) > 3:
            print(f"  ... and {len(questions) - 3} more")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())