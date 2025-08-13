#!/usr/bin/env python3
"""
Test script for the new PDF argument extraction strategy.
"""

import json
from pathlib import Path
from pedadog.generate_belief_vector import extract_arguments_from_pdfs
from pedadog.models import AISandboxModel

def load_sample_data():
    """Load the sample data for comparison."""
    sample_path = Path("data/sample_data.json")
    with open(sample_path, 'r') as f:
        return json.load(f)

def compare_structures(extracted, sample):
    """Compare extracted arguments with sample data structure."""
    print(f"\n=== STRUCTURE COMPARISON ===")
    print(f"Sample has {len(sample)} arguments")
    print(f"Extracted has {len(extracted)} arguments")
    
    # Count arguments by type
    sample_petitioner = [arg for arg in sample if arg['type'] == 'petitioner']
    sample_respondent = [arg for arg in sample if arg['type'] == 'respondent']
    extracted_petitioner = [arg for arg in extracted if arg['type'] == 'petitioner'] 
    extracted_respondent = [arg for arg in extracted if arg['type'] == 'respondent']
    
    print(f"\nSample: {len(sample_petitioner)} petitioner, {len(sample_respondent)} respondent")
    print(f"Extracted: {len(extracted_petitioner)} petitioner, {len(extracted_respondent)} respondent")
    
    # Check nested structure depth
    def get_max_depth(args, current_depth=0):
        max_depth = current_depth
        for arg in args:
            if 'sub_arguments' in arg and arg['sub_arguments']:
                depth = get_max_depth(arg['sub_arguments'], current_depth + 1)
                max_depth = max(max_depth, depth)
        return max_depth
    
    sample_depth = get_max_depth(sample)
    extracted_depth = get_max_depth(extracted)
    
    print(f"\nNesting depth - Sample: {sample_depth}, Extracted: {extracted_depth}")
    
    return True

def print_argument_tree(args, indent=0):
    """Print arguments in a tree structure for comparison."""
    for i, arg in enumerate(args):
        prefix = "  " * indent + f"{i+1}. "
        print(f"{prefix}{arg['argument'][:80]}...")
        if 'sub_arguments' in arg and arg['sub_arguments']:
            print_argument_tree(arg['sub_arguments'], indent + 1)

def main():
    print("Testing new PDF argument extraction strategy...")
    
    # Load sample data
    sample_data = load_sample_data()
    print(f"Loaded sample data with {len(sample_data)} arguments")
    
    # Set up model
    model = AISandboxModel()
    
    # Test extraction on actual PDFs
    petitioner_pdf = Path("data/petitioner.pdf")
    respondent_pdf = Path("data/respondent.pdf")
    
    if not petitioner_pdf.exists() or not respondent_pdf.exists():
        print("PDF files not found. Please ensure petitioner.pdf and respondent.pdf are in data/ folder")
        return
    
    print("\nExtracting arguments with new strategy...")
    extracted_args = extract_arguments_from_pdfs(
        petitioner_pdf=petitioner_pdf,
        respondent_pdf=respondent_pdf,
        model=model,
        use_new_strategy=True,
        context_words=3000
    )
    
    # Save extracted arguments for inspection
    output_path = Path("extracted_arguments_test.json")
    with open(output_path, 'w') as f:
        json.dump(extracted_args, f, indent=2)
    print(f"Saved extracted arguments to {output_path}")
    
    # Compare structures
    compare_structures(extracted_args, sample_data)
    
    # Print both structures for manual comparison
    print(f"\n=== SAMPLE ARGUMENTS STRUCTURE ===")
    print_argument_tree(sample_data)
    
    print(f"\n=== EXTRACTED ARGUMENTS STRUCTURE ===")
    print_argument_tree(extracted_args)
    
    # Check for key differences
    print(f"\n=== CONTENT COMPARISON ===")
    
    # Check if we have similar main arguments
    sample_main_args = [arg['argument'] for arg in sample_data]
    extracted_main_args = [arg['argument'] for arg in extracted_args]
    
    print("Sample main arguments:")
    for i, arg in enumerate(sample_main_args[:3]):  # Show first 3
        print(f"  {i+1}. {arg[:100]}...")
    
    print("\nExtracted main arguments:")
    for i, arg in enumerate(extracted_main_args[:3]):  # Show first 3
        print(f"  {i+1}. {arg[:100]}...")

if __name__ == "__main__":
    main()