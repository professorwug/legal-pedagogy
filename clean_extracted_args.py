#!/usr/bin/env python3
"""
Clean up extracted arguments to match sample format exactly.
"""

import json
import re
from pathlib import Path

def clean_argument_text(text: str) -> str:
    """Clean up argument text to match sample format."""
    # Remove Roman numerals and section letters at the beginning
    cleaned = re.sub(r'^[IVX]+\.\s*', '', text)
    cleaned = re.sub(r'^[A-Z]\.\s*', '', cleaned)
    
    # Convert to sentence case (first letter uppercase, rest lowercase unless proper noun)
    if cleaned:
        # Split into words to handle proper nouns
        words = cleaned.split()
        if words:
            # First word: capitalize first letter only
            words[0] = words[0].capitalize()
            
            # Other words: handle common legal terms and proper nouns
            for i in range(1, len(words)):
                word = words[i]
                # Keep certain terms uppercase (acronyms, legal terms)
                if word.upper() in ['FDA', 'FDCA', 'CBE', 'CRL', 'MERCK', 'LEVINE', 'CIRCUIT']:
                    words[i] = word.upper()
                # Handle possessives
                elif word.endswith("'s") or word.endswith("'s"):
                    base_word = word[:-2]
                    if base_word.upper() in ['FDA', 'FDCA', 'MERCK', 'CIRCUIT']:
                        words[i] = base_word.upper() + word[-2:]
                    else:
                        words[i] = word.lower()
                else:
                    words[i] = word.lower()
        
        cleaned = ' '.join(words)
    
    return cleaned

def clean_arguments_recursive(args, is_top_level=True):
    """Recursively clean all arguments in the structure."""
    cleaned_args = []
    
    for arg in args:
        cleaned_arg = {
            'argument': clean_argument_text(arg['argument'])
        }
        
        # Only add 'type' for top-level arguments
        if is_top_level and 'type' in arg:
            cleaned_arg['type'] = arg['type']
        
        # Handle sub_arguments
        if 'sub_arguments' in arg and arg['sub_arguments']:
            # For sub-arguments, we need to maintain the structure but clean text
            cleaned_sub_args = []
            for sub_arg in arg['sub_arguments']:
                if isinstance(sub_arg, dict):
                    cleaned_sub = {
                        'argument': clean_argument_text(sub_arg['argument'])
                    }
                    # Handle nested sub_arguments
                    if 'sub_arguments' in sub_arg and sub_arg['sub_arguments']:
                        cleaned_sub['sub_arguments'] = clean_arguments_recursive(sub_arg['sub_arguments'], is_top_level=False)
                    cleaned_sub_args.append(cleaned_sub)
                else:
                    # Handle string sub-arguments (shouldn't happen in our format)
                    cleaned_sub_args.append(clean_argument_text(str(sub_arg)))
            
            cleaned_arg['sub_arguments'] = cleaned_sub_args
        else:
            cleaned_arg['sub_arguments'] = []
        
        cleaned_args.append(cleaned_arg)
    
    return cleaned_args

def main():
    # Load extracted arguments
    extracted_path = Path("extracted_arguments_test.json")
    with open(extracted_path, 'r') as f:
        extracted_args = json.load(f)
    
    print(f"Loaded {len(extracted_args)} extracted arguments")
    
    # Clean the arguments
    cleaned_args = clean_arguments_recursive(extracted_args)
    
    # Save cleaned arguments
    cleaned_path = Path("cleaned_arguments.json")
    with open(cleaned_path, 'w') as f:
        json.dump(cleaned_args, f, indent=2)
    
    print(f"Saved cleaned arguments to {cleaned_path}")
    
    # Show comparison
    print("\n=== BEFORE CLEANING ===")
    print(f"1. {extracted_args[0]['argument'][:100]}...")
    print(f"   - {extracted_args[0]['sub_arguments'][0]['argument'][:80]}...")
    
    print("\n=== AFTER CLEANING ===")
    print(f"1. {cleaned_args[0]['argument'][:100]}...")
    print(f"   - {cleaned_args[0]['sub_arguments'][0]['argument'][:80]}...")
    
    # Load sample for comparison
    sample_path = Path("data/sample_data.json")
    with open(sample_path, 'r') as f:
        sample_args = json.load(f)
    
    print("\n=== SAMPLE FORMAT ===")
    print(f"1. {sample_args[0]['argument'][:100]}...")
    print(f"   - {sample_args[0]['sub_arguments'][0]['argument'][:80]}...")

if __name__ == "__main__":
    main()