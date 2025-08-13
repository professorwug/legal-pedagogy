#!/usr/bin/env python3
"""
Final comparison of extracted vs sample arguments.
"""

import json
from pathlib import Path

def compare_exact_match(extracted, sample):
    """Compare extracted arguments with sample for exact matches."""
    print("=== EXACT ARGUMENT MATCHING ===")
    
    # Check main arguments
    for i, (ext_arg, sample_arg) in enumerate(zip(extracted, sample)):
        ext_text = ext_arg['argument']
        sample_text = sample_arg['argument']
        
        match = ext_text == sample_text
        print(f"{i+1}. Main argument match: {'✓' if match else '✗'}")
        if not match:
            print(f"   Expected: {sample_text}")
            print(f"   Got:      {ext_text}")
        
        # Check sub-arguments
        ext_subs = ext_arg.get('sub_arguments', [])
        sample_subs = sample_arg.get('sub_arguments', [])
        
        if len(ext_subs) != len(sample_subs):
            print(f"   Sub-arg count mismatch: expected {len(sample_subs)}, got {len(ext_subs)}")
        
        for j, (ext_sub, sample_sub) in enumerate(zip(ext_subs, sample_subs)):
            ext_sub_text = ext_sub['argument']
            sample_sub_text = sample_sub['argument']
            
            sub_match = ext_sub_text == sample_sub_text
            print(f"   {i+1}.{j+1} Sub-argument match: {'✓' if sub_match else '✗'}")
            if not sub_match:
                print(f"        Expected: {sample_sub_text}")
                print(f"        Got:      {ext_sub_text}")

def main():
    # Load both files
    extracted_path = Path("extracted_arguments_test.json")
    sample_path = Path("data/sample_data.json")
    
    with open(extracted_path, 'r') as f:
        extracted = json.load(f)
    
    with open(sample_path, 'r') as f:
        sample = json.load(f)
    
    print(f"Comparing {len(extracted)} extracted vs {len(sample)} sample arguments")
    
    # Compare exact matches
    compare_exact_match(extracted, sample)
    
    # Show quality metrics
    print(f"\n=== QUALITY METRICS ===")
    print(f"Structure match: {'✓' if len(extracted) == len(sample) else '✗'}")
    
    petitioner_ext = [a for a in extracted if a['type'] == 'petitioner']
    petitioner_sample = [a for a in sample if a['type'] == 'petitioner']
    respondent_ext = [a for a in extracted if a['type'] == 'respondent']
    respondent_sample = [a for a in sample if a['type'] == 'respondent']
    
    print(f"Petitioner count: {len(petitioner_ext)} vs {len(petitioner_sample)} {'✓' if len(petitioner_ext) == len(petitioner_sample) else '✗'}")
    print(f"Respondent count: {len(respondent_ext)} vs {len(respondent_sample)} {'✓' if len(respondent_ext) == len(respondent_sample) else '✗'}")
    
    # Count total exact matches
    exact_matches = 0
    total_args = 0
    
    for ext_arg, sample_arg in zip(extracted, sample):
        total_args += 1
        if ext_arg['argument'] == sample_arg['argument']:
            exact_matches += 1
        
        # Count sub-arguments
        for ext_sub, sample_sub in zip(ext_arg.get('sub_arguments', []), sample_arg.get('sub_arguments', [])):
            total_args += 1
            if ext_sub['argument'] == sample_sub['argument']:
                exact_matches += 1
    
    match_percentage = (exact_matches / total_args) * 100 if total_args > 0 else 0
    print(f"Exact match rate: {exact_matches}/{total_args} ({match_percentage:.1f}%)")

if __name__ == "__main__":
    main()