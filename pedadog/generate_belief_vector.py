"""
Generate belief vectors from legal case PDFs.

This module provides functionality to extract arguments from legal briefs
and generate belief measurements using the thermometer module.
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF

from .thermometer import thermo, BeliefResults


DEFAULT_LLM = None  # To be set by the application


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
            text += "\n"  # Add page break
            
        doc.close()
        return text
    except Exception as e:
        raise ValueError(f"Error reading PDF {pdf_path}: {str(e)}")


def find_table_of_contents_page(text: str) -> Optional[str]:
    """
    Find and extract the TABLE OF CONTENTS section from the document text.
    
    Args:
        text: Full document text
        
    Returns:
        TABLE OF CONTENTS section text if found, None otherwise
    """
    # Look for TABLE OF CONTENTS heading (case insensitive)
    toc_pattern = r'(?i)table\s+of\s+contents.*?(?=\n\s*[A-Z][A-Z\s]{10,}|\Z)'
    
    match = re.search(toc_pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    
    return None


def extract_arguments_from_toc(toc_text: str, document_type: str) -> List[Dict[str, Any]]:
    """
    Extract arguments from the TABLE OF CONTENTS section.
    
    Args:
        toc_text: TABLE OF CONTENTS text
        document_type: Either 'petitioner' or 'respondent'
        
    Returns:
        List of argument dictionaries
    """
    arguments = []
    
    # Look for the ARGUMENT section in the TOC
    # This is a simplified pattern - real legal documents may vary significantly
    argument_section_pattern = r'(?i)argument\s*\.?\s*\n(.*?)(?=\n\s*[A-Z][A-Z\s]{10,}|\Z)'
    
    match = re.search(argument_section_pattern, toc_text, re.DOTALL)
    if not match:
        return arguments
    
    argument_text = match.group(1)
    
    # Split into lines and process
    lines = [line.strip() for line in argument_text.split('\n') if line.strip()]
    
    current_argument = None
    current_sub_arguments = []
    
    for line in lines:
        # Check if this looks like a main argument (typically starts with Roman numeral or letter)
        if re.match(r'^[IVX]+\.|\b[A-Z]\.', line) or (len(line) > 20 and not line.startswith(' ')):
            # Save previous argument if it exists
            if current_argument:
                arguments.append({
                    'argument': current_argument,
                    'sub_arguments': current_sub_arguments.copy(),
                    'type': document_type
                })
            
            # Start new argument
            current_argument = line
            current_sub_arguments = []
        
        # Check if this looks like a sub-argument (indented or numbered differently)
        elif re.match(r'^\s+[0-9]+\.|\s+[a-z]\.|\s+\([a-z]\)', line) or line.startswith('    '):
            if current_argument:  # Only add if we have a main argument
                current_sub_arguments.append(line.strip())
    
    # Don't forget the last argument
    if current_argument:
        arguments.append({
            'argument': current_argument,
            'sub_arguments': current_sub_arguments,
            'type': document_type
        })
    
    return arguments


def extract_arguments_with_llm(pdf_text: str, model, document_type: str) -> List[Dict[str, Any]]:
    """
    Use an LLM to extract arguments from the PDF text.
    
    Args:
        pdf_text: Full text of the PDF
        model: LLM model with .prompt() method
        document_type: Either 'petitioner' or 'respondent'
        
    Returns:
        List of argument dictionaries extracted by the LLM
    """
    extraction_prompt = f"""
    Please extract the legal arguments from this {document_type} brief. Look for the TABLE OF CONTENTS 
    section and identify all arguments listed under "ARGUMENT". 

    For each argument found, please return a JSON array where each object has:
    - "argument": the main argument text
    - "sub_arguments": array of any sub-arguments or sub-points
    - "type": "{document_type}"

    Here is the document text:

    {pdf_text[:10000]}  # Limit to first 10k characters to avoid token limits

    Please respond with only the JSON array, no other text.
    """
    
    try:
        response = model.prompt(extraction_prompt)
        
        # Try to extract JSON from the response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            arguments = json.loads(json_str)
            
            # Validate the structure
            validated_arguments = []
            for arg in arguments:
                if isinstance(arg, dict) and 'argument' in arg:
                    validated_arg = {
                        'argument': str(arg.get('argument', '')),
                        'sub_arguments': arg.get('sub_arguments', []),
                        'type': document_type
                    }
                    validated_arguments.append(validated_arg)
            
            return validated_arguments
        else:
            # Fallback: treat the response as a single argument
            return [{
                'argument': response,
                'sub_arguments': [],
                'type': document_type
            }]
            
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error parsing LLM response for {document_type}: {e}")
        # Return empty list on error
        return []


def extract_arguments_from_pdfs(
    petitioner_pdf: Path,
    respondent_pdf: Path,
    model=None,
    output_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Extract arguments from petitioner and respondent PDF files.
    
    Args:
        petitioner_pdf: Path to petitioner.pdf
        respondent_pdf: Path to respondent.pdf
        model: LLM model for extraction (optional, will try rule-based first)
        output_path: Path to save the extracted arguments JSON (optional)
        
    Returns:
        List of all arguments from both documents
    """
    all_arguments = []
    
    # Process petitioner PDF
    print("Processing petitioner PDF...")
    petitioner_text = extract_pdf_text(petitioner_pdf)
    
    if model:
        petitioner_args = extract_arguments_with_llm(petitioner_text, model, 'petitioner')
    else:
        # Try rule-based extraction
        toc_text = find_table_of_contents_page(petitioner_text)
        if toc_text:
            petitioner_args = extract_arguments_from_toc(toc_text, 'petitioner')
        else:
            print("Could not find TABLE OF CONTENTS in petitioner PDF")
            petitioner_args = []
    
    all_arguments.extend(petitioner_args)
    print(f"Extracted {len(petitioner_args)} arguments from petitioner brief")
    
    # Process respondent PDF
    print("Processing respondent PDF...")
    respondent_text = extract_pdf_text(respondent_pdf)
    
    if model:
        respondent_args = extract_arguments_with_llm(respondent_text, model, 'respondent')
    else:
        # Try rule-based extraction
        toc_text = find_table_of_contents_page(respondent_text)
        if toc_text:
            respondent_args = extract_arguments_from_toc(toc_text, 'respondent')
        else:
            print("Could not find TABLE OF CONTENTS in respondent PDF")
            respondent_args = []
    
    all_arguments.extend(respondent_args)
    print(f"Extracted {len(respondent_args)} arguments from respondent brief")
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(all_arguments, f, indent=2)
        print(f"Saved arguments to {output_path}")
    
    return all_arguments


def generate_belief_vector_from_arguments(
    prompt: str,
    arguments: List[Dict[str, Any]],
    models: Optional[List] = None,
    context: str = "",
    **thermo_kwargs
) -> BeliefResults:
    """
    Generate belief vectors by running thermo on prompt + arguments.
    
    Args:
        prompt: Base prompt to concatenate with each argument
        arguments: List of argument dictionaries
        models: List of models to use (defaults to DEFAULT_LLM)
        context: Additional context for the thermometer
        **thermo_kwargs: Additional arguments to pass to thermo()
        
    Returns:
        BeliefResults containing distributions for each prompt+argument combination
    """
    if models is None:
        if DEFAULT_LLM is None:
            raise ValueError("No models provided and DEFAULT_LLM is not set")
        models = [DEFAULT_LLM]
    
    # Create questions by concatenating prompt with each argument
    questions = []
    for arg in arguments:
        # Include main argument and sub-arguments in the question
        arg_text = arg['argument']
        if arg['sub_arguments']:
            sub_args_text = ' '.join(arg['sub_arguments'])
            full_arg_text = f"{arg_text} {sub_args_text}"
        else:
            full_arg_text = arg_text
        
        question = f"{prompt.strip()} {full_arg_text.strip()}"
        questions.append(question)
    
    # Run thermometer
    return thermo(
        questions=questions,
        context=context,
        models=models,
        **thermo_kwargs
    )


def generate_belief_vector_from_pdfs(
    petitioner_pdf: Path,
    respondent_pdf: Path,
    prompt: str,
    extraction_model=None,
    belief_models: Optional[List] = None,
    context: str = "",
    save_arguments_path: Optional[Path] = None,
    **thermo_kwargs
) -> tuple[List[Dict[str, Any]], BeliefResults]:
    """
    Complete pipeline: extract arguments from PDFs and generate belief vectors.
    
    Args:
        petitioner_pdf: Path to petitioner.pdf
        respondent_pdf: Path to respondent.pdf  
        prompt: Base prompt for belief measurement
        extraction_model: Model for argument extraction
        belief_models: Models for belief measurement
        context: Additional context
        save_arguments_path: Path to save extracted arguments
        **thermo_kwargs: Additional arguments for thermo()
        
    Returns:
        Tuple of (extracted_arguments, belief_results)
    """
    # Extract arguments
    arguments = extract_arguments_from_pdfs(
        petitioner_pdf,
        respondent_pdf,
        model=extraction_model,
        output_path=save_arguments_path
    )
    
    # Generate belief vectors
    belief_results = generate_belief_vector_from_arguments(
        prompt=prompt,
        arguments=arguments,
        models=belief_models,
        context=context,
        **thermo_kwargs
    )
    
    return arguments, belief_results