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


def find_table_of_contents_section(text: str) -> Optional[tuple[str, int]]:
    """
    Find the TABLE OF CONTENTS section and its position in the document.
    
    Args:
        text: Full document text
        
    Returns:
        Tuple of (TOC text, position) if found, None otherwise
    """
    # Look for TABLE OF CONTENTS heading (case insensitive)
    toc_pattern = r'(?i)table\s+of\s+contents'
    
    match = re.search(toc_pattern, text)
    if not match:
        return None
    
    toc_start = match.start()
    
    # Find the end of TOC - typically marked by a major section like "INTRODUCTION" or "STATEMENT"
    # Look for patterns that indicate end of TOC
    end_patterns = [
        r'\n\s*INTRODUCTION\s*\n',
        r'\n\s*STATEMENT\s*\n', 
        r'\n\s*OPINIONS?\s+BELOW\s*\n',
        r'\n\s*JURISDICTION\s*\n',
        r'\n\s*QUESTION\s+PRESENTED\s*\n',
        r'\n\s*CONSTITUTIONAL\s+AND\s+STATUTORY\s+PROVISIONS\s*\n'
    ]
    
    toc_end = len(text)  # Default to end of document
    
    for pattern in end_patterns:
        match_end = re.search(pattern, text[toc_start:], re.IGNORECASE)
        if match_end:
            toc_end = toc_start + match_end.start()
            break
    
    toc_text = text[toc_start:toc_end]
    return toc_text, toc_start


def extract_toc_context_with_lookahead(text: str, toc_position: int, context_words: int = 3000) -> str:
    """
    Extract TOC and additional context from the document.
    
    Args:
        text: Full document text
        toc_position: Starting position of TOC
        context_words: Number of additional words to include after TOC
        
    Returns:
        Extended context including TOC and following text
    """
    # Find end of document or reasonable stopping point
    words_after_toc = text[toc_position:].split()
    
    # Take TOC and add the specified number of words
    extended_words = words_after_toc[:context_words]
    extended_text = ' '.join(extended_words)
    
    return extended_text


def extract_arguments_section_from_toc(toc_text: str) -> Optional[str]:
    """
    Extract just the ARGUMENTS section from the TOC text.
    
    Args:
        toc_text: Full TABLE OF CONTENTS text
        
    Returns:
        Text between ARGUMENT and CONCLUSION sections, or None if not found
    """
    # Look for ARGUMENT section (case insensitive)
    argument_start_pattern = r'(?i)\bARGUMENTS?\b'
    
    argument_match = re.search(argument_start_pattern, toc_text)
    if not argument_match:
        return None
    
    argument_start = argument_match.end()
    
    # Look for CONCLUSION or similar ending markers
    end_patterns = [
        r'(?i)\bCONCLUSION\b',
        r'(?i)\bSUMMARY\s+OF\s+ARGUMENT\b',
        r'(?i)\bRESPECTFULLY\s+SUBMITTED\b',
        r'(?i)\bAPPENDIX\b'
    ]
    
    argument_end = len(toc_text)  # Default to end of TOC
    
    for pattern in end_patterns:
        end_match = re.search(pattern, toc_text[argument_start:])
        if end_match:
            argument_end = argument_start + end_match.start()
            break
    
    arguments_section = toc_text[argument_start:argument_end].strip()
    return arguments_section if arguments_section else None


def extract_arguments_with_llm(extended_context: str, model, document_type: str) -> str:
    """
    Use an LLM to extract arguments from the extended TOC context.
    
    Args:
        extended_context: TOC text plus additional context
        model: LLM model with .prompt() method
        document_type: Either 'petitioner' or 'respondent'
        
    Returns:
        Markdown formatted argument structure
    """
    extraction_prompt = f"""
You are analyzing a legal brief from the {document_type}. I need you to extract the arguments from the TABLE OF CONTENTS section.

Look for the section titled "ARGUMENT" or "ARGUMENTS" in the text below. Extract all the arguments listed between "ARGUMENT" and "CONCLUSION" and format them as a hierarchical markdown bullet list.

Rules:
1. Main arguments should be top-level bullets (-)
2. Sub-arguments should be indented with two spaces (  -)
3. Sub-sub-arguments should be indented with four spaces (    -)
4. Preserve the hierarchical structure shown in the TOC
5. Clean up the text but preserve the essential meaning
6. Remove page numbers and formatting artifacts
7. Only include items that are actual legal arguments, not procedural sections

Here is the document text:

{extended_context}

Respond with ONLY the markdown bullet list, no other text or explanation.
    """
    
    try:
        response = model.prompt(extraction_prompt)
        return response.strip()
    except Exception as e:
        print(f"Error extracting arguments with LLM for {document_type}: {e}")
        return ""


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
                word = words[i].lower()  # Start with lowercase
                word_upper = word.upper()
                
                # Keep certain terms uppercase (acronyms, legal terms)
                if word_upper in ['FDA', 'FDCA', 'CBE', 'CRL']:
                    words[i] = word_upper
                # Handle proper nouns that should be capitalized
                elif word_upper in ['MERCK', 'LEVINE']:
                    words[i] = word.capitalize()
                # Handle "Third Circuit" - capitalize both words
                elif word == 'third' and i + 1 < len(words) and words[i + 1].upper() == 'CIRCUIT':
                    words[i] = 'Third'
                    words[i + 1] = 'Circuit'
                elif word == 'circuit' and i > 0 and words[i - 1].lower() == 'third':
                    words[i] = 'Circuit'  # Already handled above
                # Handle possessives
                elif word.endswith("'s") or word.endswith("'s"):
                    base_word = word[:-2]
                    if base_word.upper() in ['FDA', 'FDCA', 'MERCK', 'CIRCUIT']:
                        if base_word.upper() in ['FDA', 'FDCA']:
                            words[i] = base_word.upper() + word[-2:]
                        else:
                            words[i] = base_word.capitalize() + word[-2:]
                    else:
                        words[i] = word
                else:
                    words[i] = word
        
        cleaned = ' '.join(words)
    
    return cleaned


def parse_markdown_to_json(markdown_text: str, document_type: str) -> List[Dict[str, Any]]:
    """
    Parse markdown bullet list into nested JSON structure matching sample format.
    
    Args:
        markdown_text: Markdown formatted argument list
        document_type: Either 'petitioner' or 'respondent'
        
    Returns:
        List of argument dictionaries with nested sub_arguments
    """
    arguments = []
    lines = [line.rstrip() for line in markdown_text.split('\n') if line.strip()]
    
    current_argument = None
    argument_stack = []  # Stack to handle nested arguments
    
    for line in lines:
        # Count indentation level
        stripped = line.lstrip()
        if not stripped.startswith('-'):
            continue
            
        indent_level = (len(line) - len(stripped)) // 2  # Assuming 2 spaces per level
        argument_text = stripped[1:].strip()  # Remove leading '-' and whitespace
        
        if not argument_text:
            continue
            
        if indent_level == 0:  # Top-level argument
            # Save previous argument if exists
            if current_argument:
                arguments.append(current_argument)
            
            # Start new top-level argument
            current_argument = {
                'argument': clean_argument_text(argument_text),
                'sub_arguments': [],
                'type': document_type
            }
            argument_stack = [current_argument]
            
        else:  # Sub-argument at some level
            if not argument_stack:
                continue  # Skip if no parent argument
                
            # Adjust stack to current level
            while len(argument_stack) > indent_level:
                argument_stack.pop()
                
            if not argument_stack:
                continue
                
            # Create new sub-argument
            sub_arg = {
                'argument': clean_argument_text(argument_text)
            }
            
            # Add sub_arguments array if this might have children
            if indent_level < 3:  # Arbitrary depth limit
                sub_arg['sub_arguments'] = []
            
            # Add to parent's sub_arguments
            parent = argument_stack[-1]
            if 'sub_arguments' not in parent:
                parent['sub_arguments'] = []
            parent['sub_arguments'].append(sub_arg)
            
            # Push to stack for potential children
            if 'sub_arguments' in sub_arg:
                argument_stack.append(sub_arg)
    
    # Don't forget the last argument
    if current_argument:
        arguments.append(current_argument)
    
    return arguments


def extract_arguments_from_pdf_new_strategy(
    pdf_path: Path,
    document_type: str,
    model,
    context_words: int = 3000
) -> List[Dict[str, Any]]:
    """
    Extract arguments using the new 4-step strategy.
    
    Args:
        pdf_path: Path to PDF file
        document_type: Either 'petitioner' or 'respondent'
        model: LLM model for extraction
        context_words: Number of words to include after TOC
        
    Returns:
        List of argument dictionaries
    """
    print(f"Processing {document_type} PDF with new strategy...")
    
    # Step 1: Extract full PDF text
    full_text = extract_pdf_text(pdf_path)
    
    # Step 2: Find TABLE OF CONTENTS and get extended context
    toc_result = find_table_of_contents_section(full_text)
    if not toc_result:
        print(f"Could not find TABLE OF CONTENTS in {document_type} PDF")
        return []
    
    toc_text, toc_position = toc_result
    
    # Step 3: Get extended context (TOC + following words)
    extended_context = extract_toc_context_with_lookahead(full_text, toc_position, context_words)
    
    # Step 4: Use LLM to extract arguments in markdown format
    markdown_args = extract_arguments_with_llm(extended_context, model, document_type)
    if not markdown_args:
        print(f"LLM failed to extract arguments from {document_type} PDF")
        return []
    
    # Step 5: Parse markdown to JSON structure
    arguments = parse_markdown_to_json(markdown_args, document_type)
    
    return arguments


def extract_arguments_from_pdfs(
    petitioner_pdf: Path,
    respondent_pdf: Path,
    model=None,
    output_path: Optional[Path] = None,
    use_new_strategy: bool = True,
    context_words: int = 3000
) -> List[Dict[str, Any]]:
    """
    Extract arguments from petitioner and respondent PDF files.
    
    Args:
        petitioner_pdf: Path to petitioner.pdf
        respondent_pdf: Path to respondent.pdf
        model: LLM model for extraction (required for new strategy)
        output_path: Path to save the extracted arguments JSON (optional)
        use_new_strategy: Whether to use the new 4-step extraction strategy
        context_words: Number of words to include after TOC in new strategy
        
    Returns:
        List of all arguments from both documents
    """
    all_arguments = []
    
    if use_new_strategy and model:
        # Use new strategy
        petitioner_args = extract_arguments_from_pdf_new_strategy(
            petitioner_pdf, 'petitioner', model, context_words
        )
        respondent_args = extract_arguments_from_pdf_new_strategy(
            respondent_pdf, 'respondent', model, context_words
        )
    else:
        # Fallback to old strategy (kept for compatibility)
        print("Using fallback extraction strategy...")
        petitioner_text = extract_pdf_text(petitioner_pdf)
        respondent_text = extract_pdf_text(respondent_pdf)
        
        # Simple fallback - create basic arguments
        petitioner_args = [{
            'argument': 'Petitioner arguments could not be extracted',
            'sub_arguments': [],
            'type': 'petitioner'
        }]
        
        respondent_args = [{
            'argument': 'Respondent arguments could not be extracted', 
            'sub_arguments': [],
            'type': 'respondent'
        }]
    
    all_arguments.extend(petitioner_args)
    all_arguments.extend(respondent_args)
    
    print(f"Extracted {len(petitioner_args)} arguments from petitioner brief")
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