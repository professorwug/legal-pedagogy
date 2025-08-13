"""
Main experiment implementation for minimum viable pedadog experiment.

This script implements the full pipeline described in the design spec:
1. Extract arguments from petitioner and respondent briefs
2. Generate judge response to petitioner brief
3. Measure initial belief distributions
4. Generate appellant response to judge
5. Measure final belief distributions for case and character beliefs
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from snakemake.script import snakemake

from pedadog import (
    generate_belief_vector_from_pdfs,
    create_character_questions_file,
    load_character_questions,
    get_question_texts,
    thermo
)
from simple_models import get_models, create_judge_model


def load_config() -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = Path(__file__).parent.parent.parent / "pedadog" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_pdf_texts(config: Dict[str, Any]) -> tuple[str, str]:
    """Load and truncate PDF texts for context."""
    from pedadog.generate_belief_vector import extract_pdf_text
    
    # Get paths from config or snakemake
    if hasattr(snakemake, 'input'):
        petitioner_path = Path(snakemake.input.petitioner_pdf)
        respondent_path = Path(snakemake.input.respondent_pdf)
    else:
        petitioner_path = Path(config['paths']['case_pdfs']['petitioner'])
        respondent_path = Path(config['paths']['case_pdfs']['respondent'])
    
    petitioner_text = extract_pdf_text(petitioner_path)
    respondent_text = extract_pdf_text(respondent_path)
    
    # Truncate for context
    limit = config['experiments']['minimum_viable']['brief_context_limit']
    petitioner_text = petitioner_text[:limit]
    respondent_text = respondent_text[:limit]
    
    return petitioner_text, respondent_text


def generate_judge_response(petitioner_text: str, config: Dict[str, Any]) -> str:
    """Generate judge response to petitioner brief."""
    judge_model = create_judge_model(config)
    
    prompt = f"""
    Based on this petitioner's brief, please provide your judicial response and questions 
    as Justice Kavanaugh would during oral arguments:
    
    {petitioner_text}
    """
    
    return judge_model.prompt(prompt)


def generate_appellant_response(judge_response: str, petitioner_text: str, config: Dict[str, Any]) -> str:
    """Generate appellant response to judge's questions."""
    appellant_model, _ = get_models()
    
    prompt = f"""
    You are the lawyer for the petitioner. The judge has responded with questions/comments 
    about your brief. Please provide your response addressing the judge's concerns.
    
    Your original brief excerpt:
    {petitioner_text}
    
    Judge's response:
    {judge_response}
    
    Your response:
    """
    
    return appellant_model.prompt(prompt)


def create_sample_data_files(config: Dict[str, Any]):
    """Create sample data files if they don't exist (for testing)."""
    
    # Create sample rubric file
    rubric_path = Path(config['paths']['character_files']['rubric'])
    if not rubric_path.exists():
        rubric_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rubric_path, 'w') as f:
            f.write("judicial temperament\n")
            f.write("legal reasoning ability\n")
            f.write("advocacy effectiveness\n")
            f.write("constitutional knowledge\n")
            f.write("precedent awareness\n")
    
    # Create sample question template
    template_path = Path(config['paths']['character_files']['question_template'])
    if not template_path.exists():
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with open(template_path, 'w') as f:
            f.write("On a scale of 0 to 1, how would you rate the appellant's ATTRIBUTE_TEXT based on this interaction?")


def run_experiment():
    """Run the complete minimum viable experiment."""
    print("Starting minimum viable pedadog experiment...")
    
    # Load configuration
    config = load_config()
    
    # Create sample data files for testing
    create_sample_data_files(config)
    
    # Set up output paths
    results_dir = Path("processed_data")
    results_dir.mkdir(exist_ok=True)
    
    # Initialize models
    print("Initializing models...")
    appellant_model, judge_model = get_models()
    
    try:
        # Step 1: Load PDF texts (for context - in real experiment would extract arguments)
        print("Loading PDF texts...")
        petitioner_text, respondent_text = load_pdf_texts(config)
        
        # For now, create mock arguments since we may not have actual PDFs
        print("Creating mock case arguments...")
        mock_arguments = [
            {
                "argument": "The constitutional provision requires strict scrutiny analysis",
                "sub_arguments": ["Previous precedent supports this interpretation", "The text is unambiguous"],
                "type": "petitioner"
            },
            {
                "argument": "The lower court decision should be reversed",
                "sub_arguments": ["Procedural errors were made", "Evidence was insufficient"],
                "type": "petitioner"
            },
            {
                "argument": "The statute is constitutional as written",
                "sub_arguments": ["Legislative intent is clear", "No fundamental rights are violated"],
                "type": "respondent"
            }
        ]
        
        # Save arguments
        with open(results_dir / "extracted_arguments.json", 'w') as f:
            json.dump(mock_arguments, f, indent=2)
        
    except Exception as e:
        print(f"Warning: Could not load PDFs ({e}), using mock arguments")
        # Use mock arguments and texts
        petitioner_text = "Sample petitioner brief arguing constitutional violation..."
        respondent_text = "Sample respondent brief defending statute..."
        
        mock_arguments = [
            {
                "argument": "The constitutional provision requires strict scrutiny analysis",
                "sub_arguments": ["Previous precedent supports this interpretation"],
                "type": "petitioner"
            },
            {
                "argument": "The statute is constitutional as written", 
                "sub_arguments": ["Legislative intent is clear"],
                "type": "respondent"
            }
        ]
        
        with open(results_dir / "extracted_arguments.json", 'w') as f:
            json.dump(mock_arguments, f, indent=2)
    
    # Step 2: Generate judge response to petitioner brief
    print("Generating judge response...")
    judge_response = generate_judge_response(petitioner_text, config)
    
    with open(results_dir / "judge_response.txt", 'w') as f:
        f.write(judge_response)
    
    # Step 3: Measure initial belief distributions (before appellant response)
    print("Measuring initial belief distributions...")
    case_questions = [arg["argument"] for arg in mock_arguments]
    brief_context = f"Petitioner brief: {petitioner_text}\n\nJudge response: {judge_response}"
    
    initial_beliefs = thermo(
        questions=case_questions,
        context=brief_context,
        models=[judge_model],
        n_samples=config['sampling']['n_samples']
    )
    
    # Step 4: Generate appellant response
    print("Generating appellant response...")
    appellant_response = generate_appellant_response(judge_response, petitioner_text, config)
    
    with open(results_dir / "appellant_response.txt", 'w') as f:
        f.write(appellant_response)
    
    # Step 5: Generate character questions
    print("Generating character questions...")
    rubric_path = Path(config['paths']['character_files']['rubric'])
    template_path = Path(config['paths']['character_files']['question_template'])
    questions_path = Path(config['paths']['character_files']['generated_questions'])
    
    character_questions_data = create_character_questions_file(
        rubric_path, template_path, questions_path
    )
    character_questions = get_question_texts(character_questions_data)
    
    # Step 6: Measure final belief distributions
    print("Measuring final belief distributions...")
    
    # Case beliefs with appellant response included
    final_context = f"{brief_context}\n\nAppellant response: {appellant_response}"
    
    final_case_beliefs = thermo(
        questions=case_questions,
        context=final_context,
        models=[judge_model],
        n_samples=config['sampling']['n_samples']
    )
    
    # Character beliefs about appellant
    character_context = f"Based on this legal interaction:\n\n{final_context}"
    
    character_beliefs = thermo(
        questions=character_questions,
        context=character_context,
        models=[judge_model],
        n_samples=config['sampling']['n_samples']
    )
    
    # Step 7: Save results to structured format
    print("Saving results...")
    
    def beliefs_to_dataframe(belief_results, experiment_stage, belief_type):
        """Convert belief results to DataFrame format."""
        rows = []
        
        for model_name in belief_results.model_names:
            for question in belief_results.questions:
                distribution = belief_results.get(model_name, question)
                if distribution:
                    for i, response in enumerate(distribution.responses):
                        if response.numeric_value is not None:
                            rows.append({
                                'case_id': 'minimum_viable_experiment',
                                'question_id': hash(question) % 10000,  # Simple hash for ID
                                'question_text': question,
                                'model_name': model_name,
                                'draw_idx': i,
                                'answer': response.numeric_value,
                                'timestamp': response.timestamp,
                                'runtime_s': response.runtime_s,
                                'experiment_stage': experiment_stage,
                                'belief_type': belief_type
                            })
        
        return pd.DataFrame(rows)
    
    # Combine all results
    all_results = []
    
    # Initial case beliefs
    initial_df = beliefs_to_dataframe(initial_beliefs, 'initial', 'case')
    all_results.append(initial_df)
    
    # Final case beliefs  
    final_df = beliefs_to_dataframe(final_case_beliefs, 'final', 'case')
    all_results.append(final_df)
    
    # Character beliefs
    character_df = beliefs_to_dataframe(character_beliefs, 'final', 'character')
    all_results.append(character_df)
    
    # Combine and save
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_parquet(results_dir / "belief_distributions.parquet", index=False)
    
    # Also save as CSV for easier inspection
    combined_df.to_csv(results_dir / "belief_distributions.csv", index=False)
    
    print(f"Experiment completed! Results saved to {results_dir}")
    print(f"Total belief measurements: {len(combined_df)}")
    print(f"Models used: {combined_df['model_name'].unique()}")
    print(f"Belief types: {combined_df['belief_type'].unique()}")


if __name__ == "__main__":
    run_experiment()