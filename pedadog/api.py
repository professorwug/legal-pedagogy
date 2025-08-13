"""
Functions which expose core functionality of pedadog in two API calls

1. extract_arguments_from_pdf(pdf_petitioner, pdf_respondent): takes two pdfs as input. Calls on `extract_arguments_from_pdfs` in generate_belief_vector.py; returns a JSON object with a list of arguments in hierarchy.

2. belief_vector(arguments_json, *justice_callable): takes a hierarchical list of arguments, combines with built-in character attributes, and returns a list of 'belief distributions' per argument, per justice. 

"""
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .generate_belief_vector import extract_arguments_from_pdfs
from .models import AISandboxModel
from .thermometer import thermo, BeliefResults


def _load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)

def extract_arguments(
        pdf_petitioner:Path, 
        pdf_respondent:Path,
        context_words:int = 3000, # number of words to include after "Table of Contents". Increase of argument extraction is incomplete.
        model = None # By default uses AISandbox model
        ):
    """
    Returns arguments in a list of the following form:
    [ argument1, argument2 ...]
    where each argument is a dictionary of form:
    {
        'argument': "The witnesses were unreliable because... e.g."
        'sub_arguments': [sub_argument1, sub_argument2...]
         # same form as above
        'type': "petitioner" or "respondent"
    }
    """
    if model is None:
        model = AISandboxModel()
    all_arguments = extract_arguments_from_pdfs(
        petitioner_pdf=pdf_petitioner,
        respondent_pdf=pdf_respondent,
        model=model,
        context_words=context_words
    )
    return all_arguments

def belief_vector(
        all_arguments, # list of dictionaries in the argument format
        context, # relevant transcript with oral argument interactions
        *judge_callables, # functions with method and .prompt(context, prompt) and the built in prompts/tooling to simulate supreme court justices. We separate the context and prompt to help future caching efforts.
        path_to_character_rubric=None,
        n_samples=20,  # number of Monte Carlo samples per question
):
    """
    Returns a belief distribution for each judge, in order passed:
    with form:
    {
    "belief 1 text": np array of sampled beliefs from judge,
    ...
    }
    Belief distributions are taken both over the inputted arguments, as well as a set of 'character' beliefs about the competence, credibility, and professionality of the user.
    """

    # Load templates from config
    config = _load_config()
    prompts = config["prompts"]
    
    # Prepare questions from arguments
    questions = []

    # Extract main arguments and sub-arguments
    for arg in all_arguments:
        # Add main argument as a question
        arg_type = arg.get("type", "unknown")
        main_arg = prompts["argument_belief_template"].format(
            arg_type=arg_type,
            argument_text=arg["argument"]
        )
        questions.append(main_arg)

        # Add sub-arguments if they exist
        if arg.get("sub_arguments"):
            for sub_arg in arg["sub_arguments"]:
                if isinstance(sub_arg, dict) and "argument" in sub_arg:
                    sub_arg_text = prompts["sub_argument_belief_template"].format(
                        arg_type=arg_type,
                        sub_argument_text=sub_arg["argument"]
                    )
                    questions.append(sub_arg_text)
                elif isinstance(sub_arg, str):
                    sub_arg_text = prompts["sub_argument_belief_template"].format(
                        arg_type=arg_type,
                        sub_argument_text=sub_arg
                    )
                    questions.append(sub_arg_text)
    
    # Add character assessment questions if rubric is provided
    if path_to_character_rubric:
        try:
            # Load character rubric (text file with one attribute per line)
            rubric_path = Path(path_to_character_rubric)
            with rubric_path.open() as f:
                rubric_lines = f.readlines()
            
            # Create questions for each rubric item using config template
            character_template = prompts["character_assessment_template"]
            for line in rubric_lines:
                line = line.strip()
                if line and not line.startswith("#"):  # Skip empty lines and comments
                    # Format the question using the template from config
                    character_question = character_template.format(
                        attribute_text=line
                    )
                    questions.append(character_question)
                    
        except (FileNotFoundError, IOError) as e:
            print(f"Warning: Could not load character rubric: {e}")
            # Continue without character questions if rubric can't be loaded
    
    # If no judges provided, return empty result
    if not judge_callables:
        return []

    # Run thermometer with all judges and questions
    belief_results = thermo(
        questions=questions,
        context=context,
        models=list(judge_callables),
        n_samples=n_samples,
        min_val=0.0,
        max_val=1.0
    )

    # Convert BeliefResults to the expected format
    # Return a list of dictionaries, one per judge
    belief_distribution_list = []

    for judge in judge_callables:
        judge_name = getattr(judge, "name", str(judge))
        judge_beliefs = {}

        # Get all results for this judge
        judge_results = belief_results.get_model_results(judge_name)

        # Convert each question's distribution to numpy array
        for question, distribution in judge_results.items():
            # Get the numeric values as a numpy array
            values = np.array(distribution.values)
            judge_beliefs[question] = values

        belief_distribution_list.append(judge_beliefs)

    return belief_distribution_list
