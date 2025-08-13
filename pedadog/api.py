"""
Functions which expose core functionality of pedadog in two API calls

1. extract_arguments_from_pdf(pdf_petitioner, pdf_respondent): takes two pdfs as input. Calls on `extract_arguments_from_pdfs` in generate_belief_vector.py; returns a JSON object with a list of arguments in hierarchy.

2. belief_vector(arguments_json, *justice_callable): takes a hierarchical list of arguments, combines with built-in character attributes, and returns a list of 'belief distributions' per argument, per justice. 

"""
import json
from pathlib import Path
from typing import List,Dict, Any, Optional

from .generate_belief_vector import extract_arguments_from_pdfs
from .models import AISandboxModel

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
    
    return belief_distribution_list
