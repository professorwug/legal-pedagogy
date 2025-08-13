Pedadog is a plug-in module for belief vector computation in oral arguments.

To use, simply copy the 'pedadog' python module into your project. Modify the config file inside that folder to your liking (e.g. adding rubric items, modifying prompts).

(You may also have to install the dependencies in pixi.toml, chiefly `PyPDF2`)

The two main functions are in pedagog/api.py:

```
from pedagog.api import extract_arguments, belief_vector
from pedagog.models import AISandboxModel

model = AISandboxModel() #provide an API key, otherwise it's auto-loaded from .env

arguments = extract_arguments(petitioner_pdf, respondent_pdf, model) # extracts TOC from PDFs; parses it with an LLM; returns a dictionary of arguments and sub-arguments.

beliefs_judge1, beliefs_judge2... = belief_vector(arguments, case_context, judge1, judge2...)
```

Arguments are returned with form:
    [ argument1, argument2 ...]
    where each argument is a dictionary of form:
    {
        'argument': "The witnesses were unreliable because... e.g."
        'sub_arguments': [sub_argument1, sub_argument2...]
         # same form as above
        'type': "petitioner" or "respondent"
    }

Belief vectors have form:
{
    "belief 1 text": np array of sampled beliefs from judge,
    ...
}

    Belief distributions are taken both over the inputted arguments, as well as a set of 'character' beliefs about the competence, credibility, and professionality of theuser
```


