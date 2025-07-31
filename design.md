# Design Spec for â€˜Pedadogâ€™, a belief extractor for simulated judges

This Python library provides convenience functions for generating â€˜belief vectorsâ€™ by systematically querying â€˜judgeâ€™ models with context.

# Distributional Belief Measurement

**[DONE]** Implemented in `pedadog/thermometer.py`: 
- Main function `thermo` takes questions, context, and model list, returns belief distributions
- `monte_carlo_belief_of` performs parallel Monte Carlo sampling with configurable bounds
- `BeliefResults` class provides nested dictionary interface with statistical methods
- Rich progress bar displays time elapsed/remaining and call counts
- Used PyMuPDF instead of PyPDF2 as requested

# Case Beliefs

Implement a function in file â€˜generate\_belief\_vector.pyâ€™ to take in a pdf file called petitioner.pdf and another pdf file called respondent.pdf, read in the files, converts to text, and then output the lists arguments from each file. Use Pypdf2 to read in the pdf files. The arguments are listed on the page titled â€œTABLE OF CONTENTSâ€ under the section titled â€˜ARGUMENTâ€™. Some arguments have sub-arguments. Store them into a JSON array where each object has fields â€˜argumentâ€™ which is the string of the main argument, â€˜sub-argumentâ€™ which is a list of sub-arguments, and â€˜typeâ€™ which indicates whether the argument is from petitioner.pdf or respondent.pdf. Save this array. This function should be implemented as prompting a provided model and returning JSON.

Implement another function that takes in a prompt and a list of arguments, concatenate the prompt with each belief, and call the \`thermo\` function. The parameters of the function call include a concatenated list of prompt and arguments as the list of questions, the context is left empty, and the list of callable LM models is set to a constant called â€œDEFAULT\_LLMâ€. 

# Character Beliefs

TO be implemented in \`make\_character\_questions.py\`, and a corresponding Snakemake rule:

A txt file, â€˜moot\_rubric.txtâ€™ contains one attribute per line used to judge. Another file â€˜character\_attribute\_question.txtâ€™ contains a template for converting these attributes into questions. To do so, simply find and replace â€œATTRIBUTE\_TEXTâ€ with each line to generate a list of questions about the appellant character. Save these questions for future use.

# Experiments

## Core Functionality Test

In \`experiments/minimum-viable/\`, following the Zetteldev framework.

Create a \`llm\_student.py\` that wraps the .prompt function from pedadog.models such that the following can happen:

1. Scaffold a â€˜simple\_models.pyâ€™ with a wrapper function that produces an â€˜appellantâ€™ and â€˜judgeâ€™ model. The appellant is prompted to behave as a lawyer making an oral argument to the Supreme Court; the â€˜judgeâ€™ prompted to be Brett Kavanaugh.  
2. Generate a response from the judge given the text from the petitionerâ€™s brief.  
3. Using the brief and the judgeâ€™s response as context, run \`thermo(brief \+ response, questions, judge model)\` and save the resultant belief distributions.  
4. Given the response, have the student model generate a RESPONSE.  
5. Then judge the generated response by running \`thermo\` on the list of case belief questions and the list of character belief questions. Save the generated responses and the belief distributions.

Then in report.qmd, plot the distributions per component of the belief vectors before and after introducing the appellant response \- with separate plots for character and case.

## **Best Practices**

Below is a living â€œhow-we-workâ€ appendix.

---

### **1 Configuration â€“ `config.yaml`**

All run-time tunables **must** live in a single YAML file at the project root and be read through `pydantic-settings` (or Hydra if you later need multirun sweeps).  
 Never hard-code a magic constant in library code.

\# config.yaml  
llms:  
  default:  
    model\_name: gpt-4o-mini  
    temperature: 0.7  
    max\_tokens: 32  
sampling:  
  n\_samples: 50          \# Monte-Carlo draws per question  
  seed: 42               \# global RNG seed (propagated to numpy & torch)  
numeric\_filter:  
  regex: "^-?(?:0(?:\\\\.\\\\d+)?|1(?:\\\\.0+)?)$"  \# accept 0â€“1 floats  
  cast: float  
parallel:  
  backend: "asyncio"     \# {asyncio | ray | multiprocessing}  
  max\_concurrency: 8  
paths:  
  raw\_beliefs: "data/raw\_beliefs.parquet"  
  agg\_beliefs: "data/agg\_beliefs.parquet"  
progress:  
  style: "rich"          \# {rich | tqdm | off}  
  refresh\_every: 1.0     \# seconds  
logging:  
  level: "INFO"  
  cache\_dir: ".cache/llm"

* **Version control** â€“ commit every change to `config.yaml`; CI will fail if the file is missing.

* **Run overrides** â€“ use `ENV` variables (`PEDADOG__SAMPLING__N_SAMPLES=200 poetry run â€¦`) rather than editing the file ad-hoc.

---

### **2 Structured Output â€“ DataFrame schema**

Raw Monte-Carlo draws are stored **row-wise** in a single Parquetâ€“compressed file:

| column | dtype | description |
| ----- | ----- | ----- |
| `case_id` | string | slugified PDF basename or experiment UUID |
| `question_id` | int | index into the *questions* list |
| `question_text` | string | literal prompt text |
| `model_name` | string | e.g. `"gpt-4o-mini"` |
| `draw_idx` | int | 0 â€¦ `n_samples-1` |
| `answer` | float | numeric âˆˆ \[0, 1\] |
| `timestamp` | datetime\[UTC\] | when the call returned |
| `runtime_s` | float | wall-clock seconds for that call |

Aggregate views (mean, variance, CI, rejection rate) are **derived**, not stored.

import pandas as pd

df \= pd.read\_parquet("data/raw\_beliefs.parquet")  
agg \= (  
    df.groupby(\["case\_id", "question\_id", "model\_name"\])  
      .answer  
      .agg(\["mean", "var", "count"\])  
      .reset\_index()  
)  
agg.to\_parquet("data/agg\_beliefs.parquet", index=False)

*Never mutate existing Parquet files*; write new versions with deterministic filenames (e.g. hashed config suffix) to preserve reproducibility.

---

### **3 Testing & Continuous Integration**

| Layer | What to test | Tools / fixtures |
| ----- | ----- | ----- |
| **Unit** | â€¢ Regex filteringâ€¢ YAML loading & defaultsâ€¢ `thermo` wrapper on a *mock* LLM that returns scripted strings | `pytest`, `pytest-asyncio`, `hypothesis` |
| **Integration** | â€¢ End-to-end PDF â†’ belief DataFrame on a 1-page toy brief | Minimal PDFs under `tests/fixtures/` |
| **Property** | â€¢ Monte-Carlo draws honour the seed (identical results on rerun)â€¢ No duplicate `(case_id, question_id, draw_idx)` | `hypothesis` stateful tests |
| **Performance** | â€¢ `n_samples=50, questions=20` completes `< 60 s` on 1 GPU-free CPU core | `pytest-benchmark` |
| **CI pipeline** | GitHub Actions (\< 15 min):1. `poetry install --no-root`2\. `pytest -n auto --cov=pedadog --cov-fail-under=90`3\. `ruff check .` & `black --check .` | Cache model calls with the built-in disk cache to avoid quota burn. |

Defects caught by tests **must** fail the build; do **not** skip with `xfail` unless documented with a linked issue.

---

### **4 Reproducibility checklist**

1. **Seed** captured in `config.yaml` and logged at top-level `logger.info`.

2. **Config hash** (SHA-256 of canonical YAML) embedded in every output filename.

3. **LLM cache** keyed on `(prompt, model_name, temperature, seed)`.

4. **Environment** lockfile (`poetry.lock`) version-controlled.

5. **Data lineage** encoded in Snakemake DAG; every artifact has a rule.

---

## Implementation Status

**[COMPLETED]** All major components have been implemented:

### Core Library (`pedadog/`)
- ✅ `thermometer.py` - Distributional belief measurement with progress tracking
- ✅ `generate_belief_vector.py` - PDF argument extraction using PyMuPDF
- ✅ `make_character_questions.py` - Character attribute question generation
- ✅ `models.py` - LLM interface wrappers (Anthropic + Mock models)
- ✅ Updated package `__init__.py` with all exports

### Configuration & Settings
- ✅ `config.yaml` - Complete configuration following design spec format
- ✅ All tunables centralized in YAML with pydantic-compatible structure

### Minimum Viable Experiment (`experiments/minimum-viable/`)
- ✅ `simple_models.py` - Appellant and judge model wrappers
- ✅ `main.py` - Complete experimental pipeline implementation
- ✅ `Snakefile` - Snakemake rules for reproducible execution
- ✅ `report.qmd` - Quarto report with belief distribution visualizations
- ✅ Directory structure with `processed_data/`, `figures/`, `tests/`

### Key Features Implemented
- Parallel Monte Carlo belief sampling with Rich progress bars
- PyMuPDF-based PDF text extraction and argument parsing
- LLM-based and rule-based argument extraction methods
- Structured parquet output following DataFrame schema
- Before/after belief comparison in Supreme Court context
- Character attribute assessment framework
- Comprehensive visualization in publication-ready formats

The implementation follows all best practices specified in the design document including configuration management, structured output, testing framework integration, and reproducibility requirements.
