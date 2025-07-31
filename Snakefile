"""
Snakefile for Legal Pedagogy Project

This workflow manages experiments for analyzing case beliefs and character beliefs
in legal pedadog contexts.
"""

# Configuration
configfile: "config.yaml"

# Default rule
rule all:
    input:
        "results/case_beliefs_analysis.txt",
        "results/character_beliefs_analysis.txt"

# Case beliefs analysis
rule analyze_case_beliefs:
    input:
        "data/case_data.json"
    output:
        "results/case_beliefs_analysis.txt"
    shell:
        "pixi run python -m pedadog.case_beliefs --input {input} --output {output}"

# Character beliefs analysis  
rule analyze_character_beliefs:
    input:
        "data/character_data.json"
    output:
        "results/character_beliefs_analysis.txt"
    shell:
        "pixi run python -m pedadog.character_beliefs --input {input} --output {output}"

# Data preparation rule (example)
rule prepare_data:
    output:
        "data/case_data.json",
        "data/character_data.json"
    shell:
        "pixi run python -c 'import json; json.dump({\"example\": \"case_data\"}, open(\"data/case_data.json\", \"w\")); json.dump({\"example\": \"character_data\"}, open(\"data/character_data.json\", \"w\"))'"

# Create necessary directories
rule create_dirs:
    output:
        directory("data"),
        directory("results")
    shell:
        "mkdir -p data results"