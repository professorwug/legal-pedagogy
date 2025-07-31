import os
import requests
import json
import base64
from mimetypes import guess_type
from openai import AzureOpenAI
from tqdm import trange, tqdm
from dotenv import load_dotenv
load_dotenv()

# Princeton's AI Sandbox - Azure endpoint for generation
sandbox_api_key=os.environ['AI_SANDBOX_KEY']

# Set the URL of the AI Sandbox API
sandbox_endpoint="https://api-ai-sandbox.princeton.edu/"

# See the following Microsoft page describing API versions
# https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation
sandbox_api_version="2025-03-01-preview"

# Set the model deployment name that the prompt should be sent to
available_models = [
    "o3-mini",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-35-turbo-16k",
    "Meta-Llama-3-1-70B-Instruct-htzs",
    "Meta-Llama-3-1-8B-Instruct-nwxcg",
    "Mistral-small-zgjes"
    ]

# Base 64 encode local image and return text to be included in JSON request
def local_image_to_data_url(image_path):
    """
    Get the url of a local image
    """
    mime_type, _ = guess_type(image_path)

    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"

# This function will submit a simple text prompt to the chosen model
def sandbox_llm(prompt, system_prompt = "You are a creative writing assistant. Complete the story in a compelling way.", temperature = 0.7, top_p=0.5,  max_tokens = 4096, model_to_be_used='gpt-4o'):
    # Establish a connection to your Azure OpenAI instance
    client = AzureOpenAI(
        api_key=sandbox_api_key,
        azure_endpoint = sandbox_endpoint,
        api_version=sandbox_api_version # current api version not in preview
    )

    response = client.chat.completions.create(
        model=model_to_be_used,
        temperature=temperature, # temperature = how creative/random the model is in generating response - 0 to 1 with 1 being most creative
        max_tokens=max_tokens, # max_tokens = token limit on context to send to the model - no stories should realistically exceed this
        top_p=top_p, # top_p = diversity of generated text by the model considering probability attached to token - 0 to 1 - ex. top_p of 0.1 = only tokens within the top 10% probability are considered
        messages=[
            {"role": "system", "content": system_prompt}, # describes model identity and purpose
            {"role": "user", "content": prompt}, # user prompt
        ]
    )
    return response.choices[0].message.content

def prompt_o3_mini(prompt, system_prompt="", model_to_be_used="o3-mini")->str:
    """
    Submit a text prompt to the chosen model with retry logic and exponential backoff.

    Args:
        prompt: The user prompt to send to the model
        system_prompt: Describes the model identity and purpose
        temperature: How creative/random the model is in generating response (0 to 1)
        top_p: Diversity parameter for token selection during generation (0 to 1)
        max_tokens: Token limit on context to send to the model
        model_to_be_used: The model to use for generation

    Returns:
        The generated text response as a string
    """
    import time
    import random
    import logging

    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    max_retries = 5
    base_delay = 1  # Base delay in seconds

    # Establish a connection to Azure OpenAI
    client = AzureOpenAI(
        api_key=sandbox_api_key,
        azure_endpoint=sandbox_endpoint,
        api_version=sandbox_api_version
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_to_be_used,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            # Calculate backoff delay with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)

            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} attempts failed. Last error: {str(e)}")
                raise Exception(f"Failed to get response after {max_retries} attempts") from e


def llm_sample_to_length(llm, prompt, target_length, tolerance=0.1, max_tries=10, max_api_errors=3):
    """
    Samples repeatedly from an LLM until receiving output within tolerance% of target_length.
    Handles API errors gracefully with retries.

    Args:
        llm: Function that takes a prompt and returns generated text
        prompt: The input prompt to send to the LLM
        target_length: The desired length of the output text (in characters)
        tolerance: Acceptable deviation from target length as a fraction (default: 0.1 or 10%)
        max_tries: Maximum number of generation attempts (default: 10)
        max_api_errors: Maximum number of API errors to tolerate before giving up (default: 3)

    Returns:
        The generated text within the target length constraints, or the closest match
        after max_tries attempts. Returns None if max_api_errors is exceeded.
    """
    import time
    import logging
    logger = logging.getLogger(__name__)

    min_length = target_length * (1 - tolerance)
    max_length = target_length * (1 + tolerance)

    closest_text = None
    closest_distance = float('inf')
    consecutive_errors = 0

    for attempt in trange(max_tries):
        try:
            # Generate text using the provided LLM function
            generated_text = llm(prompt)

            # Reset error counter on successful API call
            consecutive_errors = 0

            # If the LLM returns None or empty string, skip this attempt
            if not generated_text:
                logger.warning(f"Attempt {attempt+1}: LLM returned empty response, retrying...")
                continue

            # Calculate the length of the generated text
            text_length = len(generated_text.split(" "))

            # Check if the text is within the acceptable range
            if min_length <= text_length <= max_length:
                logger.info(f"Success on attempt {attempt+1}: Generated text with length {text_length}")
                return generated_text

            # If not within range, keep track of the closest match
            distance = abs(text_length - target_length)
            if distance < closest_distance:
                closest_distance = distance
                closest_text = generated_text

            print(f"Attempt {attempt+1}: Generated text length {text_length}, "
                  f"target range {min_length}-{max_length}")

        except Exception as e:
            consecutive_errors += 1
            logger.error(f"API error on attempt {attempt+1}: {str(e)}")

            if consecutive_errors >= max_api_errors:
                logger.error(f"Exceeded maximum consecutive API errors ({max_api_errors}), aborting.")
                return closest_text

            # Exponential backoff for API errors
            backoff_time = 2 ** consecutive_errors
            logger.info(f"Backing off for {backoff_time} seconds before retry...")
            time.sleep(backoff_time)

    # If we've exhausted all attempts, return the closest match
    if closest_text:
        logger.info(f"Failed to generate text within tolerance after {max_tries} attempts. "
              f"Returning closest match with length {len(closest_text)}")
        return closest_text
    else:
        logger.error("No successful generations. Returning None.")
        return None


# %% Testing the AI Sandbox