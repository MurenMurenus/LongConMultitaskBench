import json
import uuid
from typing import List, Dict, Any
import pandas as pd
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from separate files
from scripts.data_classes import LLMOutput, CouncilDecision
from scripts.read_booksum_data import read_booksum_data
from scripts.model_classes import LLM, PlaceholderLLM, LLMCouncil, HuggingFaceLLM


# -------------------------
# Functions for ground-truth generationß
# -------------------------

def generate_qa_pairs(
    llms: List[LLM],
    text: str,
    n: int,
    prompt_template: str = None
) -> List[Dict[str, Any]]:
    """
    Generate question-answer pairs using language models.
    
    Args:
        llms: List of language models to use for generation
        text: Context text to generate QA pairs from
        n: Number of QA pairs to generate
        
    Returns:
        List of dictionaries containing model name, question, and answer
    """
    prompt_template = prompt_template if prompt_template else (
        "Generate a question and a comprehensive answer in the following format:\n"
        "Question: generated_question Answer: generated_comprehensive_answer\n"
        "based on the following text:\n\n"
        "{text}\n\n"
        "Question:"
    )
    qa_pairs = []
    
    for llm in llms[:n]:
        # Format the prompt with the text
        prompt = prompt_template.format(text=text)
        
        # Generate response using the LLM
        try:
            llm_output = llm.generate(prompt=prompt, context="")
            # Parse the output to extract question and answer
            response_text = llm_output.text
            if "Answer:" in response_text:
                parts = response_text.split("Answer:")
                question = parts[0].split("Question:", "")[1].strip()
                answer = parts[1].strip()
            else:
                question = "NO Answer: IN RESPONSE_TEXT"
                answer = "NO Answer: IN RESPONSE_TEXT"
            
            qa_pairs.append({
                "model": llm.name,
                "question": question,
                "answer": answer
            })
        except Exception as e:
            # Fallback in case of error
            qa_pairs.append({
                "model": llm.name,
                "question": f"Failed to generate question: {str(e)}",
                "answer": "Failed to generate answer due to error."
            })
    
    return qa_pairs


def generate_structured_outputs(
    llms: List[LLM],
    text: str,
    n: int,
    prompt_template: str = None,
) -> List[LLMOutput]:
    """
    Generate structured outputs using language models.
    
    Args:
        llms: List of language models to use for generation
        text: Context text to generate structured output from
        prompt: Prompt template for structured output generation
        n: Number of outputs to generate
        
    Returns:
        List of LLMOutput objects with structured outputs
    """
    prompt_template = prompt_template if prompt_template else (
        "Extract all relations between characters in JSON format:\n"
        "{characters: [{name, relations: [{target, type}]}]}"
    )
    outputs = []
    
    for llm in llms[:n]:
        try:
            # Generate structured output
            output = llm.generate(prompt=prompt_template, context=text)
            outputs.append(output)
        except Exception as e:
            # Fallback in case of error
            fallback_output = LLMOutput(
                model_name=llm.name,
                text=f"{{\"error\": \"Failed to generate structured output: {str(e)}\"}}",
                metadata={"error": str(e)}
            )
            outputs.append(fallback_output)
    
    return outputs


def validate_json_structure(json_string: str) -> bool:
    """
    Validate if a string is valid JSON.
    
    Args:
        json_string: String to validate
        
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False


def extract_entities_from_text(
    llms: List[LLM],
    text: str,
    n: int,
    prompt_template: str = None
) -> List[Dict[str, Any]]:
    """
    Extract named entities from text using language models.
    
    Args:
        llms: List of language models to use for generation
        text: Context text to extract entities from
        n: Number of entity extractions to generate
        
    Returns:
        List of dictionaries containing model name and extracted entities
    """
    prompt_template = prompt_template if prompt_template else (
        "Extract all named entities (people, places, organizations, dates) from the following text "
        "and format them as JSON:\n\n"
        "{text}\n\n"
        "Format the output as:\n"
        "{{\"entities\": [{{\"type\": \"entity_type\", \"name\": \"entity_name\"}}]}}"
    )
    
    entities_list = []
    
    for llm in llms[:n]:
        try:
            prompt = prompt_template.format(text=text)
            output = llm.generate(prompt=prompt, context="")
            
            # Try to parse as JSON, if not valid, try to regenerate
            if validate_json_structure(output.text):
                entities = json.loads(output.text)
            else:
                output = llm.generate(prompt=prompt, context="")
                if validate_json_structure(output.text):
                    entities = json.loads(output.text)
                else:
                    entities = {"error": "Failed to generate valid json"}
                
            entities_list.append({
                "model": llm.name,
                "entities": entities,
                "metadata": output.metadata
            })
        except Exception as e:
            entities_list.append({
                "model": llm.name,
                "entities": {"error": f"Failed to extract entities: {str(e)}"},
                "metadata": {"error": str(e)}
            })
    
    return entities_list


def generate_summaries(
    llms: List[LLM],
    text: str,
    n: int,
    prompt_template: str = None
) -> List[Dict[str, Any]]:
    """
    Generate summaries of text using language models.
    
    Args:
        llms: List of language models to use for generation
        text: Context text to summarize
        n: Number of summaries to generate
        
    Returns:
        List of dictionaries containing model name and summary
    """
    prompt_template = prompt_template if prompt_template else (
        "Provide a concise summary of the following text:\n\n"
        "{text}\n\n"
        "Summary:"
    )
    
    summaries = []
    
    for llm in llms[:n]:
        try:
            prompt = prompt_template.format(text=text)
            output = llm.generate(prompt=prompt, context="")
            
            summaries.append({
                "model": llm.name,
                "summary": output.text,
                "metadata": output.metadata
            })
        except Exception as e:
            summaries.append({
                "model": llm.name,
                "summary": f"Failed to generate summary: {str(e)}",
                "metadata": {"error": str(e)}
            })
    
    return summaries
