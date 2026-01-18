from typing import List, Dict, Any
import sys
import os
import json
import random
import re

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_classes import LLM


def inject_factual_hallucination(
    llm: LLM,
    reference: str,
    prompt_template: str
) -> str:
    """
    Modify exactly one fact while preserving fluency and structure using LLM.
    
    Args:
        llm: Language model to use for generation
        reference: Original text to modify
        prompt_template: Template for the prompt with {reference} placeholder
        
    Returns:
        Text with one factual hallucination injected
    """
    # Generate the hallucinated text using the LLM
    try:
        prompt = prompt_template.format(reference=reference)
        output = llm.generate(prompt=prompt, context="")
        if output.text.strip():
            return output.text.strip()
        else:
            # Fallback if LLM doesn't return anything
            return "FAILED TO INJECT HALLUCINATIONS"
    except Exception as e:
        # Fallback to error message if there's an error
        return "FAILED TO INJECT HALLUCINATIONS"


def inject_qa_hallucination(
    llm: LLM,
    original_text: str,
    question: str,
    original_answer: str,
    prompt_template: str
) -> str:
    """
    Inject hallucinations into QA pairs using LLM.
    
    Args:
        llm: Language model to use for generation
        original_text: Original text the QA pair is based on
        question: The question
        original_answer: The original answer to the question
        prompt_template: Template for the prompt with {original_text}, {question}, and {original_answer} placeholders
        
    Returns:
        Hallucinated answer to the question
    """
    # Generate the hallucinated answer using the LLM
    try:
        prompt = prompt_template.format(
            original_text=original_text,
            question=question,
            original_answer=original_answer
        )
        output = llm.generate(prompt=prompt, context="")
        if output.text.strip():
            return output.text.strip()
        else:
            # Fallback if LLM doesn't return anything
            return "FAILED TO INJECT HALLUCINATIONS"
    except Exception as e:
        # Fallback to error message if there's an error
        return "FAILED TO INJECT HALLUCINATIONS"


def inject_structural_error(
    llm: LLM,
    reference: str,
    prompt_template: str
) -> str:
    """
    Break JSON / schema while preserving factual content using LLM.
    
    Args:
        llm: Language model to use for generation
        reference: Valid JSON string to corrupt
        prompt: Custom prompt for the LLM (optional)
        
    Returns:
        Corrupted JSON string with structural errors
    """
    # Generate the corrupted JSON using the LLM
    try:
        prompt = prompt_template.format(reference=reference)
        output = llm.generate(prompt=prompt, context="")
        if output.text.strip():
            return output.text.strip()
        else:
            # Fallback if LLM doesn't return anything
            return "FAILED TO INJECT HALLUCINATIONS"
    except Exception as e:
        # Fallback to error message if there's an error
        return "FAILED TO INJECT HALLUCINATIONS"


def inject_temporal_hallucination(
    llm: LLM,
    original_text: str,
    model_answer: str,
    prompt_template: str
) -> str:
    """
    Modify temporal information (dates, times, sequences) in the text using LLM.
    
    Args:
        llm: Language model to use for generation
        original_text: Original text the answer is based on
        model_answer: The model's answer to modify
        prompt_template: Template for the prompt with {original_text} and {model_answer} placeholders
        
    Returns:
        Text with temporal hallucination injected
    """
    # Generate the temporally hallucinated text using the LLM
    try:
        prompt = prompt_template.format(
            original_text=original_text,
            model_answer=model_answer
        )
        output = llm.generate(prompt=prompt, context="")
        if output.text.strip():
            return output.text.strip()
        else:
            # Fallback if LLM doesn't return anything
            return "FAILED TO INJECT HALLUCINATIONS"
    except Exception as e:
        # Fallback to error message if there's an error
        return "FAILED TO INJECT HALLUCINATIONS"


def inject_numerical_hallucination(
    llm: LLM,
    reference: str,
    prompt_template: str
) -> str:
    """
    Modify numerical information in the text using LLM.
    
    Args:
        llm: Language model to use for generation
        reference: Original text to modify
        prompt: Custom prompt for the LLM (optional)
        
    Returns:
        Text with numerical hallucination injected
    """
    # Generate the numerically hallucinated text using the LLM
    try:
        prompt = prompt_template.format(reference=reference)
        output = llm.generate(prompt=prompt, context="")
        if output.text.strip():
            return output.text.strip()
        else:
            # Fallback if LLM doesn't return anything
            return "FAILED TO INJECT HALLUCINATIONS"
    except Exception as e:
        # Fallback to error message if there's an error
        return "FAILED TO INJECT HALLUCINATIONS"
