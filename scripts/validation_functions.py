from typing import List, Dict, Any
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.model_classes import LLMCouncil
from scripts.data_classes import CouncilDecision


def validate_qa_pairs(
    council: LLMCouncil,
    original_text: str,
    qa_pairs: List[Dict[str, Any]],
    prompt: str = None
) -> List[CouncilDecision]:
    """
    Validate QA pairs using LLMCouncil.
    
    Args:
        council: LLMCouncil to use for validation
        original_text: Original book chapter text
        qa_pairs: List of QA pairs to validate
        prompt: Custom prompt for validation (optional)
        
    Returns:
        List of CouncilDecision objects for each QA pair
    """
    decisions = []
    
    for qa_pair in qa_pairs:
        try:
            question = qa_pair.get("question", "")
            answer = qa_pair.get("answer", "")
            
            # Use provided prompt or create a default one
            if prompt is None:
                instruction = f"Validate if the answer is correct and supported by the reference text."
            else:
                instruction = prompt
            
            # Use council to verify the QA pair
            decision = council.verify(
                instruction=instruction,
                reference=original_text,
                candidate=f"Question: {question}\nAnswer: {answer}"
            )
            decisions.append(decision)
        except Exception as e:
            # Create a fallback decision in case of error
            fallback_decision = CouncilDecision(
                approved=False,
                votes={},
                rationale=f"Error during validation: {str(e)}"
            )
            decisions.append(fallback_decision)
    
    return decisions


def validate_structured_outputs(
    council: LLMCouncil,
    original_text: str,
    structured_outputs: List[Dict[str, Any]],
    prompt: str = None
) -> List[CouncilDecision]:
    """
    Validate structured outputs using LLMCouncil.
    
    Args:
        council: LLMCouncil to use for validation
        original_text: Original book chapter text
        structured_outputs: List of structured outputs to validate
        prompt: Custom prompt for validation (optional)
        
    Returns:
        List of CouncilDecision objects for each structured output
    """
    decisions = []
    
    for output in structured_outputs:
        try:
            # Extract the structured output text
            output_text = output.get("text", "") if isinstance(output, dict) else getattr(output, "text", "")
            
            # Use provided prompt or create a default one
            if prompt is None:
                instruction = f"Validate if the structured output correctly represents information from the reference text."
            else:
                instruction = prompt
            
            # Use council to verify the structured output
            decision = council.verify(
                instruction=instruction,
                reference=original_text,
                candidate=output_text
            )
            decisions.append(decision)
        except Exception as e:
            # Create a fallback decision in case of error
            fallback_decision = CouncilDecision(
                approved=False,
                votes={},
                rationale=f"Error during validation: {str(e)}"
            )
            decisions.append(fallback_decision)
    
    return decisions


def validate_entity_extractions(
    council: LLMCouncil,
    original_text: str,
    entity_extractions: List[Dict[str, Any]],
    prompt: str = None
) -> List[CouncilDecision]:
    """
    Validate entity extractions using LLMCouncil.
    
    Args:
        council: LLMCouncil to use for validation
        original_text: Original book chapter text
        entity_extractions: List of entity extractions to validate
        prompt: Custom prompt for validation (optional)
        
    Returns:
        List of CouncilDecision objects for each entity extraction
    """
    decisions = []
    
    for extraction in entity_extractions:
        try:
            # Extract the entities text
            entities_text = extraction.get("entities", "")
            
            # Use provided prompt or create a default one
            if prompt is None:
                instruction = f"Validate if the extracted entities are correctly identified from the reference text."
            else:
                instruction = prompt
            
            # Use council to verify the entity extraction
            decision = council.verify(
                instruction=instruction,
                reference=original_text,
                candidate=str(entities_text)
            )
            decisions.append(decision)
        except Exception as e:
            # Create a fallback decision in case of error
            fallback_decision = CouncilDecision(
                approved=False,
                votes={},
                rationale=f"Error during validation: {str(e)}"
            )
            decisions.append(fallback_decision)
    
    return decisions


def validate_summaries(
    council: LLMCouncil,
    original_text: str,
    summaries: List[Dict[str, Any]],
    prompt: str = None
) -> List[CouncilDecision]:
    """
    Validate summaries using LLMCouncil.
    
    Args:
        council: LLMCouncil to use for validation
        original_text: Original book chapter text
        summaries: List of summaries to validate
        prompt: Custom prompt for validation (optional)
        
    Returns:
        List of CouncilDecision objects for each summary
    """
    decisions = []
    
    for summary in summaries:
        try:
            # Extract the summary text
            summary_text = summary.get("summary", "")
            
            # Use provided prompt or create a default one
            if prompt is None:
                instruction = f"Validate if the summary accurately represents the main points of the reference text."
            else:
                instruction = prompt
            
            # Use council to verify the summary
            decision = council.verify(
                instruction=instruction,
                reference=original_text,
                candidate=summary_text
            )
            decisions.append(decision)
        except Exception as e:
            # Create a fallback decision in case of error
            fallback_decision = CouncilDecision(
                approved=False,
                votes={},
                rationale=f"Error during validation: {str(e)}"
            )
            decisions.append(fallback_decision)
    
    return decisions
