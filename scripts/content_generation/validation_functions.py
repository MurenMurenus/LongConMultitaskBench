from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.models.model_classes import LLMCouncil
from scripts.models.data_classes import CouncilDecision


def validate_with_council(
    council: LLMCouncil,
    original_text: str,
    candidates: List[Dict[str, Any]],
    instruction: str = None
) -> List[CouncilDecision]:
    """
    Validate answers using LLMCouncil.
    
    Args:
        council: LLMCouncil to use for validation
        original_text: Original book chapter text
        candidates: List of answers to validate
        instruction: Instruction for judges
        
    Returns:
        List of CouncilDecision objects for each candidate text
    """
    decisions = []
    
    for candidate_answer in candidates:
        try:
            decision = council.verify(
                instruction=instruction,
                reference=original_text,
                candidate=candidate_answer
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
