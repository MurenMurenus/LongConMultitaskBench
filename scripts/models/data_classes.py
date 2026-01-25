import uuid
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class LLMOutput:
    model_name: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class CouncilDecision:
    approved: bool
    votes: Dict[str, bool]
    rationales: Dict[str, bool]


@dataclass
class BenchmarkRow:
    benchmark_id: str
    chapter_id: str
    original_text: str
    qa_pairs: List[Dict[str, Any]]
    hallucinated_answers: List[str]
    structured_prompt: str
    structured_outputs: List[Dict[str, Any]]
    structural_errors: List[str]
    council_validation: Any
    perplexity_scores: Any
    textual_similarity_scores: Any
