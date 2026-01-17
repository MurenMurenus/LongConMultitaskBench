import json
import uuid
from typing import List, Dict, Any
import pandas as pd
import sys
import os

from tqdm import tqdm

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from separate files
from scripts.data_classes import LLMOutput, CouncilDecision
from scripts.ground_truth_functions import extract_entities_from_text, generate_qa_pairs, generate_structured_outputs, generate_summaries
from scripts.hallucination_functions import (
    inject_factual_hallucination, 
    inject_structural_error,
    inject_temporal_hallucination,
    inject_numerical_hallucination
)
from scripts.prompts import (
    qa_prompt_template,
    structured_prompt_template,
    entity_extraction_prompt_template,
    summary_prompt_template
)
from scripts.read_booksum_data import read_booksum_data
from scripts.model_classes import LLM, PlaceholderLLM, LLMCouncil, HuggingFaceLLM


# -------------------------
# Benchmark row builder
# -------------------------
def build_benchmark_row(
    chapter_id: str,
    chapter_text: str,
    llms: List[LLM],
    council: LLMCouncil,
    qa_prompt_template: str,
    structured_prompt_template: str,
    entity_extraction_prompt_template: str,
    summary_prompt_template: str
) -> Dict[str, Any]:
    
    # Generate different types of outputs
    qa_pairs = generate_qa_pairs(llms, chapter_text, n=5, prompt_template=qa_prompt_template)
    structured_outputs = generate_structured_outputs(llms, chapter_text, n=5, prompt_template=structured_prompt_template)
    entity_extractions = extract_entities_from_text(llms, chapter_text, n=3, prompt_template=entity_extraction_prompt_template)
    summaries = generate_summaries(llms, chapter_text, n=3, prompt_template=summary_prompt_template)
    
    # Inject hallucinations in answers from qa_pairs
    hallucinated_answers = [
        inject_factual_hallucination(llms[0], qa["answer"], "Change one fact")
        for qa in qa_pairs
    ]
    broken_structures = [
        inject_structural_error(llms[0], s.text)
        for s in structured_outputs[:3]
    ]
    temporal_hallucinations = [
        inject_temporal_hallucination(llms[0], summary["summary"])
        for summary in summaries
    ]
    
    # Generate numerical hallucinations for entity extractions
    numerical_hallucinations = []
    for entity_extraction in entity_extractions:
        if "entities" in entity_extraction and isinstance(entity_extraction["entities"], dict):
            # Convert entities to text for hallucination
            entities_text = json.dumps(entity_extraction["entities"])
            hallucinated_entities = inject_numerical_hallucination(llms[0], entities_text)
            numerical_hallucinations.append(hallucinated_entities)
        else:
            numerical_hallucinations.append("No entities to hallucinate")

    return {
        "benchmark_id": str(uuid.uuid4()),
        "chapter_id": chapter_id,
        "original_text": chapter_text,

        # QA
        "qa_prompt_template": qa_prompt_template,
        "qa_pairs": qa_pairs,
        "hallucinated_answers": hallucinated_answers,

        # Structured extraction
        "structured_prompt_template": structured_prompt_template,
        "structured_outputs": [s for s in structured_outputs],
        "structural_errors": broken_structures,

        # Entity extraction
        "entity_extraction_prompt_template": entity_extraction_prompt_template,
        "entity_extractions": entity_extractions,
        "numerical_hallucinations": numerical_hallucinations,

        # Summaries
        "summary_prompt_template": summary_prompt_template,
        "summaries": summaries,
        "temporal_hallucinations": temporal_hallucinations,
    }


# -------------------------
# Dataset builder
# -------------------------

def build_benchmark_dataset(
    texts: List[Dict[str, str]],
    llms: List[LLM],
    council: LLMCouncil,
    qa_prompt_template: str,
    structured_prompt_template: str,
    entity_extraction_prompt_template: str,
    summary_prompt_template: str
) -> pd.DataFrame:

    rows = []
    for item in tqdm(texts):
        row = build_benchmark_row(
            chapter_id=item["chapter_id"],
            chapter_text=item["chapter_text"],
            llms=llms,
            council=council,
            qa_prompt_template=qa_prompt_template,
            structured_prompt_template=structured_prompt_template,
            entity_extraction_prompt_template=entity_extraction_prompt_template,
            summary_prompt_template=summary_prompt_template
        )
        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------
# Benchmark generation pipeline
# -------------------------
if __name__ == "__main__":

    # hf_llm = HuggingFaceLLM(
    #     name="Qwen3-4B-Instruct-2507", 
    #     model_name="local_models/Qwen3-4B-Instruct-2507"
    # )

    hf_llm = PlaceholderLLM("Qwen3-4B-Instruct-2507")
    generation_llms = [hf_llm]  # can be multiple copies of same llm, if we want many generations from same model
    print("MODELS FOR GENERATION INITIALIZED")

    council = LLMCouncil(judges=[PlaceholderLLM("gpt-4"), PlaceholderLLM("claude")])
    print("LLM COUNCIL INITIALIZED")
    
    booksum_data = read_booksum_data(file_path="data/benchmark_base_chapters_example.json")
    print("BENCHMARK BASIS LOADED")

    print("BUILDING BENCHMARK DATASET")
    df_generated = build_benchmark_dataset(
        texts=booksum_data,
        llms=generation_llms,
        council=council,
        qa_prompt_template=qa_prompt_template,
        structured_prompt_template=structured_prompt_template,
        entity_extraction_prompt_template=entity_extraction_prompt_template,
        summary_prompt_template=summary_prompt_template
    )

    print("SAVING BENCHMARK DATASET")
    df_generated.to_json("data/LongConMultitaskBenchmark.json", orient="records", lines=True)
    print("BENCHMARK DATASET SAVED")
