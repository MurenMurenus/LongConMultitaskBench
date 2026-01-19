import json
import uuid
from typing import List, Dict, Any
import pandas as pd
import sys
import os

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ground_truth_functions import extract_entities_from_text, generate_qa_pairs, generate_structured_outputs, generate_summaries
from scripts.hallucination_functions import (
    inject_structural_error,
    inject_temporal_hallucination,
    inject_numerical_hallucination,
    inject_qa_hallucination
)
from scripts.validation_functions import validate_with_council
from scripts.prompts import (
    qa_prompt_template,
    structured_prompt_template,
    entity_extraction_prompt_template,
    summary_prompt_template,
    inject_factual_hallucination_prompt_template,
    inject_structural_error_prompt_template,
    inject_temporal_hallucination_prompt_template,
    inject_numerical_hallucination_prompt_template,
    inject_qa_hallucination_prompt_template,
    validate_qa_pairs_council_prompt,
    validate_structured_outputs_council_prompt,
    validate_entity_extractions_council_prompt,
    validate_summaries_council_prompt
)
from scripts.read_booksum_data import read_booksum_data
from scripts.model_classes import LLM, PlaceholderLLM, LLMCouncil, HuggingFaceLLM, OpenAILLM


# -------------------------
# Benchmark row builder
# -------------------------
def build_benchmark_row(
    chapter_id: str,
    chapter_text: str,
    llms: List[LLM],
    council: LLMCouncil
) -> Dict[str, Any]:
    
    # ---
    # Generate ground truth outputs
    # ---
    # QA
    print("Generating QA pairs...")
    qa_pairs = generate_qa_pairs(llms, chapter_text, n=5, prompt_template=qa_prompt_template)
    print(f"Generated {len(qa_pairs)} QA pairs")
    print("Validating QA pairs...")
    qa_validations = validate_with_council(
        council=council, 
        original_text=chapter_text, 
        candidates=qa_pairs, 
        instruction=validate_qa_pairs_council_prompt
    )
    print(f"Validated {len(qa_validations)} QA pairs")
    
    # Structured output
    print("Generating structured outputs...")
    structured_outputs = generate_structured_outputs(llms, chapter_text, n=5, prompt_template=structured_prompt_template)
    print(f"Generated {len(structured_outputs)} structured outputs")
    print("Validating structured outputs...")
    structured_validations = validate_with_council(
        council=council, 
        original_text=chapter_text, 
        candidates=structured_outputs, 
        instruction=validate_structured_outputs_council_prompt
    )
    print(f"Validated {len(structured_validations)} structured outputs")
    
    # Entity extraction
    print("Extracting entities...")
    entity_extractions = extract_entities_from_text(llms, chapter_text, n=3, prompt_template=entity_extraction_prompt_template)
    print(f"Extracted entities from {len(entity_extractions)} samples")
    print("Validating entity extractions...")
    entity_validations = validate_with_council(
        council=council, 
        original_text=chapter_text, 
        candidates=entity_extractions, 
        instruction=validate_entity_extractions_council_prompt
    )
    print(f"Validated {len(entity_validations)} entity extractions")
    
    # Summaries
    print("Generating summaries...")
    summaries = generate_summaries(llms, chapter_text, n=3, prompt_template=summary_prompt_template)
    print(f"Generated {len(summaries)} summaries")
    print("Validating summaries...")
    summary_validations = validate_with_council(
        council=council, 
        original_text=chapter_text, 
        candidates=summaries, 
        instruction=validate_summaries_council_prompt
    )
    print(f"Validated {len(summary_validations)} summaries")
    
    # ---
    # Inject hallucinations to answers from qa_pairs
    # ---
    print("Injecting QA hallucinations...")
    hallucinated_answers = [
        inject_qa_hallucination(
            llms[0],
            chapter_text,
            qa["question"],
            qa["answer"],
            prompt_template=inject_qa_hallucination_prompt_template
        )
        for qa in qa_pairs
    ]
    print(f"Injected hallucinations in {len(hallucinated_answers)} answers")
    
    print("Injecting structural errors...")
    broken_structures = [
        inject_structural_error(llms[0], s.text, prompt_template=inject_structural_error_prompt_template)
        for s in structured_outputs
    ]
    print(f"Injected structural errors in {len(broken_structures)} outputs")
    
    print("Injecting temporal hallucinations...")
    temporal_hallucinations = [
        inject_temporal_hallucination(
            llms[0], 
            chapter_text, 
            summary["summary"], 
            prompt_template=inject_temporal_hallucination_prompt_template
        )
        for summary in summaries
    ]
    print(f"Injected temporal hallucinations in {len(temporal_hallucinations)} summaries")
    
    print("Injecting numerical hallucinations...")
    numerical_hallucinations = []
    for entity_extraction in entity_extractions:
        if "entities" in entity_extraction and isinstance(entity_extraction["entities"], dict):
            entities_text = json.dumps(entity_extraction["entities"])
            hallucinated_entities = inject_numerical_hallucination(llms[0], entities_text, prompt_template=inject_numerical_hallucination_prompt_template)
            numerical_hallucinations.append(hallucinated_entities)
        else:
            numerical_hallucinations.append("No entities to hallucinate")
    print(f"Injected numerical hallucinations in {len(numerical_hallucinations)} entity extractions")

    return {
        "benchmark_id": str(uuid.uuid4()),
        "chapter_id": chapter_id,
        "original_text": chapter_text,

        # QA
        "qa_prompt_template": qa_prompt_template,
        "qa_pairs": qa_pairs,
        "qa_validations": qa_validations,
        "qa_hallucinations": hallucinated_answers,

        # Structured extraction
        "structured_prompt_template": structured_prompt_template,
        "structured_outputs": [s for s in structured_outputs],
        "structured_validations": structured_validations,
        "structured_hallucinations": broken_structures,

        # Entity extraction
        "entity_extraction_prompt_template": entity_extraction_prompt_template,
        "entity_extractions": entity_extractions,
        "entity_validations": entity_validations,
        "entity_hallucinations": numerical_hallucinations,

        # Summaries
        "summary_prompt_template": summary_prompt_template,
        "summaries": summaries,
        "summary_validations": summary_validations,
        "summary_hallucinations": temporal_hallucinations,
    }


# -------------------------
# Dataset builder
# -------------------------
def build_benchmark_dataset(
    texts: List[Dict[str, str]],
    llms: List[LLM],
    council: LLMCouncil
) -> pd.DataFrame:

    rows = []
    for item in tqdm(texts):
        row = build_benchmark_row(
            chapter_id=item["chapter_id"],
            chapter_text=item["chapter_text"],
            llms=llms,
            council=council
        )
        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------
# Benchmark generation pipeline
# -------------------------
if __name__ == "__main__":
    print("INITIALIZING MODELS FOR GENERATION")
    # hf_llm = HuggingFaceLLM(
    #     name="Qwen3-4B-Instruct",
    #     model_name="local_models/Qwen3-4B-Instruct"
    # )
    hf_llm1 = HuggingFaceLLM(
        name="Llama-3.2-1B-Instruct-1",
        model_name="local_models/Llama-3.2-1B-Instruct"
    )
    hf_llm2 = HuggingFaceLLM(
        name="Llama-3.2-1B-Instruct-2",
        model_name="local_models/Llama-3.2-1B-Instruct"
    )

    # hf_llm = PlaceholderLLM("Qwen3-4B-Instruct-2507")
    generation_llms = [hf_llm1, hf_llm2]  # can be multiple copies of same llm, if we want many generations from same model
    print("MODELS FOR GENERATION INITIALIZED")

    # Initialize judges for the council
    print("INITIALIZING LLM COUNCIL JUDGES")
    # Use local HuggingFace models for the council
    judge1 = HuggingFaceLLM(
        name="Llama-3.2-1B-Instruct-Judge1",
        model_name="local_models/Llama-3.2-1B-Instruct"
    )
    judge2 = HuggingFaceLLM(
        name="Llama-3.2-1B-Instruct-Judge2",
        model_name="local_models/Llama-3.2-1B-Instruct"
    )
    judge3 = PlaceholderLLM("Qwen3-4B-Instruct-Judge3")
    judges = [judge1, judge2, judge3]
    print("HuggingFace judges initialized successfully")
    
    council = LLMCouncil(judges=judges)
    print("LLM COUNCIL INITIALIZED")
    
    print("LOADING BENCHMARK BASIS DATA")
    booksum_data = read_booksum_data(file_path="data/benchmark_base_chapters_example.json")
    print(f"BENCHMARK BASIS LOADED ({len(booksum_data)} chapters)")

    print("BUILDING BENCHMARK DATASET")
    df_generated = build_benchmark_dataset(
        texts=booksum_data,
        llms=generation_llms,
        council=council
    )

    print("SAVING BENCHMARK DATASET")
    df_generated.to_json("data/LongConMultitaskBenchmark.json", orient="records", lines=True)
    print("BENCHMARK DATASET SAVED")
    print(f"Generated dataset contains {len(df_generated)} rows")
