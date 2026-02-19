import json
import uuid
from typing import List, Dict, Any
import pandas as pd
import sys
import os
import random

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.content_generation.ground_truth_functions import extract_entities_from_text, generate_qa_pairs, generate_structured_outputs, generate_summaries
from scripts.content_generation.hallucination_functions import (
    inject_structural_error,
    inject_temporal_hallucination,
    inject_entity_hallucination,
    inject_qa_hallucination
)
from scripts.content_generation.validation_functions import validate_with_council
from scripts.prompts import (
    qa_prompt_template,
    structured_prompt_template,
    entity_extraction_prompt_template,
    summary_prompt_template,
    inject_factual_hallucination_prompt_template,
    inject_structural_error_prompt_template,
    inject_temporal_hallucination_prompt_template,
    inject_entity_hallucination_prompt_template,
    inject_qa_hallucination_prompt_template,
    validate_qa_pairs_council_prompt,
    validate_structured_outputs_council_prompt,
    validate_entity_extractions_council_prompt,
    validate_summaries_council_prompt
)
from scripts.read_booksum_data import read_booksum_data
from scripts.models.model_classes import LLM, PlaceholderLLM, LLMCouncil, HuggingFaceLLM, OpenAILLM


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
        candidates=[model_output['answer'] for model_output in qa_pairs],
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
        candidates=[model_output.text for model_output in structured_outputs],
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
        candidates=[model_output['entities'] for model_output in entity_extractions], 
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
        candidates=[model_output['summary'] for model_output in summaries],
        instruction=validate_summaries_council_prompt
    )
    print(f"Validated {len(summary_validations)} summaries")
    
    # ---
    # Inject hallucinations in half of the cases
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
        if random.random() < 0.5
            else "[NO_HALLUCINATION]"
        for qa in qa_pairs
    ]
    print(f"Injected hallucinations in {len([h for h in hallucinated_answers if h != "[NO_HALLUCINATION]"])}/{len(hallucinated_answers)} answers")
    
    print("Injecting structural errors...")
    broken_structures = [
        inject_structural_error(llms[0], s.text, prompt_template=inject_structural_error_prompt_template)
        if random.random() < 0.5
            else "[NO_HALLUCINATION]"
        for s in structured_outputs
    ]
    print(f"Injected structural errors in {len([b for b in broken_structures if b != "[NO_HALLUCINATION]"])}/{len(broken_structures)} outputs")
    
    print("Injecting temporal hallucinations...")
    temporal_hallucinations = [
        inject_temporal_hallucination(
            llms[0],
            chapter_text,
            summary["summary"],
            prompt_template=inject_temporal_hallucination_prompt_template
        )
        if random.random() < 0.5
            else "[NO_HALLUCINATION]"
        for summary in summaries
    ]
    print(f"Injected temporal hallucinations in {len([t for t in temporal_hallucinations if t != "[NO_HALLUCINATION]"])}/{len(temporal_hallucinations)} summaries")
    
    print("Injecting entity hallucinations...")
    entity_hallucinations = []
    for entity_extraction in entity_extractions:
        if random.random() < 0.5:
            if "entities" in entity_extraction and isinstance(entity_extraction["entities"], dict):
                entities_text = json.dumps(entity_extraction["entities"])
                hallucinated_entities = inject_entity_hallucination(llms[0], entities_text, prompt_template=inject_entity_hallucination_prompt_template)
                entity_hallucinations.append(hallucinated_entities)
            else:
                entity_hallucinations.append("No entities to hallucinate")
        else:
            entity_hallucinations.append("[NO_HALLUCINATION]")
    print(f"Injected numerical hallucinations in {len([e for e in entity_hallucinations if e != "[NO_HALLUCINATION]"])}/{len(entity_hallucinations)} entity extractions")

    return {
        "chapter_id": chapter_id,
        "original_text": chapter_text,

        # QA
        "qa_prompt_template": qa_prompt_template,
        "qa_pair": qa_pairs,
        "qa_validation": qa_validations,
        "qa_hallucination": hallucinated_answers,

        # Structured extraction
        "structured_prompt_template": structured_prompt_template,
        "structured_output": [s for s in structured_outputs],
        "structured_validation": structured_validations,
        "structured_hallucination": broken_structures,

        # Entity extraction
        "entity_extraction_prompt_template": entity_extraction_prompt_template,
        "entity_extraction": entity_extractions,
        "entity_validation": entity_validations,
        "entity_hallucination": entity_hallucinations,

        # Summaries
        "summary_prompt_template": summary_prompt_template,
        "summary": summaries,
        "summary_validation": summary_validations,
        "summary_hallucination": temporal_hallucinations,
    }


# -------------------------
# Dataset builder
# -------------------------
def build_benchmark_dataset(
    texts: List[Dict[str, str]],
    llms: List[LLM],
    council: LLMCouncil,
    output_file: str = "data/LongConMultitaskBenchmark.jsonl"
) -> pd.DataFrame:
    
    # Load existing benchmark dataset if it exists
    existing_df = pd.DataFrame()
    existing_chapter_ids = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_json(output_file, lines=True)
            existing_chapter_ids = set(existing_df["chapter_id"].tolist())
            print(f"Found {len(existing_chapter_ids)} existing chapters in dataset")
        except Exception as e:
            print(f"Warning: Could not load existing dataset: {e}")
    else:
        # Empty file creation
        with open(output_file, "w") as f:
            pass
    
    rows = []
    for item in tqdm(texts):
        # Skip if chapter_id already exists
        if item["chapter_id"] in existing_chapter_ids:
            continue
            
        row = build_benchmark_row(
            chapter_id=item["chapter_id"],
            chapter_text=item["chapter_text"],
            llms=llms,
            council=council
        )

        with open(output_file, 'a') as f:
            json_line = pd.DataFrame(row).to_json(orient='records', lines=True)
            f.write(json_line)


# -------------------------
# Benchmark generation pipeline
# -------------------------
if __name__ == "__main__":

    DATASET_BASIS_PATH = "data/benchmark_base_chapters.json"
    OUTPUT_DATASET_PATH = "data/LongConMultitaskBenchmark.jsonl"

    print("LOADING BENCHMARK BASIS DATA")
    booksum_data = read_booksum_data(file_path=DATASET_BASIS_PATH)
    print(f"BENCHMARK BASIS LOADED ({len(booksum_data)} chapters)")

    # ---
    # MODELS INIT
    # ---
    print("INITIALIZING MODELS FOR GENERATION")

    # local setup
    # hf_llm = HuggingFaceLLM(
    #     name="Qwen3-4B-Instruct",
    #     model_name="local_models/Qwen3-4B-Instruct"
    # )
    # hf_llm1 = HuggingFaceLLM(
    #     name="Llama-3.2-1B-Instruct-1",
    #     model_name="local_models/Llama-3.2-1B-Instruct"
    # )
    # hf_llm2 = HuggingFaceLLM(
    #     name="Llama-3.2-1B-Instruct-2",
    #     model_name="local_models/Llama-3.2-1B-Instruct"
    # )

    # gpu setup
    hf_llm1 = HuggingFaceLLM(
        name="Qwen3-4B-Instruct-2507",
        model_name="Qwen/Qwen3-4B-Instruct-2507"
    )
    # hf_llm1 = HuggingFaceLLM(
    #     name="gpt-oss-20b",
    #     model_name="openai/gpt-oss-20b"
    # )
    hf_llm2 = HuggingFaceLLM(
        name="Qwen2.5-7B-Instruct",
        model_name="Qwen/Qwen2.5-7B-Instruct"
    )

    # hf_llm = PlaceholderLLM("Qwen3-4B-Instruct-2507")
    generation_llms = [hf_llm2]  # can be multiple copies of same llm, if we want many generations from same model

    # Initialize judges for the council
    print("INITIALIZING LLM COUNCIL JUDGES")

    # local setup
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

    # gpu setup
    judges = [hf_llm1, hf_llm2]
    council = LLMCouncil(judges=judges)
    
    # ---
    # BENCHMARK GENERATION
    # ---
    print("BUILDING BENCHMARK DATASET")
    build_benchmark_dataset(
        texts=booksum_data,
        llms=generation_llms,
        council=council,
        output_file=OUTPUT_DATASET_PATH
    )

    
