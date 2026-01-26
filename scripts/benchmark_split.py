import os
import pandas as pd
import json
import uuid
from typing import List, Dict, Any


def split_into_labeled_benchmarks(
    benchmark_file: str = "data/LongConMultitaskBenchmark.jsonl",
    output_dir: str = "data/labeled_benchmarks"
) -> None:
    """
    Split the benchmark dataset into labeled subsets for each task type.
    
    Creates benchmark files with columns 'benchmark_id', 'chapter_id', 'context', 'output', 'label'
    - Regenerates 'benchmark_id' with random uuid
    - Gets 'chapter_id' from original dataset
    - Makes 'label' from hallucinations column
    - If there is hallucinated output in hallucinations column, 'output' contains this output and 'label' is 1
    - Otherwise 'label' = 0 and 'output' is the original non-hallucinated output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the benchmark dataset
    df = pd.read_json(benchmark_file, lines=True)
    
    print(f"Loaded {len(df)} benchmark entries")

    # Process QA pairs
    print("Processing QA pairs...")
    qa_rows = []
    for _, row in df.iterrows():
        chapter_id = row['chapter_id']
        original_text = row['original_text']
        prompt = row['qa_pair']['question']
        
        # Process valid QA pairs
        if row['qa_hallucination'] != "[NO_HALLUCINATION]":
            # Use hallucinated output with label 1
            output = row['qa_hallucination']
            label = 1
        else:
            # Use original output with label 0
            output = row['qa_pair']['answer']
            label = 0
            
        qa_rows.append({
            'benchmark_id': str(uuid.uuid4()),
            'chapter_id': chapter_id,
            'context': original_text,
            'prompt': prompt,
            'output': output,
            'label': label
        })
    
    qa_df = pd.DataFrame(qa_rows)
    qa_output_path = os.path.join(output_dir, "qa_benchmark.csv")
    qa_df.to_csv(qa_output_path, index=False)
    print(f"Saved QA benchmark with {len(qa_rows)} entries to {qa_output_path}")

    # Process structured outputs
    print("Processing structured outputs...")
    structured_rows = []
    for _, row in df.iterrows():
        chapter_id = row['chapter_id']
        original_text = row['original_text']
        prompt = row['structured_prompt_template']
        
        if row['structured_hallucination'] != "[NO_HALLUCINATION]":
            # Use hallucinated output with label 1
            output = row['structured_hallucination']
            label = 1
        else:
            # Use original output with label 0
            output = row['structured_output']['text']
            label = 0
            
        structured_rows.append({
            'benchmark_id': str(uuid.uuid4()),
            'chapter_id': chapter_id,
            'context': original_text,
            'prompt': prompt,
            'output': output,
            'label': label
        })
    
    structured_df = pd.DataFrame(structured_rows)
    structured_output_path = os.path.join(output_dir, "structured_benchmark.csv")
    structured_df.to_csv(structured_output_path, index=False)
    print(f"Saved structured benchmark with {len(structured_rows)} entries to {structured_output_path}")

    # Process entity extractions
    print("Processing entity extractions...")
    entity_rows = []
    for _, row in df.iterrows():
        chapter_id = row['chapter_id']
        original_text = row['original_text']
        prompt = row['entity_extraction_prompt_template']
        
        if row['entity_hallucination'] != "[NO_HALLUCINATION]":
            # Use hallucinated output with label 1
            output = row['entity_hallucination']
            label = 1
        else:
            # Use original output with label 0
            output = json.dumps(row['entity_extraction'].get('entities', {}))
            label = 0
            
        entity_rows.append({
            'benchmark_id': str(uuid.uuid4()),
            'chapter_id': chapter_id,
            'context': original_text,
            'prompt': prompt,
            'output': output,
            'label': label
        })
    
    # Save entity benchmark
    entity_df = pd.DataFrame(entity_rows)
    entity_output_path = os.path.join(output_dir, "entity_benchmark.csv")
    entity_df.to_csv(entity_output_path, index=False)
    print(f"Saved entity benchmark with {len(entity_rows)} entries to {entity_output_path}")

    # Process summaries
    print("Processing summaries...")
    summary_rows = []
    for _, row in df.iterrows():
        chapter_id = row['chapter_id']
        original_text = row['original_text']
        prompt = row['summary_prompt_template']
        
        if row['summary_hallucination'] != "[NO_HALLUCINATION]":
            # Use hallucinated output with label 1
            output = row['summary_hallucination']
            label = 1
        else:
            # Use original output with label 0
            output = row['summary']['summary']
            label = 0
            
        summary_rows.append({
            'benchmark_id': str(uuid.uuid4()),
            'chapter_id': chapter_id,
            'context': original_text,
            'prompt': prompt,
            'output': output,
            'label': label
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_output_path = os.path.join(output_dir, "summary_benchmark.csv")
    summary_df.to_csv(summary_output_path, index=False)
    print(f"Saved summary benchmark with {len(summary_rows)} entries to {summary_output_path}")

    print("Benchmark splitting completed!")


if __name__ == "__main__":
    # Script can be run from terminal
    import argparse
    
    parser = argparse.ArgumentParser(description="Split benchmark dataset into labeled subsets")
    parser.add_argument(
        "--benchmark-file", 
        default="data/LongConMultitaskBenchmark.jsonl",
        help="Path to the input benchmark file (default: data/LongConMultitaskBenchmark.jsonl)"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/labeled_benchmarks",
        help="Directory to save the output benchmark files (default: data/labeled_benchmarks)"
    )
    
    args = parser.parse_args()
    
    split_into_labeled_benchmarks(
        benchmark_file=args.benchmark_file,
        output_dir=args.output_dir
    )
