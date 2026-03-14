import pandas as pd
import sys
import os
import json
from typing import List, Dict, Tuple
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set HYDRA_FULL_ERROR=1 for full error reporting
os.environ["HYDRA_FULL_ERROR"] = "1"
from scripts.models.model_classes import LLM, HuggingFaceLLM, PlaceholderLLM
from scripts.prompts import (
    hallucination_detection_prompt,
    qa_hallucination_detection_prompt,
    structured_hallucination_detection_prompt,
    entity_hallucination_detection_prompt,
    summary_hallucination_detection_prompt
)


def load_benchmark(benchmark_path: str) -> pd.DataFrame:
    """
    Load a benchmark CSV file.
    
    Args:
        benchmark_path: Path to the benchmark CSV file
        
    Returns:
        DataFrame with benchmark data
    """
    print(f"Loading benchmark from {benchmark_path}")
    df = pd.read_csv(benchmark_path)
    print(f"Loaded {len(df)} benchmark samples")
    return df


def initialize_llm(
    model_path: str = None,
    model_name: str = None
) -> LLM:
    """
    Initialize an LLM for hallucination detection.
    
    Args:
        model_path: Path to the model or 'OpenAI_API' for placeholder
        model_name: Name for the model instance
        
    Returns:
        Initialized LLM instance
    """
    if model_path == 'OpenAI_API':
        llm = PlaceholderLLM(name=model_name or "OpenAI-Placeholder")
    else:
        llm = HuggingFaceLLM(
            name=model_name or "HuggingFace-Model",
            model_name=model_path
        )
    
    return llm


def classify_hallucination(llm: LLM, context: str, prompt_text: str, output: str, benchmark_type: str) -> int:
    """
    Classify if an output is hallucinated or not using the LLM.
    
    Args:
        llm: The LLM to use for classification
        context: The original context/reference text
        prompt_text: The prompt used to generate the output
        output: The generated output to evaluate
        benchmark_type: The type of benchmark (qa, structured, entity, summary)
        
    Returns:
        1 if hallucinated, 0 if not hallucinated
    """
    # Select the appropriate prompt template based on benchmark type
    if benchmark_type == "qa":
        detection_prompt = qa_hallucination_detection_prompt
    elif benchmark_type == "structured":
        detection_prompt = structured_hallucination_detection_prompt
    elif benchmark_type == "entity":
        detection_prompt = entity_hallucination_detection_prompt
    elif benchmark_type == "summary":
        detection_prompt = summary_hallucination_detection_prompt
    else:
        # Default to general hallucination detection prompt
        detection_prompt = hallucination_detection_prompt
    
    # Create the prompt for hallucination detection
    if benchmark_type == "qa":
        prompt = detection_prompt.format(context=context, prompt=prompt_text, output=output)
    elif benchmark_type in ["structured", "entity"]:
        prompt = detection_prompt.format(context=context, prompt=prompt_text, output=output)
    elif benchmark_type == "summary":
        prompt = detection_prompt.format(context=context, output=output)
    else:
        # For general case, use context and output only
        prompt = detection_prompt.format(context=context, output=output)
    
    # Generate response from LLM
    llm_output = llm.generate(prompt, "")
    response_text = llm_output.text.upper().strip()
    
    # Parse the response
    if "HALLUCINATED" in response_text:
        return 1
    elif "CORRECT" in response_text:
        return 0
    else:
        # Default to non-hallucinated if uncertain
        print(f"Unclear response from LLM: {response_text}")
        return 0


def evaluate_benchmark(df: pd.DataFrame, llm: LLM, benchmark_type: str) -> Tuple[List[int], List[int], List[Dict]]:
    """
    Evaluate all samples in a benchmark using the LLM.
    
    Args:
        df: DataFrame with benchmark data
        llm: The LLM to use for evaluation
        benchmark_type: The type of benchmark (qa, structured, entity, summary)
        
    Returns:
        Tuple of (predictions, ground_truth_labels, detailed_results)
    """
    predictions = []
    ground_truth = []
    detailed_results = []
    
    print("Evaluating benchmark samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating samples"):
        # Get context and output
        context = row['context']
        prompt_text = row['prompt']
        output = row['output']
        true_label = row['label']
        benchmark_id = row['benchmark_id']
        chapter_id = row['chapter_id']
        
        # Classify using LLM
        predicted_label = classify_hallucination(llm, context, prompt_text, output, benchmark_type)
        
        predictions.append(predicted_label)
        ground_truth.append(true_label)
        
        # Store detailed results
        detailed_results.append({
            'benchmark_id': benchmark_id,
            'chapter_id': chapter_id,
            'model_name': llm.name,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'context': context,
            'prompt': prompt_text,
            'output': output
        })
    
    print("Evaluation complete!")
    return predictions, ground_truth, detailed_results


def save_predictions(
    detailed_results: List[Dict],
    benchmark_type: str,
    model_name: str,
    output_dir: str = "results"
):
    """
    Save predictions to CSV file.
    
    Args:
        detailed_results: List of detailed results for each sample
        benchmark_type: Type of benchmark evaluated
        model_name: Name of the model used for evaluation
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame(detailed_results)
    predictions_filename = f"{benchmark_type}_predictions.csv"
    predictions_path = os.path.join(output_dir, predictions_filename)
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to run benchmark evaluation.
    """
    # Define benchmark paths
    benchmark_dir = cfg.paths.benchmarks_dir
    benchmark_files = {
        "qa": "qa_benchmark.csv",
        "entity": "entity_benchmark.csv",
        "structured": "structured_benchmark.csv",
        "summary": "summary_benchmark.csv"
    }
    
    # Select which benchmark to evaluate from config
    benchmark_type = cfg.benchmark.eval_type
    benchmark_path = os.path.join(benchmark_dir, benchmark_files[benchmark_type])
    
    # Get model configuration from Hydra config
    # Allow specifying model through config.evaluation.model_choice
    model_env = cfg.evaluation.model_env
    model_name = cfg.evaluation.model_choice
    model_config = cfg.evaluation.evaluation_models[model_env][model_name]
    
    # Load benchmark
    df = load_benchmark(benchmark_path)
    
    # Initialize LLM from config using instantiate
    llm = instantiate(model_config)
    
    # Evaluate benchmark (limit to first 100 samples for faster execution)
    # df = df.head(100)  # Remove this limit for full evaluation
    predictions, ground_truth, detailed_results = evaluate_benchmark(df, llm, benchmark_type)
    
    # Save predictions to file
    save_predictions(
        detailed_results,
        benchmark_type,
        llm.name,
        output_dir=f"results_{llm.name}"
    )
    
    # Print basic completion info
    print("\n" + "="*50)
    print("BENCHMARK EVALUATION COMPLETE")
    print("="*50)
    print(f"Benchmark type: {benchmark_type}")
    print(f"Model used: {llm.name}")
    print(f"Samples evaluated: {len(predictions)}")
    print(f"Predictions saved to results_{llm.name}/{benchmark_type}_predictions.csv")


if __name__ == "__main__":
    main()
