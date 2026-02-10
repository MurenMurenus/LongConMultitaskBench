import pandas as pd
import sys
import os
import json
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.models.model_classes import LLM, HuggingFaceLLM, PlaceholderLLM
from scripts.prompts import hallucination_detection_prompt


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


def classify_hallucination(llm: LLM, context: str, output: str) -> int:
    """
    Classify if an output is hallucinated or not using the LLM.
    
    Args:
        llm: The LLM to use for classification
        context: The original context/reference text
        output: The generated output to evaluate
        
    Returns:
        1 if hallucinated, 0 if not hallucinated
    """
    # Create the prompt for hallucination detection
    prompt = hallucination_detection_prompt.format(context=context, output=output)
    
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


def evaluate_benchmark(df: pd.DataFrame, llm: LLM) -> Tuple[List[int], List[int], List[Dict]]:
    """
    Evaluate all samples in a benchmark using the LLM.
    
    Args:
        df: DataFrame with benchmark data
        llm: The LLM to use for evaluation
        
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
        output = row['output']
        true_label = row['label']
        benchmark_id = row['benchmark_id']
        chapter_id = row['chapter_id']
        
        # Classify using LLM
        predicted_label = classify_hallucination(llm, context, output)
        
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
            'output': output
        })
    
    print("Evaluation complete!")
    return predictions, ground_truth, detailed_results


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics


def save_predictions_and_metrics(
    detailed_results: List[Dict],
    metrics: Dict[str, float],
    benchmark_type: str,
    model_name: str,
    output_dir: str = "results"
):
    """
    Save predictions and metrics to files.
    
    Args:
        detailed_results: List of detailed results for each sample
        metrics: Dictionary of calculated metrics
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
    
    # Save metrics to JSON
    metrics_filename = f"{benchmark_type}_metrics.json"
    metrics_path = os.path.join(output_dir, metrics_filename)
    
    # Add metadata to metrics
    metrics_with_metadata = {
        "benchmark_type": benchmark_type,
        "model_name": model_name,
        "samples_evaluated": len(detailed_results),
        "metrics": metrics
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


def main():
    """
    Main function to run benchmark evaluation.
    """
    # Define benchmark paths
    benchmark_dir = "benchmarks"
    benchmark_files = {
        "qa": "qa_benchmark.csv",
        "entity": "entity_benchmark.csv",
        "structured": "structured_benchmark.csv",
        "summary": "summary_benchmark.csv"
    }
    
    # Select which benchmark to evaluate
    benchmark_type = "qa"  # Change this to evaluate different benchmarks
    benchmark_path = os.path.join(benchmark_dir, benchmark_files[benchmark_type])
    
    # Model configuration
    model_path = 'Qwen/Qwen3-4B-Instruct-2507'
    model_name = 'Qwen3-4B-Instruct-2507'
    
    # Load benchmark
    df = load_benchmark(benchmark_path)
    
    # Initialize LLM
    llm = initialize_llm(
        model_path=model_path,
        model_name=model_name
    )
    
    # Evaluate benchmark (limit to first 100 samples for faster execution)
    df_subset = df.head(100)  # Remove this limit for full evaluation
    predictions, ground_truth, detailed_results = evaluate_benchmark(df_subset, llm)
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions)
    
    # Save predictions and metrics to files
    save_predictions_and_metrics(detailed_results, metrics, benchmark_type, model_name)
    
    # Print results
    print("\n" + "="*50)
    print("BENCHMARK EVALUATION RESULTS")
    print("="*50)
    print(f"Benchmark type: {benchmark_type}")
    print(f"Model used: {model_name}")
    print(f"Samples evaluated: {len(predictions)}")
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.capitalize()}: {value:.4f}")
    
    # Count hallucinated vs non-hallucinated
    pred_hallucinated = sum(predictions)
    true_hallucinated = sum(ground_truth)
    print(f"\nHallucination counts:")
    print(f"  Predicted hallucinated: {pred_hallucinated}")
    print(f"  Actual hallucinated: {true_hallucinated}")
    print(f"  Predicted non-hallucinated: {len(predictions) - pred_hallucinated}")
    print(f"  Actual non-hallucinated: {len(ground_truth) - true_hallucinated}")


if __name__ == "__main__":
    main()
