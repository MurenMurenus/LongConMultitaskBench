import pandas as pd
import sys
import json
import os
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import hydra
from omegaconf import DictConfig

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set HYDRA_FULL_ERROR=1 for full error reporting
os.environ["HYDRA_FULL_ERROR"] = "1"

def load_predictions(predictions_path: str) -> pd.DataFrame:
    """
    Load predictions from CSV file.
    
    Args:
        predictions_path: Path to the predictions CSV file
        
    Returns:
        DataFrame with predictions data
    """
    print(f"Loading predictions from {predictions_path}")
    df = pd.read_csv(predictions_path)
    print(f"Loaded {len(df)} predictions")
    return df


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


def save_metrics(
    metrics: Dict[str, float],
    benchmark_type: str,
    model_name: str,
    samples_evaluated: int,
    output_dir: str = "results"
):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of calculated metrics
        benchmark_type: Type of benchmark evaluated
        model_name: Name of the model used for evaluation
        samples_evaluated: Number of samples evaluated
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON
    metrics_filename = f"{benchmark_type}_metrics.json"
    metrics_path = os.path.join(output_dir, metrics_filename)
    
    # Add metadata to metrics
    metrics_with_metadata = {
        "benchmark_type": benchmark_type,
        "model_name": model_name,
        "samples_evaluated": samples_evaluated,
        "metrics": metrics
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_with_metadata, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main function to calculate metrics from predictions.
    """
    # Configuration from Hydra config
    benchmark_type = cfg.benchmark.eval_type
    
    # Get model configuration from Hydra config for consistent naming
    model_env = cfg.evaluation.get("model_env", "gpu")
    model_choice = cfg.evaluation.get("model_choice", "qwen2_5_7b")
    
    # Get the model name from the configuration
    model_config = cfg.evaluation.evaluation_models[model_env][model_choice]
    model_name = model_config.get("name", model_choice)

    results_dir = f"results_{model_name}"
    
    # Load predictions
    predictions_filename = f"{benchmark_type}_predictions.csv"
    predictions_path = os.path.join(results_dir, predictions_filename)
    df = load_predictions(predictions_path)
    
    # Extract true and predicted labels
    y_true = df['true_label'].tolist()
    y_pred = df['predicted_label'].tolist()
    
    # Calculate metrics based on config
    metrics = {}
    for metric_config in cfg.metrics.classification:
        if metric_config.enabled:
            metric_name = metric_config.name
            if metric_name == "accuracy":
                metrics[metric_name] = accuracy_score(y_true, y_pred)
            elif metric_name == "precision":
                metrics[metric_name] = precision_score(y_true, y_pred, zero_division=0)
            elif metric_name == "recall":
                metrics[metric_name] = recall_score(y_true, y_pred, zero_division=0)
            elif metric_name == "f1":
                metrics[metric_name] = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate additional metrics based on config
    for metric_config in cfg.metrics.additional:
        if metric_config.enabled:
            metric_name = metric_config.name
            if metric_name == "matthews_corrcoef":
                from sklearn.metrics import matthews_corrcoef
                metrics[metric_name] = matthews_corrcoef(y_true, y_pred)
    
    # Save metrics
    save_metrics(
        metrics,
        benchmark_type,
        model_name,
        len(df),
        output_dir=results_dir
    )
    
    # Print results
    print("\n" + "="*50)
    print("METRICS CALCULATION RESULTS")
    print("="*50)
    print(f"Benchmark type: {benchmark_type}")
    print(f"Model used: {model_name}")
    print(f"Samples evaluated: {len(y_true)}")
    print("\nMetrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.capitalize()}: {value:.4f}")
    
    # Count hallucinated vs non-hallucinated
    pred_hallucinated = sum(y_pred)
    true_hallucinated = sum(y_true)
    print(f"\nHallucination counts:")
    print(f"  Predicted hallucinated: {pred_hallucinated}")
    print(f"  Actual hallucinated: {true_hallucinated}")
    print(f"  Predicted non-hallucinated: {len(y_pred) - pred_hallucinated}")
    print(f"  Actual non-hallucinated: {len(y_true) - true_hallucinated}")


if __name__ == "__main__":
    main()
