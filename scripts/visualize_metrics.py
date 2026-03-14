#!/usr/bin/env python3
"""
Script to visualize metrics from model evaluation results.
Creates comparative histograms for different models and benchmark types.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any


def find_results_directories(base_path: str = ".") -> List[str]:
    """
    Find all directories that start with 'results_'.
    
    Args:
        base_path: Base path to search for results directories
        
    Returns:
        List of paths to results directories
    """
    pattern = os.path.join(base_path, "results_*")
    return glob.glob(pattern)


def load_metrics_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Load all metric files from a results directory.
    
    Args:
        directory: Path to results directory
        
    Returns:
        List of dictionaries containing metric data
    """
    metrics_files = glob.glob(os.path.join(directory, "*_metrics.json"))
    metrics_data = []
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                metrics_data.append(data)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    return metrics_data


def collect_all_metrics(results_dirs: List[str]) -> pd.DataFrame:
    """
    Collect metrics from all results directories into a single DataFrame.
    
    Args:
        results_dirs: List of results directory paths
        
    Returns:
        DataFrame with all metrics data
    """
    all_data = []
    
    for directory in results_dirs:
        model_name = os.path.basename(directory)
        metrics_list = load_metrics_from_directory(directory)
        
        for metrics_data in metrics_list:
            benchmark_type = metrics_data.get("benchmark_type", "unknown")
            samples_evaluated = metrics_data.get("samples_evaluated", 0)
            
            # Extract individual metrics
            metrics = metrics_data.get("metrics", {})
            
            for metric_name, metric_value in metrics.items():
                all_data.append({
                    "model": model_name,
                    "benchmark_type": benchmark_type,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "samples_evaluated": samples_evaluated
                })
    
    return pd.DataFrame(all_data)


def create_comparative_histograms(df: pd.DataFrame, output_dir: str = "visualizations"):
    """
    Create comparative histograms for metrics across models.
    
    Args:
        df: DataFrame with metrics data
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique benchmark types and metric names
    benchmark_types = df["benchmark_type"].unique()
    metric_names = df["metric_name"].unique()
    
    # Create histograms for each benchmark type
    for benchmark_type in benchmark_types:
        benchmark_data = df[df["benchmark_type"] == benchmark_type]
        
        if benchmark_data.empty:
            continue
            
        # Create a subplot for each metric
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Metric Comparison - {benchmark_type.title()} Benchmark', fontsize=16)
        
        axes = axes.flatten()
        
        for i, metric_name in enumerate(metric_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            metric_data = benchmark_data[benchmark_data["metric_name"] == metric_name]
            
            if metric_data.empty:
                ax.set_title(f'{metric_name.title()}')
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Create bar plot
            sns.barplot(data=metric_data, x="model", y="metric_value", ax=ax)
            ax.set_title(f'{metric_name.title()}')
            ax.set_ylabel('Score')
            ax.set_xlabel('Model')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
        
        # Hide unused subplots
        for j in range(len(metric_names), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{benchmark_type}_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {benchmark_type} comparison plot to {output_path}")
    
    # Create overall comparison plots
    create_overall_comparison_plots(df, output_dir)


def create_overall_metrics_histogram(df: pd.DataFrame, output_dir: str):
    """
    Create an overall histogram showing all metrics grouped by model.
    
    Args:
        df: DataFrame with metrics data
        output_dir: Directory to save visualizations
    """
    # Create a combined metric identifier
    df_copy = df.copy()
    df_copy['metric_identifier'] = df_copy['benchmark_type'] + '_' + df_copy['metric_name']
    
    # Create a wide format dataframe for easier plotting
    pivot_df = df_copy.pivot_table(
        index='model',
        columns='metric_identifier',
        values='metric_value'
    )
    
    # Create the overall histogram
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Melt the dataframe for easier plotting with seaborn
    melted_df = df_copy[['model', 'metric_identifier', 'metric_value']]
    
    # Create bar plot with all metrics grouped by model
    sns.barplot(data=melted_df, x='model', y='metric_value', hue='metric_identifier', ax=ax)
    
    ax.set_title('Overall Metrics Comparison Across All Models', fontsize=16)
    ax.set_ylabel('Metric Value')
    ax.set_xlabel('Model')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels on bars (for better readability)
    # Note: This might be cluttered with too many bars, so we'll skip individual labels
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "overall_metrics_histogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall metrics histogram to {output_path}")


def create_overall_comparison_plots(df: pd.DataFrame, output_dir: str):
    """
    Create overall comparison plots showing all models and metrics.
    
    Args:
        df: DataFrame with metrics data
        output_dir: Directory to save visualizations
    """
    # Pivot the data for easier plotting
    pivot_df = df.pivot_table(
        index=['model'], 
        columns=['benchmark_type', 'metric_name'], 
        values='metric_value'
    )
    
    # Create heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'Metric Value'})
    plt.title('Overall Model Performance Comparison')
    plt.ylabel('Model')
    plt.xlabel('Benchmark Type - Metric')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "overall_performance_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overall performance heatmap to {output_path}")


def main():
    """Main function to generate visualizations."""
    print("Finding results directories...")
    results_dirs = find_results_directories()
    
    if not results_dirs:
        print("No results directories found!")
        return
    
    print(f"Found {len(results_dirs)} results directories:")
    for directory in results_dirs:
        print(f"  - {directory}")
    
    print("\nCollecting metrics data...")
    df = collect_all_metrics(results_dirs)
    
    if df.empty:
        print("No metrics data found!")
        return
    
    print(f"Collected {len(df)} metric entries")
    print(f"Models found: {df['model'].unique()}")
    print(f"Benchmark types: {df['benchmark_type'].unique()}")
    print(f"Metrics: {df['metric_name'].unique()}")
    
    print("\nGenerating visualizations...")
    create_comparative_histograms(df)
    create_overall_metrics_histogram(df, "visualizations")
    create_overall_comparison_plots(df, "visualizations")
    
    print("\nVisualization generation complete!")


if __name__ == "__main__":
    main()
