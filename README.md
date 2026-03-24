# LongConMultitaskBench

## Overview
LongConMultitaskBench is a benchmark for evaluating models on various long-context tasks, including QA, entity extraction, structured output generation, and summarization. The project uses Hydra for flexible configuration management.

## Features
- **Hydra Configuration System**: Flexible configuration management with override capabilities
- **Multiple Model Support**: Hugging Face models (local CPU/GPU)
- **LLM Council**: Multi-model evaluation with judge aggregation
- **Benchmark Generation**: Automated ground truth generation with hallucination injection
- **Benchmark Evaluation**: Hallucination detection and classification
- **Metrics Calculation**: Comprehensive evaluation metrics (accuracy, precision, recall, F1)
- **Visualization**: Comparative histograms and heatmaps for result analysis

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
```
LongConMultitaskBench/
├── conf/                          # Hydra configuration files
│   ├── config.yaml                # Main configuration
│   ├── generation/
│   │   └── models.yaml            # Generation models config
│   ├── evaluation/
│   │   └── models.yaml            # Evaluation models config
│   └── metrics/
│       └── config.yaml            # Metrics configuration
├── scripts/
│   ├── benchmark_generation.py    # Benchmark dataset generation
│   ├── benchmark_split.py         # Dataset splitting
│   ├── benchmark_evaluation.py    # Benchmark evaluation pipeline
│   ├── calculate_metrics.py       # Metrics calculation
│   ├── visualize_metrics.py       # Results visualization
│   └── ...
├── data/                          # Dataset files
├── benchmarks/                    # Benchmark CSV files
├── results_*/                     # Evaluation results
└── visualizations/                # Generated plots
```

## Configuration System

### Main Configuration (`conf/config.yaml`)
- **paths**: Directory paths for data, benchmarks, results, models
- **dataset**: Benchmark generation paths
- **benchmark**: Benchmark types and evaluation type selection
- **evaluation**: Model environment and choice settings

### Model Configurations
- **`conf/generation/models.yaml`**: Models for benchmark generation and LLM council judges
- **`conf/evaluation/models.yaml`**: Models for benchmark evaluation (prediction generation)

### Metrics Configuration (`conf/metrics/config.yaml`)
- Enabled metrics selection
- Additional metrics options (ROC AUC, Matthews correlation)

## Usage

### 1. Benchmark Generation
Generate the benchmark dataset with ground truth and hallucinated samples:

```bash
python scripts/benchmark_generation.py
```

**Configuration overrides:**
```bash
# Use CPU models for generation
python scripts/benchmark_generation.py generation_models=local council_judges=local

# Use GPU models (default)
python scripts/benchmark_generation.py generation_models=gpu council_judges=gpu
```

### 2. Benchmark Evaluation
Evaluate models on benchmark tasks:

```bash
python scripts/benchmark_evaluation.py
```

**Configuration overrides:**
```bash
# Select benchmark type (qa, entity, structured, summary)
python scripts/benchmark_evaluation.py benchmark.eval_type=qa

# Select model from GPU environment
python scripts/benchmark_evaluation.py evaluation.model_choice=qwen3_4b_2507

# Select model from local environment
python scripts/benchmark_evaluation.py evaluation.model_env=local evaluation.model_choice=qwen3_4b
```

### 3. Metrics Calculation
Calculate evaluation metrics from predictions:

```bash
python scripts/calculate_metrics.py
```

**Configuration overrides:**
```bash
# Select model (must match evaluation model_choice)
python scripts/calculate_metrics.py evaluation.model_choice=qwen3_4b_2507
```

### 4. Visualization
Generate comparative visualizations:

```bash
python scripts/visualize_metrics.py
```

This creates:
- Per-benchmark comparison plots (`visualizations/{type}_comparison.png`)
- Overall metrics histogram (`visualizations/overall_metrics_histogram.png`)
- Performance heatmap (`visualizations/overall_performance_heatmap.png`)

## Available Scripts

| Script | Description |
|--------|-------------|
| `prompts.py` | Prompt templates for various tasks |
| `benchmark_generation.py` | Generate benchmark dataset with ground truth and hallucinations |
| `benchmark_split.py` | Split benchmark data |
| `benchmark_evaluation.py` | Evaluate models on benchmark tasks |
| `calculate_metrics.py` | Calculate metrics from evaluation results |
| `visualize_metrics.py` | Create visualization plots |

## Benchmark Types

- **QA**: Question-Answering with hallucination detection
- **Entity**: Entity extraction and validation
- **Structured**: Structured output generation
- **Summary**: Text summarization with temporal hallucination detection

## Models

The framework supports multiple model implementations:

### Hugging Face Models
- Local models (CPU): `qwen3_4b`, `llama_3_2_1b`
- GPU models: `qwen3_4b_2507`, `qwen2_5_7b`, `qwen3_8b`, `gpt_oss_20b`

### Special Models
- `QwenThinkingLLM`: Qwen3-8B with thinking capability
- `PlaceholderLLM`: For testing

## LLM Council
The LLM Council aggregates multiple LLMs as judges for validating generated content. Configurable judges include:
- Local judges (CPU-based for development)
- GPU judges (production-ready)

## Debugging
For detailed error reporting with Hydra:
```bash
export HYDRA_FULL_ERROR=1
```

Or run with prefix:
```bash
HYDRA_FULL_ERROR=1 python scripts/benchmark_generation.py
```

## Results
Evaluation results are saved in `results_{model_name}/` directories:
- `{benchmark_type}_predictions.csv`: Detailed predictions
- `{benchmark_type}_metrics.json`: Aggregated metrics

## Example Workflow

```bash
# 1. Generate benchmark dataset
python scripts/benchmark_generation.py

# 2. Split benchmark data into one-scenario (one task type) benchmarks
python scripts/benchmark_split.py

# 3. Evaluate a model on QA benchmark
python scripts/benchmark_evaluation.py evaluation.model_choice=qwen2_5_7b benchmark.eval_type=qa

# 4. Calculate metrics
python scripts/calculate_metrics.py evaluation.model_choice=qwen2_5_7b benchmark.eval_type=qa

# 5. Generate visualizations
python scripts/visualize_metrics.py
