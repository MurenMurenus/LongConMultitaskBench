# Hydra Configuration System

This project uses Hydra for configuration management. The configuration files are organized as follows:

## Configuration Structure

```
conf/
├── config.yaml              # Main configuration file
├── generation/
│   └── models.yaml          # Models for benchmark generation
├── evaluation/
│   └── models.yaml          # Models for benchmark evaluation
└── metrics/
    └── config.yaml          # Metrics configuration
```

## Installation

To use the Hydra configuration system, you need to install the required packages:

```bash
pip install -r requirements.txt
```

This will install Hydra and other required dependencies.

## Debugging

For better error reporting during development, set the HYDRA_FULL_ERROR environment variable:

```bash
export HYDRA_FULL_ERROR=1
```

Or run scripts with it enabled:

```bash
HYDRA_FULL_ERROR=1 python scripts/benchmark_generation.py
```

## Configuration Files

### Generation Models (`conf/generation/models.yaml`)

Contains configurations for models used in benchmark generation:
- `generation_models`: Models used for generating ground truth data
- `council_judges`: Models used as judges in the LLM council for validation

### Evaluation Models (`conf/evaluation/models.yaml`)

Contains configurations for models used in benchmark evaluation:
- `evaluation_models`: Models used for generating predictions on benchmarks

### Metrics Configuration (`conf/metrics/config.yaml`)

Contains configurations for metrics calculation:
- `metrics`: Defines which metrics to calculate
- `classification`: Standard classification metrics (accuracy, precision, recall, F1)
- `additional`: Additional metrics that can be enabled

## Usage

### Benchmark Generation

```bash
python scripts/benchmark_generation.py
```

To override configurations:
```bash
python scripts/benchmark_generation.py generation_models=gpu
```

### Benchmark Evaluation

```bash
python scripts/benchmark_evaluation.py
```

To use a different model:
```bash
python scripts/benchmark_evaluation.py evaluation.model_choice=qwen3_4b_2507
```

To evaluate a different benchmark type:
```bash
python scripts/benchmark_evaluation.py benchmark.eval_type=qa
```

To use a different model environment:
```bash
python scripts/benchmark_evaluation.py evaluation.model_env=local evaluation.model_choice=qwen3_4b
```

### Metrics Calculation

```bash
python scripts/calculate_metrics.py
```

To use a different model name (consistent with evaluation):
```bash
python scripts/calculate_metrics.py evaluation.model_choice=qwen3_4b_2507
```

To enable additional metrics:
```bash
python scripts/calculate_metrics.py metrics.additional.roc_auc.enabled=true
```

## Configuration Override Examples

### Use CPU models for generation:
```bash
python scripts/benchmark_generation.py generation_models=local council_judges=local
```

### Use API models for evaluation:
```bash
python scripts/benchmark_evaluation.py evaluation_models=api
```

### Enable additional metrics:
```bash
python scripts/calculate_metrics.py metrics.additional.roc_auc.enabled=true metrics.additional.matthews_corrcoef.enabled=true
