# LongConMultitaskBench

## Overview
LongConMultitaskBench is a benchmark for evaluating models on various long-context tasks.

## Features
- Supports multiple LLM implementations (Hugging Face, OpenAI API)
- LLM Council for multi-model evaluation
- Benchmark generation for long-context tasks
- Detailed logging during benchmark generation
- Optimized CPU loading for Hugging Face models

## Installation
```bash
pip install -r requirements.txt
```

## Usage
To generate the benchmark dataset:
```bash
python scripts/benchmark_generation.py
```

## Models
The framework supports multiple model implementations:
- Hugging Face models (local)
- OpenAI API models
- Placeholder models (for testing)

## LLM Council
The LLM Council aggregates multiple LLMs as judges for evaluating model outputs.
It now includes OpenAI models as evaluators by default.

## Configuration
Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
