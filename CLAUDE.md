# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project studying the Superficial Alignment Hypothesis (SAH) in machine learning. The codebase uses PyTorch Lightning and Hydra for experiment configuration and management, with support for SLURM cluster execution.

## Development Commands

### Environment Setup

```bash
uv sync                        # Install dependencies
. .venv/bin/activate          # Activate virtual environment
```

### Running Experiments

```bash
# Main entry point for experiments
python sah/main.py --help

# Run specific experiment configurations
python sah/main.py experiment=<experiment_name>

# Example experiments:
python sah/main.py experiment=train-with-strategy
python sah/main.py experiment=grammar-entropy
python sah/main.py experiment=pretrain-then-finetune
```

### Testing and Code Quality

```bash
# Run tests
pytest                        # Run all tests
pytest sah/                   # Run tests in specific directory
pytest -v                     # Verbose output

# Code formatting and linting
ruff check                    # Check code style
ruff format                   # Format code
```

### SLURM Cluster Execution

The `slurm/` directory contains various shell scripts for running experiments on SLURM clusters:

- `slurm/train-with-strategy.sh` - Train models with data selection strategies
- `slurm/grammar-entropy.sh` - Estimate grammar entropy
- `slurm/pretrain-then-finetune.sh` - Pretrain then finetune experiments

## Architecture

### Core Components

1. **Main Entry Point**: `sah/main.py`

    - Uses Hydra for configuration management
    - Instantiates algorithms, trainers, and datamodules
    - Calls `train_and_evaluate` function

2. **Experiment Framework**: `sah/experiment.py`

    - Contains `train_and_evaluate` function using `functools.singledispatch`
    - Handles Lightning training and evaluation loops
    - Supports custom training procedures for different algorithms

3. **Configuration System**: `sah/configs/`

    - Hydra-based configuration with YAML files
    - Modular config structure: algorithm, trainer, datamodule, experiment
    - Supports cluster configurations for distributed training

4. **Algorithms**: `sah/algorithms/`

    - Lightning modules for various experiments
    - Key algorithms include:
        - `train_with_strategy.py` - Data selection strategies
        - `pretrain_then_finetune.py` - Multi-stage training
        - `estimate_entropy.py` - Entropy estimation
        - `llm_finetuning.py` - Language model fine-tuning

5. **Data Strategies**: `sah/algorithms/data_strategies/`

    - `hashed_ngram.py` - Hash-based n-gram filtering
    - `proposed_strategy.py` - Custom data selection strategy

6. **Networks**: `sah/algorithms/networks/`

    - Neural network architectures
    - Transformer variants and utility functions

### Key Design Patterns

- **Hydra Configuration**: All experiments use Hydra for configuration management
- **Lightning Modules**: Algorithms extend `lightning.LightningModule`
- **Single Dispatch**: `train_and_evaluate` uses `@functools.singledispatch` for algorithm-specific training
- **Modular Architecture**: Clear separation between algorithms, networks, and data handling

### Data Flow

1. Configuration parsed by Hydra in `main.py`
2. Algorithm, trainer, and datamodule instantiated based on config
3. Training orchestrated by `train_and_evaluate` function
4. Results logged via Lightning loggers (WandB, TensorBoard)

## Configuration Structure

Experiments are configured through YAML files in `sah/configs/`:

- `experiment/` - High-level experiment definitions
- `algorithm/` - Algorithm-specific configurations
- `trainer/` - Lightning trainer settings
- `cluster/` - SLURM cluster configurations

## Testing

Tests are co-located with source code using `*_test.py` naming convention. The project uses pytest with these key features:

- Doctest support enabled
- Custom markers for incremental testing
- CUDA deterministic mode for reproducible tests
