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

# Primary experiments (Iterative Strategy - Proposed Method):
python sah/main.py experiment=finetune-with-strategy-iterative

# Other experiments:
python sah/main.py experiment=finetune-with-strategy  # With configurable strategy
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

- `slurm/finetune-with-strategy.sh` - Run iterative strategy and other finetuning experiments
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
        - `finetune_with_strategy.py` - Parameter-efficient finetuning with pluggable strategies
        - `pretrain_then_finetune.py` - Multi-stage training
        - `estimate_entropy.py` - Entropy estimation
        - `llm_finetuning.py` - Language model fine-tuning

5. **Finetuning Strategies**: `sah/algorithms/strategies/`

    - **Base Strategy** (`base_strategy.py`): Abstract base class defining the strategy interface
    - **Iterative Strategy** (`iterative.py`): **\[PROPOSED RESEARCH METHOD\]** - Gradient-based parameter-efficient finetuning
    - **LoRA Strategy** (`lora.py`): Low-Rank Adaptation using PEFT library
    - **Adam/SGD Strategies** (`adam.py`, `sgd.py`): Standard full-parameter optimization baselines
    - **Compressed Finetune** (`compressed_finetune.py`): Batch-wise gradient compression with scale optimization

6. **Data Strategies**: `sah/algorithms/data_strategies/`

    - `hashed_ngram.py` - Hash-based n-gram filtering
    - `proposed_strategy.py` - Custom data selection strategy

7. **Networks**: `sah/algorithms/networks/`

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

## Finetune with Strategy Framework

### Overview

The `FinetuneWithStrategy` algorithm (`sah/algorithms/finetune_with_strategy.py`) is the main framework for studying parameter-efficient finetuning methods. It provides a plugin architecture where different optimization strategies can be swapped via configuration.

**Key Features:**

- Delegates all training logic to pluggable strategy objects
- Supports FSDP (Fully Sharded Data Parallel) for distributed training
- Tracks and logs communication cost in bits
- Works with HuggingFace transformers (primarily Llama models)

**Configuration:**

- Config: `sah/configs/algorithm/finetune_with_strategy.yaml`
- Strategies: `sah/configs/algorithm/strategy/*.yaml`
- Example: `sah/configs/experiment/finetune-with-strategy-iterative.yaml`

### Strategy Pattern

All strategies inherit from `BaseStrategy` and implement these key methods:

1. **`setup(pl_module, stage)`**: Initialize strategy (replace layers, setup PEFT, etc.)
2. **`training_step(pl_module, batch, batch_idx)`**: Define how to compute loss
3. **`configure_optimizers(pl_module)`**: Return optimizer for trainable parameters
4. **`compute_bits(pl_module)`**: Calculate communication cost in bits
5. **`train_dataloader(pl_module)`**: Return data loader (optional custom sampling)
6. **`on_train_start(pl_module)`**: Hook called at training start
7. **`on_train_batch_end(...)`**: Hook called after each batch

### Available Strategies

#### 1. Iterative Strategy (Proposed Method) ⭐

**File**: `sah/algorithms/strategies/iterative.py`

**This is the primary research contribution being studied in this work.**

**Method Overview:**
The iterative strategy implements a novel parameter-efficient finetuning approach that stores gradients in memory and learns scalar weights to combine them. Instead of updating all model parameters, it:

1. **Pre-computes gradients**: On the first batch, computes and stores gradients for `grads_in_memory` examples
2. **Learns combination weights**: Optimizes scalar scale parameters that weight the stored gradients
3. **Efficient representation**: Model update = `original_weights + Σ(scale_i × gradient_i)`

**Key Components:**

- **`ModifiedLinear`**: Custom linear layer that stores:

    - `main_weight`/`main_bias`: Original parameters
    - `weight_grads`/`bias_grads`: Stored gradients (shape: `[grads_in_memory, *param_shape]`)
    - `scale`: Learnable scalar weights (shape: `[grads_in_memory]`)
    - `activated`: Flag to switch between gradient computation mode and optimization mode

- **Training Process**:

    - **Batch 0** (Gradient Collection Phase):
        1. Merge any previous scale×gradient updates into weights
        2. Reset optimizer state
        3. Compute gradients for each example individually via `update_gradients()`
        4. Store gradients in `weight_grads` and `bias_grads` tensors
        5. Activate layers for scale optimization
    - **Batch 1+** (Scale Optimization Phase):
        1. Forward pass uses: `W_effective = W + Σ(scale_i × grad_i) / n`
        2. Backward pass only updates scale parameters
        3. Optimizer (AdamW) updates scales to minimize loss

- **`SamplerWithSpecialFirstBatch`**: Custom sampler that ensures:

    - First batch has size `grads_in_memory` (for gradient collection)
    - Remaining batches have normal `batch_size`
    - Supports distributed training across multiple GPUs

**Hyperparameters:**

```yaml
lr: 1e-5 to 4e-5          # Learning rate for scale parameters
grads_in_memory: 1-3      # Number of gradients to store per layer
```

**Communication Cost:**

- **Bits = 0** (only needs to communicate which examples were used, not the updates themselves)

**File Locations:**

- Implementation: `sah/algorithms/strategies/iterative.py:9`
- Config: `sah/configs/algorithm/strategy/iterative.yaml`
- Experiment: `sah/configs/experiment/finetune-with-strategy-iterative.yaml`

#### 2. LoRA Strategy (Baseline)

**File**: `sah/algorithms/strategies/lora.py`

Low-Rank Adaptation from the PEFT library. Adds trainable low-rank matrices to attention layers.

**Method**:

- Freezes original weights
- Adds `A` and `B` matrices where `ΔW = B·A` (with `rank = r`)
- Update: `W_effective = W + (B·A) × (lora_alpha / r)`

**Hyperparameters:**

```yaml
lr: 1e-3                  # Learning rate
r: 8                      # Rank of LoRA matrices
lora_alpha: 8             # Scaling factor
lora_dropout: 0.0         # Dropout rate
```

**Communication Cost:**

- Bits = trainable parameters × 16 (only LoRA matrices)

#### 3. SGD Strategy (Full Finetuning Baseline)

**File**: `sah/algorithms/strategies/sgd.py`

Standard stochastic gradient descent on all model parameters.

**Method**: Traditional full-parameter finetuning with SGD optimizer.

**Hyperparameters:**

```yaml
lr: 1e-5                  # Learning rate
```

**Communication Cost:**

- Bits = all parameters × 16 (entire model)

#### 4. Adam Strategy (Full Finetuning Baseline)

**File**: `sah/algorithms/strategies/adam.py`

Standard Adam optimizer on all model parameters.

**Method**: Traditional full-parameter finetuning with Adam optimizer.

**Hyperparameters:**

```yaml
lr: 1e-5                  # Learning rate
```

**Communication Cost:**

- Bits = all parameters × 16 (entire model)

#### 5. Compressed Finetune Strategy

**File**: `sah/algorithms/strategies/compressed_finetune.py`

Alternative compression approach using batch-wise gradient accumulation and scale learning.

**Method**:

1. **Compression Phase**: Accumulate gradients from multiple batches as perturbations
2. **Optimization Phase**: Learn scale parameters to combine perturbations
3. **Compile Phase**: Merge scaled perturbations into weights periodically

**Hyperparameters:**

```yaml
lr: 1e-5                      # Base learning rate
scale_lr: 1e-5                # Learning rate for scale parameters
grad_accumulation_steps: 6    # Gradients to accumulate
compress_batches_every: 200   # How often to compile into weights
```

**Communication Cost:**

- Bits = (parameters × slots × cycles + tokens × cycles) × 16

### Running Finetune with Strategy Experiments

```bash
# Run with iterative strategy (proposed method)
python sah/main.py experiment=finetune-with-strategy-iterative

# Run with different strategies
python sah/main.py experiment=finetune-with-strategy algorithm/strategy=lora
python sah/main.py experiment=finetune-with-strategy algorithm/strategy=adam
python sah/main.py experiment=finetune-with-strategy algorithm/strategy=compressed

# Override hyperparameters
python sah/main.py experiment=finetune-with-strategy-iterative \
    algorithm.strategy.lr=4e-5 \
    algorithm.strategy.grads_in_memory=3 \
    algorithm.max_examples=1000

# Run on SLURM cluster
sbatch slurm/finetune-with-strategy.sh
```

### Experiment Results

Results are automatically saved to `results.csv` in the output directory with:

- `experiment_name`: Name of the experiment
- `bits`: Communication cost in bits (computed by `strategy.compute_bits()`)

## Testing

Tests are co-located with source code using `*_test.py` naming convention. The project uses pytest with these key features:

- Doctest support enabled
- Custom markers for incremental testing
- CUDA deterministic mode for reproducible tests
