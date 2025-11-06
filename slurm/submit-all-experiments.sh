#!/bin/bash

STRATEGY=${1:-adam}

echo "Submitting all experiment combinations with strategy: $STRATEGY"
echo ""

# echo "Job 1: Qwen + MetaMath (1 GPU, batch_size=8)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath $STRATEGY

# echo "Job 2: Qwen + OpenThoughts (3 GPUs, batch_size=2)"
# sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh qwen openthoughts $STRATEGY algorithm.batch_size=2

# echo "Job 3: SmolLM + MetaMath (1 GPU, batch_size=8)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath $STRATEGY

# echo "Job 4: SmolLM + OpenThoughts (3 GPUs, batch_size=2)"
# sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh smollm openthoughts $STRATEGY algorithm.batch_size=2

echo "Job 5: Qwen + MetaMath (1 GPU, batch_size=1, ICL)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath icl algorithm.batch_size=1

echo "Job 6: SmolLM + MetaMath (1 GPU, batch_size=1, ICL)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath icl algorithm.batch_size=1

echo ""
echo "All jobs submitted! Check status with: squeue -u $USER"
