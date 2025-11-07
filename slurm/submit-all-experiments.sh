#!/bin/bash

echo "Submitting all experiment combinations"
echo ""

echo "Job 1: Qwen + MetaMath (1 GPU, batch_size=8, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath adam

echo "Job 2: Qwen + MetaMath (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath icl algorithm.batch_size=1

echo "Job 3: Qwen + MMLU (1 GPU, batch_size=1, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen mmlu adam

echo "Job 4: Qwen + MMLU (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen mmlu icl algorithm.batch_size=1

echo "Job 5: SmolLM + MetaMath (1 GPU, batch_size=8, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath adam

echo "Job 6: SmolLM + MetaMath (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath icl algorithm.batch_size=1

echo "Job 7: SmolLM + MMLU (1 GPU, batch_size=1, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm mmlu adam

echo "Job 8: SmolLM + MMLU (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm mmlu icl algorithm.batch_size=1

echo "Job 9: SmolLM360 + MetaMath (1 GPU, batch_size=8, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 metamath adam

echo "Job 10: SmolLM360 + MetaMath (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 metamath icl algorithm.batch_size=1

echo "Job 11: SmolLM360 + MMLU (1 GPU, batch_size=1, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 mmlu adam

echo "Job 12: SmolLM360 + MMLU (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 mmlu icl algorithm.batch_size=1

echo "Job 13: Qwen + MetaMath (1 GPU, batch_size=8, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath lora

echo "Job 14: Qwen + MMLU (1 GPU, batch_size=1, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen mmlu lora

echo "Job 15: SmolLM + MetaMath (1 GPU, batch_size=8, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath lora

echo "Job 16: SmolLM + MMLU (1 GPU, batch_size=1, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm mmlu lora

echo "Job 17: SmolLM360 + MetaMath (1 GPU, batch_size=8, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 metamath lora

echo "Job 18: SmolLM360 + MMLU (1 GPU, batch_size=1, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 mmlu lora

echo ""
echo "All jobs submitted! Check status with: squeue -u $USER"
