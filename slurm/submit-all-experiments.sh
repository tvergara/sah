#!/bin/bash

echo "Submitting all experiment combinations"
echo ""

# Hyperparameter sweep configuration
SEEDS=(1 2)
LR_ADAM=(1e-4 1e-5 1e-6)
LR_LORA=(1e-3 1e-4 1e-5)

# # Job 1: Qwen + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 1: Qwen + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 2: Qwen + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 2: Qwen + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 3: SmolLM + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 3: SmolLM + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 4: SmolLM + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 4: SmolLM + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 5: Qwen + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 5: Qwen + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 6: SmolLM + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 6: SmolLM + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 7: Qwen + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 7: Qwen + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 8: Qwen + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 8: Qwen + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 9: Qwen + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 9: Qwen + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 10: SmolLM + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 10: SmolLM + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 11: SmolLM + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 11: SmolLM + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 12: SmolLM + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 12: SmolLM + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 13: SmolLM-step2750k + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 13: SmolLM-step2750k + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 14: SmolLM-step2750k + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 14: SmolLM-step2750k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 15: SmolLM-step2750k + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 15: SmolLM-step2750k + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 16: SmolLM-step2750k + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 16: SmolLM-step2750k + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 17: SmolLM-step2750k + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 17: SmolLM-step2750k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 18: SmolLM-step2750k + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 18: SmolLM-step2750k + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 19: SmolLM-step4125k + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 19: SmolLM-step4125k + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 20: SmolLM-step4125k + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 20: SmolLM-step4125k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 21: SmolLM-step4125k + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 21: SmolLM-step4125k + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 22: SmolLM-step4125k + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 22: SmolLM-step4125k + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 23: SmolLM-step4125k + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 23: SmolLM-step4125k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 24: SmolLM-step4125k + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 24: SmolLM-step4125k + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 25: SmolLM-step4875k + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 25: SmolLM-step4875k + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 26: SmolLM-step4875k + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 26: SmolLM-step4875k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 27: SmolLM-step4875k + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 27: SmolLM-step4875k + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 28: SmolLM-step4875k + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 28: SmolLM-step4875k + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 29: SmolLM-step4875k + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 29: SmolLM-step4875k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 30: SmolLM-step4875k + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 30: SmolLM-step4875k + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 31: Hubble-1B-500B-Standard + PiQA (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 31: Hubble-1B-500B-Standard + PiQA (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh hubble-1b-500b-standard piqa adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 32: Hubble-1B-500B-Standard + PiQA (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 32: Hubble-1B-500B-Standard + PiQA (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh hubble-1b-500b-standard piqa icl algorithm.batch_size=1 seed=$seed
# done

# # Job 33: Hubble-1B-500B-Standard + PiQA (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 33: Hubble-1B-500B-Standard + PiQA (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh hubble-1b-500b-standard piqa lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 34: Hubble-1B-500B-Perturbed + PiQA (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 34: Hubble-1B-500B-Perturbed + PiQA (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh hubble-1b-500b-perturbed piqa adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 35: Hubble-1B-500B-Perturbed + PiQA (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 35: Hubble-1B-500B-Perturbed + PiQA (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh hubble-1b-500b-perturbed piqa icl algorithm.batch_size=1 seed=$seed
# done

# # Job 36: Hubble-1B-500B-Perturbed + PiQA (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 36: Hubble-1B-500B-Perturbed + PiQA (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh hubble-1b-500b-perturbed piqa lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 37: SmolLM3 + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 37: SmolLM3 + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3 metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 38: SmolLM3 + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 38: SmolLM3 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 39: SmolLM3 + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 39: SmolLM3 + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3 metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 40: SmolLM3 + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 40: SmolLM3 + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3 flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 41: SmolLM3 + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 41: SmolLM3 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3 flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 42: SmolLM3 + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 42: SmolLM3 + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3 flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 43: OLMo2-7B + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 43: OLMo2-7B + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh olmo2-7b metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 44: OLMo2-7B + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 44: OLMo2-7B + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-7b metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 45: OLMo2-7B + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 45: OLMo2-7B + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo2-7b metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 46: OLMo2-7B + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 46: OLMo2-7B + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh olmo2-7b flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 47: OLMo2-7B + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 47: OLMo2-7B + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-7b flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 48: OLMo2-7B + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 48: OLMo2-7B + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo2-7b flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 49: OLMo2-7B-step464k + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 49: OLMo2-7B-step464k + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh olmo2-7b-step464k metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 50: OLMo2-7B-step464k + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 50: OLMo2-7B-step464k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-7b-step464k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 51: OLMo2-7B-step464k + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 51: OLMo2-7B-step464k + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo2-7b-step464k metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 52: OLMo2-7B-step464k + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 52: OLMo2-7B-step464k + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh olmo2-7b-step464k flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 53: OLMo2-7B-step464k + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 53: OLMo2-7B-step464k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-7b-step464k flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 54: OLMo2-7B-step464k + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 54: OLMo2-7B-step464k + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo2-7b-step464k flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 55: OLMo2-7B-step696k + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 55: OLMo2-7B-step696k + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh olmo2-7b-step696k metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 56: OLMo2-7B-step696k + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 56: OLMo2-7B-step696k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-7b-step696k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 57: OLMo2-7B-step696k + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 57: OLMo2-7B-step696k + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo2-7b-step696k metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 58: OLMo2-7B-step696k + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 58: OLMo2-7B-step696k + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh olmo2-7b-step696k flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 59: OLMo2-7B-step696k + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 59: OLMo2-7B-step696k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-7b-step696k flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 60: OLMo2-7B-step696k + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 60: OLMo2-7B-step696k + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo2-7b-step696k flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 61: OLMo2-7B-step812k + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 61: OLMo2-7B-step812k + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh olmo2-7b-step812k metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 62: OLMo2-7B-step812k + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 62: OLMo2-7B-step812k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-7b-step812k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 63: OLMo2-7B-step812k + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 63: OLMo2-7B-step812k + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo2-7b-step812k metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 64: OLMo2-7B-step812k + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 64: OLMo2-7B-step812k + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:3 slurm/run-experiment.sh olmo2-7b-step812k flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 65: OLMo2-7B-step812k + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 65: OLMo2-7B-step812k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-7b-step812k flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 66: OLMo2-7B-step812k + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 66: OLMo2-7B-step812k + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo2-7b-step812k flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 67: SmolLM3-stage1 + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 67: SmolLM3-stage1 + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 68: SmolLM3-stage1 + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 68: SmolLM3-stage1 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 69: SmolLM3-stage1 + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 69: SmolLM3-stage1 + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 70: SmolLM3-stage1 + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 70: SmolLM3-stage1 + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 71: SmolLM3-stage1 + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 71: SmolLM3-stage1 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 72: SmolLM3-stage1 + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 72: SmolLM3-stage1 + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 73: SmolLM3-stage2 + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 73: SmolLM3-stage2 + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage2 metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 74: SmolLM3-stage2 + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 74: SmolLM3-stage2 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage2 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 75: SmolLM3-stage2 + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 75: SmolLM3-stage2 + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage2 metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 76: SmolLM3-stage2 + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 76: SmolLM3-stage2 + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage2 flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 77: SmolLM3-stage2 + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 77: SmolLM3-stage2 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage2 flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 78: SmolLM3-stage2 + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 78: SmolLM3-stage2 + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage2 flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 79: SmolLM3-stage3 + MetaMath (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 79: SmolLM3-stage3 + MetaMath (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage3 metamath adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 80: SmolLM3-stage3 + MetaMath (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 80: SmolLM3-stage3 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage3 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # Job 81: SmolLM3-stage3 + MetaMath (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 81: SmolLM3-stage3 + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage3 metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 82: SmolLM3-stage3 + FLORES (adam) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ADAM[@]}"; do
#     echo "Job 82: SmolLM3-stage3 + FLORES (adam, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage3 flores adam seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # Job 83: SmolLM3-stage3 + FLORES (icl) with seed sweep
# for seed in "${SEEDS[@]}"; do
#   echo "Job 83: SmolLM3-stage3 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage3 flores icl algorithm.batch_size=1 seed=$seed
# done

# # Job 84: SmolLM3-stage3 + FLORES (lora) with hyperparameter sweep
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_LORA[@]}"; do
#     echo "Job 84: SmolLM3-stage3 + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage3 flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# Job 85: SmolLM + LIMA (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 85: SmolLM + LIMA (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm lima adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 86: SmolLM + LIMA (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 86: SmolLM + LIMA (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm lima icl algorithm.batch_size=1 seed=$seed
done

# Job 87: SmolLM + LIMA (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 87: SmolLM + LIMA (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm lima lora seed=$seed algorithm.strategy.lr=$lr
  done
done

echo ""
echo "All jobs submitted! Check status with: squeue -u $USER"
