#!/bin/bash

echo "Submitting all experiment combinations"
echo ""

# Hyperparameter sweep configuration
SEEDS=(1 2)
LR_ADAM=(1e-4 1e-5 1e-6)
LR_LORA=(1e-3 1e-4 1e-5)

# Job 1: Qwen + MetaMath (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 1: Qwen + MetaMath (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 2: Qwen + MetaMath (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 2: Qwen + MetaMath (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath icl algorithm.batch_size=1 seed=$seed
done

# Job 3: SmolLM + MetaMath (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 3: SmolLM + MetaMath (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 4: SmolLM + MetaMath (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 4: SmolLM + MetaMath (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath icl algorithm.batch_size=1 seed=$seed
done

# Job 5: Qwen + MetaMath (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 5: Qwen + MetaMath (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 6: SmolLM + MetaMath (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 6: SmolLM + MetaMath (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 7: Qwen + FLORES (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 7: Qwen + FLORES (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen flores adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 8: Qwen + FLORES (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 8: Qwen + FLORES (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen flores icl algorithm.batch_size=1 seed=$seed
done

# Job 9: Qwen + FLORES (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 9: Qwen + FLORES (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen flores lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 10: SmolLM + FLORES (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 10: SmolLM + FLORES (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm flores adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 11: SmolLM + FLORES (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 11: SmolLM + FLORES (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm flores icl algorithm.batch_size=1 seed=$seed
done

# Job 12: SmolLM + FLORES (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 12: SmolLM + FLORES (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm flores lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 13: SmolLM-step2750k + MetaMath (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 13: SmolLM-step2750k + MetaMath (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k metamath adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 14: SmolLM-step2750k + MetaMath (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 14: SmolLM-step2750k + MetaMath (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k metamath icl algorithm.batch_size=1 seed=$seed
done

# Job 15: SmolLM-step2750k + MetaMath (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 15: SmolLM-step2750k + MetaMath (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k metamath lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 16: SmolLM-step2750k + FLORES (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 16: SmolLM-step2750k + FLORES (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k flores adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 17: SmolLM-step2750k + FLORES (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 17: SmolLM-step2750k + FLORES (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k flores icl algorithm.batch_size=1 seed=$seed
done

# Job 18: SmolLM-step2750k + FLORES (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 18: SmolLM-step2750k + FLORES (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k flores lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 19: SmolLM-step4125k + MetaMath (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 19: SmolLM-step4125k + MetaMath (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k metamath adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 20: SmolLM-step4125k + MetaMath (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 20: SmolLM-step4125k + MetaMath (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k metamath icl algorithm.batch_size=1 seed=$seed
done

# Job 21: SmolLM-step4125k + MetaMath (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 21: SmolLM-step4125k + MetaMath (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k metamath lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 22: SmolLM-step4125k + FLORES (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 22: SmolLM-step4125k + FLORES (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k flores adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 23: SmolLM-step4125k + FLORES (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 23: SmolLM-step4125k + FLORES (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k flores icl algorithm.batch_size=1 seed=$seed
done

# Job 24: SmolLM-step4125k + FLORES (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 24: SmolLM-step4125k + FLORES (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k flores lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 25: SmolLM-step4875k + MetaMath (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 25: SmolLM-step4875k + MetaMath (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k metamath adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 26: SmolLM-step4875k + MetaMath (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 26: SmolLM-step4875k + MetaMath (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k metamath icl algorithm.batch_size=1 seed=$seed
done

# Job 27: SmolLM-step4875k + MetaMath (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 27: SmolLM-step4875k + MetaMath (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k metamath lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 28: SmolLM-step4875k + FLORES (adam) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ADAM[@]}"; do
    echo "Job 28: SmolLM-step4875k + FLORES (adam, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k flores adam seed=$seed algorithm.strategy.lr=$lr
  done
done

# Job 29: SmolLM-step4875k + FLORES (icl) with seed sweep
for seed in "${SEEDS[@]}"; do
  echo "Job 29: SmolLM-step4875k + FLORES (icl, seed=$seed)"
  sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k flores icl algorithm.batch_size=1 seed=$seed
done

# Job 30: SmolLM-step4875k + FLORES (lora) with hyperparameter sweep
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_LORA[@]}"; do
    echo "Job 30: SmolLM-step4875k + FLORES (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k flores lora seed=$seed algorithm.strategy.lr=$lr
  done
done

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

echo ""
echo "All jobs submitted! Check status with: squeue -u $USER"
