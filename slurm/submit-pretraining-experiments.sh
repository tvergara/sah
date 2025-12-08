#!/bin/bash

echo "Submitting checkpoint experiment combinations"
echo ""

SEEDS=(1)
LR_ONLINE_CODING=(1e-4 1e-5)
LR_FULL_FT=(1e-5 1e-6)
MAX_EXAMPLES=(8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768)

# # SmolLM3-Stage1 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage1 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage1 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage1 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage1 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage1 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage1 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage1 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage1 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage1 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage1 flores icl algorithm.batch_size=1 seed=$seed
# done

# SmolLM3-Stage1 + IFEval + Online Coding
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ONLINE_CODING[@]}"; do
    for max_ex in "${MAX_EXAMPLES[@]}"; do
      echo "SmolLM3-Stage1 + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
      sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
    done
  done
done

# # SmolLM3-Stage1 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage1 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage1 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage1 + MetaMath + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "SmolLM3-Stage1 + MetaMath (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:4 slurm/run-experiment.sh smollm3-stage1 metamath full-ft seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # SmolLM3-Stage1 + FLORES + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "SmolLM3-Stage1 + FLORES (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:4 slurm/run-experiment.sh smollm3-stage1 flores full-ft seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# SmolLM3-Stage1 + IFEval + Full FT
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_FULL_FT[@]}"; do
    echo "SmolLM3-Stage1 + IFEval (full-ft, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval full-ft seed=$seed algorithm.strategy.lr=$lr
  done
done

# # SmolLM3-Stage1 + MetaMath + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "SmolLM3-Stage1 + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage1 metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # SmolLM3-Stage1 + FLORES + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "SmolLM3-Stage1 + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage1 flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# SmolLM3-Stage1 + IFEval + LoRA
for seed in "${SEEDS[@]}"; do
  for lr in "${LR_ONLINE_CODING[@]}"; do
    echo "SmolLM3-Stage1 + IFEval (lora, seed=$seed, lr=$lr)"
    sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval lora seed=$seed algorithm.strategy.lr=$lr
  done
done

# # OLMo3-7B-Step1414k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step1414k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step1414k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step1414k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step1414k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step1414k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step1414k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step1414k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step1414k + MetaMath + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "OLMo3-7B-Step1414k + MetaMath (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo3-7b-step1414k metamath full-ft seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # OLMo3-7B-Step1414k + FLORES + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "OLMo3-7B-Step1414k + FLORES (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh olmo3-7b-step1414k flores full-ft seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # OLMo3-7B-Step1414k + IFEval + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "OLMo3-7B-Step1414k + IFEval (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval full-ft seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # OLMo3-7B-Step1414k + MetaMath + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "OLMo3-7B-Step1414k + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # OLMo3-7B-Step1414k + FLORES + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "OLMo3-7B-Step1414k + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # OLMo3-7B-Step1414k + IFEval + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "OLMo3-7B-Step1414k + IFEval (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

echo ""
echo "All jobs submitted!"
