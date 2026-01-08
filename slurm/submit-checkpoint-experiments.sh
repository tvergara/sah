#!/bin/bash

echo "Submitting checkpoint experiment combinations"
echo ""

SEEDS=(1)
LR_ONLINE_CODING=(1e-4 1e-5)
MAX_EXAMPLES=(1024 2048 4096 8192 16384 32768)

################################################################################
# SmolLM3 (Final)
################################################################################

# # SmolLM3 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 metamath icl algorithm.batch_size=1 seed=$seed
# done

# SmolLM3 + MetaMath + URIAL
# echo "SmolLM3 + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 metamath urial seed=1

# # SmolLM3 + MetaMath + Baseline
# echo "SmolLM3 + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 metamath baseline seed=1

# # SmolLM3 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 flores icl algorithm.batch_size=1 seed=$seed
# done

# SmolLM3 + FLORES + URIAL
# echo "SmolLM3 + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 flores urial seed=1

# # SmolLM3 + FLORES + Baseline
# echo "SmolLM3 + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 flores baseline seed=1

# SmolLM3 + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3 + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3 + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # SmolLM3 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3 + IFEval + URIAL
# echo "SmolLM3 + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 ifeval urial seed=1

# # SmolLM3 + IFEval + Baseline
# echo "SmolLM3 + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3 ifeval baseline seed=1

################################################################################
# SmolLM3-Step40k
################################################################################

# # SmolLM3-Step40k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step40k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step40k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step40k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k metamath icl algorithm.batch_size=1 seed=$seed
# done

# SmolLM3-Step40k + MetaMath + URIAL
# echo "SmolLM3-Step40k + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k metamath urial seed=1

# # SmolLM3-Step40k + MetaMath + Baseline
# echo "SmolLM3-Step40k + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k metamath baseline seed=1

# # SmolLM3-Step40k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step40k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step40k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step40k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step40k + FLORES + URIAL
# echo "SmolLM3-Step40k + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k flores urial seed=1

# # SmolLM3-Step40k + FLORES + Baseline
# echo "SmolLM3-Step40k + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k flores baseline seed=1

# SmolLM3-Step40k + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3-Step40k + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3-Step40k + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # SmolLM3-Step40k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step40k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step40k + IFEval + URIAL
# echo "SmolLM3-Step40k + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k ifeval urial seed=1

# # SmolLM3-Step40k + IFEval + Baseline
# echo "SmolLM3-Step40k + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step40k ifeval baseline seed=1

################################################################################
# SmolLM3-Step1720k
################################################################################

# # SmolLM3-Step1720k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step1720k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step1720k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step1720k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step1720k + MetaMath + URIAL
# echo "SmolLM3-Step1720k + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k metamath urial seed=1

# # SmolLM3-Step1720k + MetaMath + Baseline
# echo "SmolLM3-Step1720k + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k metamath baseline seed=1

# # SmolLM3-Step1720k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step1720k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step1720k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step1720k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step1720k + FLORES + URIAL
# echo "SmolLM3-Step1720k + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k flores urial seed=1

# # SmolLM3-Step1720k + FLORES + Baseline
# echo "SmolLM3-Step1720k + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k flores baseline seed=1

# SmolLM3-Step1720k + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3-Step1720k + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3-Step1720k + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # SmolLM3-Step1720k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step1720k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step1720k + IFEval + URIAL
# echo "SmolLM3-Step1720k + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k ifeval urial seed=1

# # SmolLM3-Step1720k + IFEval + Baseline
# echo "SmolLM3-Step1720k + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-step1720k ifeval baseline seed=1

################################################################################
# SmolLM3-Stage2
################################################################################

# # SmolLM3-Stage2 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage2 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage2 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage2 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage2 + MetaMath + URIAL
# echo "SmolLM3-Stage2 + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 metamath urial seed=1

# # SmolLM3-Stage2 + MetaMath + Baseline
# echo "SmolLM3-Stage2 + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 metamath baseline seed=1

# # SmolLM3-Stage2 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage2 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage2 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage2 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage2 + FLORES + URIAL
# echo "SmolLM3-Stage2 + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 flores urial seed=1

# # SmolLM3-Stage2 + FLORES + Baseline
# echo "SmolLM3-Stage2 + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 flores baseline seed=1

# SmolLM3-Stage2 + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3-Stage2 + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3-Stage2 + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # SmolLM3-Stage2 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage2 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage2 + IFEval + URIAL
# echo "SmolLM3-Stage2 + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 ifeval urial seed=1

# # SmolLM3-Stage2 + IFEval + Baseline
# echo "SmolLM3-Stage2 + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage2 ifeval baseline seed=1

################################################################################
# SmolLM3-Stage3
################################################################################

# # SmolLM3-Stage3 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage3 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage3 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage3 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage3 + MetaMath + URIAL
# echo "SmolLM3-Stage3 + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 metamath urial seed=1

# # SmolLM3-Stage3 + MetaMath + Baseline
# echo "SmolLM3-Stage3 + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 metamath baseline seed=1

# # SmolLM3-Stage3 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage3 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage3 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage3 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage3 + FLORES + URIAL
# echo "SmolLM3-Stage3 + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 flores urial seed=1

# # SmolLM3-Stage3 + FLORES + Baseline
# echo "SmolLM3-Stage3 + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 flores baseline seed=1

# SmolLM3-Stage3 + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3-Stage3 + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "SmolLM3-Stage3 + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # SmolLM3-Stage3 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage3 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage3 + IFEval + URIAL
# echo "SmolLM3-Stage3 + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 ifeval urial seed=1

# # SmolLM3-Stage3 + IFEval + Baseline
# echo "SmolLM3-Stage3 + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh smollm3-stage3 ifeval baseline seed=1

################################################################################
# OLMo3-7B-Step0
################################################################################

# # OLMo3-7B-Step0 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step0 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step0 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step0 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step0 + MetaMath + URIAL
# echo "OLMo3-7B-Step0 + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 metamath urial seed=1

# # OLMo3-7B-Step0 + MetaMath + Baseline
# echo "OLMo3-7B-Step0 + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 metamath baseline seed=1

# # OLMo3-7B-Step0 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step0 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step0 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step0 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step0 + FLORES + URIAL
# echo "OLMo3-7B-Step0 + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 flores urial seed=1

# # OLMo3-7B-Step0 + FLORES + Baseline
# echo "OLMo3-7B-Step0 + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 flores baseline seed=1

# OLMo3-7B-Step0 + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Step0 + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Step0 + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# OLMo3-7B-Step0 + IFEval + ICL
echo "OLMo3-7B-Step0 + IFEval (icl, seed=1)"
sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 ifeval icl algorithm.batch_size=1 seed=1

# # OLMo3-7B-Step0 + IFEval + URIAL
# echo "OLMo3-7B-Step0 + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 ifeval urial seed=1

# # OLMo3-7B-Step0 + IFEval + Baseline
# echo "OLMo3-7B-Step0 + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step0 ifeval baseline seed=1

################################################################################
# OLMo3-7B-Step707k
################################################################################

# # OLMo3-7B-Step707k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step707k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step707k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step707k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step707k + MetaMath + URIAL
# echo "OLMo3-7B-Step707k + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k metamath urial seed=1

# # OLMo3-7B-Step707k + MetaMath + Baseline
# echo "OLMo3-7B-Step707k + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k metamath baseline seed=1

# # OLMo3-7B-Step707k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step707k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step707k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step707k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step707k + FLORES + URIAL
# echo "OLMo3-7B-Step707k + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k flores urial seed=1

# # OLMo3-7B-Step707k + FLORES + Baseline
# echo "OLMo3-7B-Step707k + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k flores baseline seed=1

# OLMo3-7B-Step707k + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Step707k + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Step707k + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # OLMo3-7B-Step707k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step707k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step707k + IFEval + URIAL
# echo "OLMo3-7B-Step707k + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k ifeval urial seed=1

# # OLMo3-7B-Step707k + IFEval + Baseline
# echo "OLMo3-7B-Step707k + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-step707k ifeval baseline seed=1

################################################################################
# OLMo3-7B-Stage2-Step6k
################################################################################

# # OLMo3-7B-Stage2-Step6k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step6k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step6k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step6k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step6k + MetaMath + URIAL
# echo "OLMo3-7B-Stage2-Step6k + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k metamath urial seed=1

# # OLMo3-7B-Stage2-Step6k + MetaMath + Baseline
# echo "OLMo3-7B-Stage2-Step6k + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k metamath baseline seed=1

# # OLMo3-7B-Stage2-Step6k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step6k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step6k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step6k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step6k + FLORES + URIAL
# echo "OLMo3-7B-Stage2-Step6k + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k flores urial seed=1

# # OLMo3-7B-Stage2-Step6k + FLORES + Baseline
# echo "OLMo3-7B-Stage2-Step6k + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k flores baseline seed=1

# OLMo3-7B-Stage2-Step6k + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Stage2-Step6k + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Stage2-Step6k + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# OLMo3-7B-Stage2-Step6k + IFEval + ICL
echo "OLMo3-7B-Stage2-Step6k + IFEval (icl, seed=1)"
sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k ifeval icl algorithm.batch_size=1 seed=1

# # OLMo3-7B-Stage2-Step6k + IFEval + URIAL
# echo "OLMo3-7B-Stage2-Step6k + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k ifeval urial seed=1

# # OLMo3-7B-Stage2-Step6k + IFEval + Baseline
# echo "OLMo3-7B-Stage2-Step6k + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k ifeval baseline seed=1

################################################################################
# OLMo3-7B-Stage2-Step12k
################################################################################

# # OLMo3-7B-Stage2-Step12k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step12k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step12k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step12k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step12k + MetaMath + URIAL
# echo "OLMo3-7B-Stage2-Step12k + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k metamath urial seed=1

# # OLMo3-7B-Stage2-Step12k + MetaMath + Baseline
# echo "OLMo3-7B-Stage2-Step12k + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k metamath baseline seed=1

# # OLMo3-7B-Stage2-Step12k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step12k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step12k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step12k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step12k + FLORES + URIAL
# echo "OLMo3-7B-Stage2-Step12k + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k flores urial seed=1

# # OLMo3-7B-Stage2-Step12k + FLORES + Baseline
# echo "OLMo3-7B-Stage2-Step12k + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k flores baseline seed=1

# OLMo3-7B-Stage2-Step12k + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Stage2-Step12k + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Stage2-Step12k + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # OLMo3-7B-Stage2-Step12k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step12k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step12k + IFEval + URIAL
# echo "OLMo3-7B-Stage2-Step12k + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k ifeval urial seed=1

# # OLMo3-7B-Stage2-Step12k + IFEval + Baseline
# echo "OLMo3-7B-Stage2-Step12k + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k ifeval baseline seed=1

################################################################################
# OLMo3-7B-Stage2-Step24k
################################################################################

# # OLMo3-7B-Stage2-Step24k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step24k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step24k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step24k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step24k + MetaMath + URIAL
# echo "OLMo3-7B-Stage2-Step24k + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k metamath urial seed=1

# # OLMo3-7B-Stage2-Step24k + MetaMath + Baseline
# echo "OLMo3-7B-Stage2-Step24k + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k metamath baseline seed=1

# # OLMo3-7B-Stage2-Step24k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step24k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step24k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step24k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step24k + FLORES + URIAL
# echo "OLMo3-7B-Stage2-Step24k + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k flores urial seed=1

# # OLMo3-7B-Stage2-Step24k + FLORES + Baseline
# echo "OLMo3-7B-Stage2-Step24k + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k flores baseline seed=1

# OLMo3-7B-Stage2-Step24k + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Stage2-Step24k + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Stage2-Step24k + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# OLMo3-7B-Stage2-Step24k + IFEval + ICL
echo "OLMo3-7B-Stage2-Step24k + IFEval (icl, seed=1)"
sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k ifeval icl algorithm.batch_size=1 seed=1

# # OLMo3-7B-Stage2-Step24k + IFEval + URIAL
# echo "OLMo3-7B-Stage2-Step24k + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k ifeval urial seed=1

# # OLMo3-7B-Stage2-Step24k + IFEval + Baseline
# echo "OLMo3-7B-Stage2-Step24k + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k ifeval baseline seed=1

################################################################################
# OLMo3-7B-Stage2-Step48k
################################################################################

# # OLMo3-7B-Stage2-Step48k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step48k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step48k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step48k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step48k + MetaMath + URIAL
# echo "OLMo3-7B-Stage2-Step48k + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k metamath urial seed=1

# # OLMo3-7B-Stage2-Step48k + MetaMath + Baseline
# echo "OLMo3-7B-Stage2-Step48k + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k metamath baseline seed=1

# # OLMo3-7B-Stage2-Step48k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step48k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step48k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step48k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step48k + FLORES + URIAL
# echo "OLMo3-7B-Stage2-Step48k + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k flores urial seed=1

# # OLMo3-7B-Stage2-Step48k + FLORES + Baseline
# echo "OLMo3-7B-Stage2-Step48k + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k flores baseline seed=1

# OLMo3-7B-Stage2-Step48k + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Stage2-Step48k + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Stage2-Step48k + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # OLMo3-7B-Stage2-Step48k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step48k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step48k + IFEval + URIAL
# echo "OLMo3-7B-Stage2-Step48k + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k ifeval urial seed=1

# # OLMo3-7B-Stage2-Step48k + IFEval + Baseline
# echo "OLMo3-7B-Stage2-Step48k + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k ifeval baseline seed=1

################################################################################
# OLMo3-7B-Instruct-Step200
################################################################################

# # OLMo3-7B-Instruct-Step200 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Instruct-Step200 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Instruct-Step200 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Instruct-Step200 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Instruct-Step200 + MetaMath + URIAL
# echo "OLMo3-7B-Instruct-Step200 + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 metamath urial seed=1

# # OLMo3-7B-Instruct-Step200 + MetaMath + Baseline
# echo "OLMo3-7B-Instruct-Step200 + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 metamath baseline seed=1

# # OLMo3-7B-Instruct-Step200 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Instruct-Step200 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Instruct-Step200 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Instruct-Step200 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Instruct-Step200 + FLORES + URIAL
# echo "OLMo3-7B-Instruct-Step200 + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 flores urial seed=1

# # OLMo3-7B-Instruct-Step200 + FLORES + Baseline
# echo "OLMo3-7B-Instruct-Step200 + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 flores baseline seed=1

# OLMo3-7B-Instruct-Step200 + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Instruct-Step200 + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Instruct-Step200 + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # OLMo3-7B-Instruct-Step200 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Instruct-Step200 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Instruct-Step200 + IFEval + URIAL
# echo "OLMo3-7B-Instruct-Step200 + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 ifeval urial seed=1

# # OLMo3-7B-Instruct-Step200 + IFEval + Baseline
# echo "OLMo3-7B-Instruct-Step200 + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step200 ifeval baseline seed=1

################################################################################
# OLMo3-7B-Instruct-Step400
################################################################################

# # OLMo3-7B-Instruct-Step400 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Instruct-Step400 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Instruct-Step400 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Instruct-Step400 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Instruct-Step400 + MetaMath + URIAL
# echo "OLMo3-7B-Instruct-Step400 + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 metamath urial seed=1

# # OLMo3-7B-Instruct-Step400 + MetaMath + Baseline
# echo "OLMo3-7B-Instruct-Step400 + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 metamath baseline seed=1

# # OLMo3-7B-Instruct-Step400 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Instruct-Step400 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Instruct-Step400 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Instruct-Step400 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Instruct-Step400 + FLORES + URIAL
# echo "OLMo3-7B-Instruct-Step400 + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 flores urial seed=1

# # OLMo3-7B-Instruct-Step400 + FLORES + Baseline
# echo "OLMo3-7B-Instruct-Step400 + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 flores baseline seed=1

# OLMo3-7B-Instruct-Step400 + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Instruct-Step400 + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-7B-Instruct-Step400 + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# # OLMo3-7B-Instruct-Step400 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Instruct-Step400 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Instruct-Step400 + IFEval + URIAL
# echo "OLMo3-7B-Instruct-Step400 + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 ifeval urial seed=1

# # OLMo3-7B-Instruct-Step400 + IFEval + Baseline
# echo "OLMo3-7B-Instruct-Step400 + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-step400 ifeval baseline seed=1

################################################################################
# OLMo3-7B-Instruct-Final
################################################################################

# # OLMo3-7B-Instruct-Final + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Instruct-Final + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Instruct-Final + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Instruct-Final + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Instruct-Final + MetaMath + URIAL
# echo "OLMo3-7B-Instruct-Final + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final metamath urial seed=1

# # OLMo3-7B-Instruct-Final + MetaMath + Baseline
# echo "OLMo3-7B-Instruct-Final + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final metamath baseline seed=1

# # OLMo3-7B-Instruct-Final + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Instruct-Final + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Instruct-Final + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Instruct-Final + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Instruct-Final + FLORES + URIAL
# echo "OLMo3-7B-Instruct-Final + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final flores urial seed=1

# # OLMo3-7B-Instruct-Final + FLORES + Baseline
# echo "OLMo3-7B-Instruct-Final + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final flores baseline seed=1

# # OLMo3-7B-Instruct-Final + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Instruct-Final + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Instruct-Final + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Instruct-Final + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Instruct-Final + IFEval + URIAL
# echo "OLMo3-7B-Instruct-Final + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final ifeval urial seed=1

# # OLMo3-7B-Instruct-Final + IFEval + Baseline
# echo "OLMo3-7B-Instruct-Final + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-7b-instruct-final ifeval baseline seed=1

################################################################################
# OLMo3-1025-7B
################################################################################

# # OLMo3-1025-7B + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-1025-7B + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# OLMo3-1025-7B + MetaMath + Online Coding (PARTIAL - one job missing)
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-1025-7B + MetaMath (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b metamath online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

# # OLMo3-1025-7B + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-1025-7B + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-1025-7B + MetaMath + URIAL
# echo "OLMo3-1025-7B + MetaMath (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b metamath urial seed=1

# # OLMo3-1025-7B + MetaMath + Baseline
# echo "OLMo3-1025-7B + MetaMath (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b metamath baseline seed=1

# # OLMo3-1025-7B + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-1025-7B + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-1025-7B + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-1025-7B + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-1025-7B + FLORES + URIAL
# echo "OLMo3-1025-7B + FLORES (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b flores urial seed=1

# # OLMo3-1025-7B + FLORES + Baseline
# echo "OLMo3-1025-7B + FLORES (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b flores baseline seed=1

# OLMo3-1025-7B + IFEval + Online Coding
for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-1025-7B + IFEval (online-coding, seed=1, lr=1e-4, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b ifeval online-coding seed=1 algorithm.strategy.lr=1e-4 algorithm.max_examples=$max_ex
done

for max_ex in "${MAX_EXAMPLES[@]}"; do
  echo "OLMo3-1025-7B + IFEval (online-coding, seed=1, lr=1e-5, max_examples=$max_ex)"
  sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b ifeval online-coding seed=1 algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex
done

# OLMo3-1025-7B + IFEval + ICL
echo "OLMo3-1025-7B + IFEval (icl, seed=1)"
sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b ifeval icl algorithm.batch_size=1 seed=1

# # OLMo3-1025-7B + IFEval + URIAL
# echo "OLMo3-1025-7B + IFEval (urial)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b ifeval urial seed=1

# # OLMo3-1025-7B + IFEval + Baseline
# echo "OLMo3-1025-7B + IFEval (baseline)"
# sbatch --gres=gpu:h100:1 slurm/run-experiment.sh olmo3-1025-7b ifeval baseline seed=1

echo ""
echo "All jobs submitted!"
