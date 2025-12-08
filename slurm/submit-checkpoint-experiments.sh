#!/bin/bash

echo "Submitting checkpoint experiment combinations"
echo ""

SEEDS=(1)
LR_ONLINE_CODING=(1e-4 1e-5)
MAX_EXAMPLES=(8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768)

# # SmolLM3 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3 flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3 + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3 + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3 ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step40k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step40k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step40k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step40k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step40k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step40k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step40k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step40k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step40k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step40k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step40k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step40k flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step40k + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step40k + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step40k ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step40k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step40k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step40k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step1720k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step1720k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step1720k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step1720k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step1720k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step1720k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step1720k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step1720k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step1720k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step1720k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step1720k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step1720k flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Step1720k + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Step1720k + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step1720k ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Step1720k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Step1720k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-step1720k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage2 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage2 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage2 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage2 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage2 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage2 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage2 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage2 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage2 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage2 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage2 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage2 flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage2 + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage2 + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage2 ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage2 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage2 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage2 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage3 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage3 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage3 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage3 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage3 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage3 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage3 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage3 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage3 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage3 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage3 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage3 flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage3 + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage3 + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage3 ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage3 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage3 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage3 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step0 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step0 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step0 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step0 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step0 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step0 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step0 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step0 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step0 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step0 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step0 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step0 flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step0 + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step0 + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step0 ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step0 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step0 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step0 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step707k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step707k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step707k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step707k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step707k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step707k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step707k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step707k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step707k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step707k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step707k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step707k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step707k + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step707k + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step707k ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step707k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step707k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step707k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step6k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step6k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step6k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step6k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step6k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step6k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step6k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step6k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step6k + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step6k + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step6k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step6k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step6k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step12k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step12k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step12k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step12k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step12k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step12k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step12k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step12k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step12k + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step12k + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step12k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step12k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step12k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step24k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step24k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step24k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step24k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step24k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step24k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step24k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step24k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step24k + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step24k + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step24k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step24k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step24k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step48k + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step48k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step48k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step48k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k metamath icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step48k + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step48k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step48k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step48k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k flores icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Stage2-Step48k + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Stage2-Step48k + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Stage2-Step48k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Stage2-Step48k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-stage2-step48k ifeval icl algorithm.batch_size=1 seed=$seed
# done

echo ""
echo "All jobs submitted!"
