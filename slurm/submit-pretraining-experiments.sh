#!/bin/bash

echo "Submitting checkpoint experiment combinations"
echo ""

SEEDS=(1)
LR_ONLINE_CODING=(1e-4 1e-5)
LR_FULL_FT=(1e-5 1e-6)
LR_PHASE_TWO=(1e-4 1e-5)
MAX_EXAMPLES=(8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768)
GRADS_IN_MEMORY=(8 32 64)
SCALES_LR=(1e-2 2e-3)

################################################################################
# SmolLM3-Stage1
################################################################################

# # SmolLM3-Stage1 + MetaMath + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage1 + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage1 + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage1 + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage1 + MetaMath + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "SmolLM3-Stage1 + MetaMath (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath full-ft seed=$seed algorithm.strategy.lr=$lr algorithm.batch_size=4 trainer.accumulate_grad_batches=2
#   done
# done

# # SmolLM3-Stage1 + MetaMath + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "SmolLM3-Stage1 + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # SmolLM3-Stage1 + MetaMath + Baseline
# echo "SmolLM3-Stage1 + MetaMath (baseline)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath baseline seed=1

# # SmolLM3-Stage1 + MetaMath + URIAL
# echo "SmolLM3-Stage1 + MetaMath (urial)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath urial seed=1

# # SmolLM3-Stage1 + MetaMath + LM Head
# echo "SmolLM3-Stage1 + MetaMath (lm-head)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath lm-head seed=1

# # SmolLM3-Stage1 + MetaMath + BLoRA
# for seed in "${SEEDS[@]}"; do
#   for scales_lr in "${SCALES_LR[@]}"; do
#     echo "SmolLM3-Stage1 + MetaMath (blora, seed=$seed, scales_lr=$scales_lr, r=1, prune_rank=False)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh smollm3-stage1 metamath blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=1 algorithm.strategy.prune_rank=False
#
#     echo "SmolLM3-Stage1 + MetaMath (blora, seed=$seed, scales_lr=$scales_lr, r=2, prune_rank=True)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh smollm3-stage1 metamath blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=2 algorithm.strategy.prune_rank=True
#   done
# done

# # SmolLM3-Stage1 + MetaMath + Phase One
# echo "SmolLM3-Stage1 + MetaMath (phase-one)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath phase-one seed=1

# # SmolLM3-Stage1 + MetaMath + Phase Two
# for lr in "${LR_PHASE_TWO[@]}"; do
#   for grads in "${GRADS_IN_MEMORY[@]}"; do
#     echo "SmolLM3-Stage1 + MetaMath (phase-two, lr=$lr, grads_in_memory=$grads)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 metamath phase-two seed=1 algorithm.strategy.lr=$lr algorithm.strategy.grads_in_memory=$grads
#   done
# done

# # SmolLM3-Stage1 + FLORES + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage1 + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage1 + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage1 + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage1 + FLORES + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "SmolLM3-Stage1 + FLORES (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores full-ft seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # SmolLM3-Stage1 + FLORES + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "SmolLM3-Stage1 + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # SmolLM3-Stage1 + FLORES + Baseline
# echo "SmolLM3-Stage1 + FLORES (baseline)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores baseline seed=1

# # SmolLM3-Stage1 + FLORES + URIAL
# echo "SmolLM3-Stage1 + FLORES (urial)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores urial seed=1

# # SmolLM3-Stage1 + FLORES + LM Head
# echo "SmolLM3-Stage1 + FLORES (lm-head)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores lm-head seed=1

# # SmolLM3-Stage1 + FLORES + BLoRA
# for seed in "${SEEDS[@]}"; do
#   for scales_lr in "${SCALES_LR[@]}"; do
#     echo "SmolLM3-Stage1 + FLORES (blora, seed=$seed, scales_lr=$scales_lr, r=1, prune_rank=False)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh smollm3-stage1 flores blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=1 algorithm.strategy.prune_rank=False

#     echo "SmolLM3-Stage1 + FLORES (blora, seed=$seed, scales_lr=$scales_lr, r=2, prune_rank=True)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh smollm3-stage1 flores blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=2 algorithm.strategy.prune_rank=True
#   done
# done

# # SmolLM3-Stage1 + FLORES + Phase One
# echo "SmolLM3-Stage1 + FLORES (phase-one)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 flores phase-one seed=1

# # SmolLM3-Stage1 + FLORES + Phase Two
# for lr in "${LR_PHASE_TWO[@]}"; do
#   for grads in "${GRADS_IN_MEMORY[@]}"; do
#     echo "SmolLM3-Stage1 + FLORES (phase-two, lr=$lr, grads_in_memory=$grads)"
#     sbatch --gres=gpu:1 slurm/run-experiment.sh smollm3-stage1 flores phase-two seed=1 algorithm.strategy.lr=$lr algorithm.strategy.grads_in_memory=$grads
#   done
# done

# # SmolLM3-Stage1 + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "SmolLM3-Stage1 + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # SmolLM3-Stage1 + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "SmolLM3-Stage1 + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # SmolLM3-Stage1 + IFEval + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "SmolLM3-Stage1 + IFEval (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval full-ft seed=$seed algorithm.strategy.lr=$lr algorithm.batch_size=4 trainer.accumulate_grad_batches=2
#   done
# done

# # SmolLM3-Stage1 + IFEval + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "SmolLM3-Stage1 + IFEval (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # SmolLM3-Stage1 + IFEval + Baseline
# echo "SmolLM3-Stage1 + IFEval (baseline)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval baseline seed=1

# # SmolLM3-Stage1 + IFEval + URIAL
# echo "SmolLM3-Stage1 + IFEval (urial)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval urial seed=1

# # SmolLM3-Stage1 + IFEval + LM Head
# echo "SmolLM3-Stage1 + IFEval (lm-head)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval lm-head seed=1

# # SmolLM3-Stage1 + IFEval + BLoRA
# for seed in "${SEEDS[@]}"; do
#   for scales_lr in "${SCALES_LR[@]}"; do
#     echo "SmolLM3-Stage1 + IFEval (blora, seed=$seed, scales_lr=$scales_lr, r=1, prune_rank=False)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh smollm3-stage1 ifeval blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=1 algorithm.strategy.prune_rank=False
#
#     echo "SmolLM3-Stage1 + IFEval (blora, seed=$seed, scales_lr=$scales_lr, r=2, prune_rank=True)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh smollm3-stage1 ifeval blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=2 algorithm.strategy.prune_rank=True
#   done
# done

# # SmolLM3-Stage1 + IFEval + Phase One
# echo "SmolLM3-Stage1 + IFEval (phase-one)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval phase-one seed=1

# # SmolLM3-Stage1 + IFEval + Phase Two (depends on phase-one completing first)
# for lr in "${LR_PHASE_TWO[@]}"; do
#   for grads in "${GRADS_IN_MEMORY[@]}"; do
#     echo "SmolLM3-Stage1 + IFEval (phase-two, lr=$lr, grads_in_memory=$grads)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm3-stage1 ifeval phase-two seed=1 algorithm.strategy.lr=$lr algorithm.strategy.grads_in_memory=$grads
#   done
# done

################################################################################
# OLMo3-7B-Step1414k
################################################################################

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

# # OLMo3-7B-Step1414k + MetaMath + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "OLMo3-7B-Step1414k + MetaMath (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath full-ft seed=$seed algorithm.strategy.lr=$lr algorithm.batch_size=4 trainer.accumulate_grad_batches=2
#   done
# done

# # OLMo3-7B-Step1414k + MetaMath + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "OLMo3-7B-Step1414k + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # OLMo3-7B-Step1414k + MetaMath + Baseline
# echo "OLMo3-7B-Step1414k + MetaMath (baseline)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath baseline seed=1

# # OLMo3-7B-Step1414k + MetaMath + URIAL
# echo "OLMo3-7B-Step1414k + MetaMath (urial)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath urial seed=1

# # OLMo3-7B-Step1414k + MetaMath + LM Head
# echo "OLMo3-7B-Step1414k + MetaMath (lm-head)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath lm-head seed=1

# # OLMo3-7B-Step1414k + MetaMath + BLoRA
# for seed in "${SEEDS[@]}"; do
#   for scales_lr in "${SCALES_LR[@]}"; do
#     echo "OLMo3-7B-Step1414k + MetaMath (blora, seed=$seed, scales_lr=$scales_lr, r=1, prune_rank=False)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-7b-step1414k metamath blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=1 algorithm.strategy.prune_rank=False

#     echo "OLMo3-7B-Step1414k + MetaMath (blora, seed=$seed, scales_lr=$scales_lr, r=2, prune_rank=True)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-7b-step1414k metamath blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=2 algorithm.strategy.prune_rank=True
#   done
# done

# # OLMo3-7B-Step1414k + MetaMath + Phase One
# echo "OLMo3-7B-Step1414k + MetaMath (phase-one)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath phase-one seed=1

# # OLMo3-7B-Step1414k + MetaMath + Phase Two
# for lr in "${LR_PHASE_TWO[@]}"; do
#   for grads in "${GRADS_IN_MEMORY[@]}"; do
#     echo "OLMo3-7B-Step1414k + MetaMath (phase-two, lr=$lr, grads_in_memory=$grads)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k metamath phase-two seed=1 algorithm.strategy.lr=$lr algorithm.strategy.grads_in_memory=$grads
#   done
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

# # OLMo3-7B-Step1414k + FLORES + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "OLMo3-7B-Step1414k + FLORES (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k flores full-ft seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # OLMo3-7B-Step1414k + FLORES + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "OLMo3-7B-Step1414k + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k flores lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # OLMo3-7B-Step1414k + FLORES + Baseline
# echo "OLMo3-7B-Step1414k + FLORES (baseline)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k flores baseline seed=1

# # OLMo3-7B-Step1414k + FLORES + URIAL
# echo "OLMo3-7B-Step1414k + FLORES (urial)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k flores urial seed=1

# # OLMo3-7B-Step1414k + FLORES + LM Head
# echo "OLMo3-7B-Step1414k + FLORES (lm-head)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k flores lm-head seed=1

# # OLMo3-7B-Step1414k + FLORES + BLoRA
# for seed in "${SEEDS[@]}"; do
#   for scales_lr in "${SCALES_LR[@]}"; do
#     echo "OLMo3-7B-Step1414k + FLORES (blora, seed=$seed, scales_lr=$scales_lr, r=1, prune_rank=False)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-7b-step1414k flores blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=1 algorithm.strategy.prune_rank=False

#     echo "OLMo3-7B-Step1414k + FLORES (blora, seed=$seed, scales_lr=$scales_lr, r=2, prune_rank=True)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-7b-step1414k flores blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=2 algorithm.strategy.prune_rank=True
#   done
# done

# # OLMo3-7B-Step1414k + FLORES + Phase One
# echo "OLMo3-7B-Step1414k + FLORES (phase-one)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k flores phase-one seed=1

# # OLMo3-7B-Step1414k + FLORES + Phase Two
# for lr in "${LR_PHASE_TWO[@]}"; do
#   for grads in "${GRADS_IN_MEMORY[@]}"; do
#     echo "OLMo3-7B-Step1414k + FLORES (phase-two, lr=$lr, grads_in_memory=$grads)"
#     sbatch --gres=gpu:1 slurm/run-experiment.sh olmo3-7b-step1414k flores phase-two seed=1 algorithm.strategy.lr=$lr algorithm.strategy.grads_in_memory=$grads
#   done
# done

# # OLMo3-7B-Step1414k + IFEval + Online Coding
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-7B-Step1414k + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex
#     done
#   done
# done

# # OLMo3-7B-Step1414k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-7B-Step1414k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-7B-Step1414k + IFEval + Full FT
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_FULL_FT[@]}"; do
#     echo "OLMo3-7B-Step1414k + IFEval (full-ft, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval full-ft seed=$seed algorithm.strategy.lr=$lr algorithm.batch_size=4 trainer.accumulate_grad_batches=2
#   done
# done

# # OLMo3-7B-Step1414k + IFEval + LoRA
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "OLMo3-7B-Step1414k + IFEval (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval lora seed=$seed algorithm.strategy.lr=$lr
#   done
# done

# # OLMo3-7B-Step1414k + IFEval + Baseline
# echo "OLMo3-7B-Step1414k + IFEval (baseline)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval baseline seed=1

# # OLMo3-7B-Step1414k + IFEval + URIAL
# echo "OLMo3-7B-Step1414k + IFEval (urial)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval urial seed=1

# # OLMo3-7B-Step1414k + IFEval + LM Head
# echo "OLMo3-7B-Step1414k + IFEval (lm-head)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval lm-head seed=1

# # OLMo3-7B-Step1414k + IFEval + BLoRA
# for seed in "${SEEDS[@]}"; do
#   for scales_lr in "${SCALES_LR[@]}"; do
#     echo "OLMo3-7B-Step1414k + IFEval (blora, seed=$seed, scales_lr=$scales_lr, r=1, prune_rank=False)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-7b-step1414k ifeval blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=1 algorithm.strategy.prune_rank=False
#
#     echo "OLMo3-7B-Step1414k + IFEval (blora, seed=$seed, scales_lr=$scales_lr, r=2, prune_rank=True)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-7b-step1414k ifeval blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=2 algorithm.strategy.prune_rank=True
#   done
# done

# # OLMo3-7B-Step1414k + IFEval + Phase One
# echo "OLMo3-7B-Step1414k + IFEval (phase-one)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval phase-one seed=1

# # OLMo3-7B-Step1414k + IFEval + Phase Two (depends on phase-one completing first)
# for lr in "${LR_PHASE_TWO[@]}"; do
#   for grads in "${GRADS_IN_MEMORY[@]}"; do
#     echo "OLMo3-7B-Step1414k + IFEval (phase-two, lr=$lr, grads_in_memory=$grads)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-7b-step1414k ifeval phase-two seed=1 algorithm.strategy.lr=$lr algorithm.strategy.grads_in_memory=$grads
#   done
# done

################################################################################
# OLMo3-32B-Step656k
################################################################################

# OLMo3-32B-Step656k + MetaMath + LoRA (QLoRA with DDP)
echo "OLMo3-32B-Step656k + MetaMath (lora, seed=1, lr=0.0001)"
sbatch --gres=gpu:h100:4 --time=24:00:00 --export=ALL,NCCL_P2P_DISABLE=1 slurm/run-experiment.sh olmo3-32b-step656k metamath lora seed=1 algorithm.strategy.lr=0.0001 algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.strategy=ddp trainer.devices=4

echo "OLMo3-32B-Step656k + MetaMath (lora, seed=1, lr=1e-05)"
sbatch --gres=gpu:h100:4 --time=24:00:00 --export=ALL,NCCL_P2P_DISABLE=1 slurm/run-experiment.sh olmo3-32b-step656k metamath lora seed=1 algorithm.strategy.lr=1e-05 algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.strategy=ddp trainer.devices=4

# # OLMo3-32B-Step656k + MetaMath + Online Coding (QLoRA)
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-32B-Step656k + MetaMath (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 --time=24:00:00 slurm/run-experiment.sh olmo3-32b-step656k metamath online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#     done
#   done
# done

# # OLMo3-32B-Step656k + MetaMath + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-32B-Step656k + MetaMath (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k metamath icl algorithm.batch_size=1 seed=$seed algorithm.max_length=512
# done

# # OLMo3-32B-Step656k + MetaMath + LoRA (QLoRA)
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "OLMo3-32B-Step656k + MetaMath (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k metamath lora seed=$seed algorithm.strategy.lr=$lr algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#   done
# done

# # OLMo3-32B-Step656k + MetaMath + Baseline
# echo "OLMo3-32B-Step656k + MetaMath (baseline)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k metamath baseline seed=1

# # OLMo3-32B-Step656k + MetaMath + URIAL
# echo "OLMo3-32B-Step656k + MetaMath (urial)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k metamath urial seed=1

# # OLMo3-32B-Step656k + MetaMath + LM Head
# echo "OLMo3-32B-Step656k + MetaMath (lm-head)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k metamath lm-head seed=1

# # OLMo3-32B-Step656k + MetaMath + BLoRA (QLoRA)
# for seed in "${SEEDS[@]}"; do
#   for scales_lr in "${SCALES_LR[@]}"; do
#     echo "OLMo3-32B-Step656k + MetaMath (blora, seed=$seed, scales_lr=$scales_lr, r=1, prune_rank=False)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-32b-step656k metamath blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=1 algorithm.strategy.prune_rank=False algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4

#     echo "OLMo3-32B-Step656k + MetaMath (blora, seed=$seed, scales_lr=$scales_lr, r=2, prune_rank=True)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-32b-step656k metamath blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=2 algorithm.strategy.prune_rank=True algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#   done
# done

# # OLMo3-32B-Step656k + FLORES + Online Coding (QLoRA)
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-32B-Step656k + FLORES (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 --time=24:00:00 slurm/run-experiment.sh olmo3-32b-step656k flores online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#     done
#   done
# done

# # OLMo3-32B-Step656k + FLORES + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-32B-Step656k + FLORES (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k flores icl algorithm.batch_size=1 seed=$seed algorithm.max_length=512
# done

# # OLMo3-32B-Step656k + FLORES + LoRA (QLoRA)
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "OLMo3-32B-Step656k + FLORES (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k flores lora seed=$seed algorithm.strategy.lr=$lr algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#   done
# done

# # OLMo3-32B-Step656k + FLORES + Baseline
# echo "OLMo3-32B-Step656k + FLORES (baseline)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k flores baseline seed=1

# # OLMo3-32B-Step656k + FLORES + URIAL
# echo "OLMo3-32B-Step656k + FLORES (urial)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k flores urial seed=1

# # OLMo3-32B-Step656k + FLORES + LM Head
# echo "OLMo3-32B-Step656k + FLORES (lm-head)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k flores lm-head seed=1

# # OLMo3-32B-Step656k + FLORES + BLoRA (QLoRA)
# for seed in "${SEEDS[@]}"; do
#   for scales_lr in "${SCALES_LR[@]}"; do
#     echo "OLMo3-32B-Step656k + FLORES (blora, seed=$seed, scales_lr=$scales_lr, r=1, prune_rank=False)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-32b-step656k flores blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=1 algorithm.strategy.prune_rank=False algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4

#     echo "OLMo3-32B-Step656k + FLORES (blora, seed=$seed, scales_lr=$scales_lr, r=2, prune_rank=True)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-32b-step656k flores blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=2 algorithm.strategy.prune_rank=True algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#   done
# done

# # OLMo3-32B-Step656k + IFEval + Online Coding (QLoRA)
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     for max_ex in "${MAX_EXAMPLES[@]}"; do
#       echo "OLMo3-32B-Step656k + IFEval (online-coding, seed=$seed, lr=$lr, max_examples=$max_ex)"
#       sbatch --gres=gpu:a100l:1 --time=24:00:00 slurm/run-experiment.sh olmo3-32b-step656k ifeval online-coding seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#     done
#   done
# done

# # OLMo3-32B-Step656k + IFEval + ICL
# for seed in "${SEEDS[@]}"; do
#   echo "OLMo3-32B-Step656k + IFEval (icl, seed=$seed)"
#   sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k ifeval icl algorithm.batch_size=1 seed=$seed
# done

# # OLMo3-32B-Step656k + IFEval + LoRA (QLoRA)
# for seed in "${SEEDS[@]}"; do
#   for lr in "${LR_ONLINE_CODING[@]}"; do
#     echo "OLMo3-32B-Step656k + IFEval (lora, seed=$seed, lr=$lr)"
#     sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k ifeval lora seed=$seed algorithm.strategy.lr=$lr algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#   done
# done

# # OLMo3-32B-Step656k + IFEval + Baseline
# echo "OLMo3-32B-Step656k + IFEval (baseline)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k ifeval baseline seed=1

# # OLMo3-32B-Step656k + IFEval + URIAL
# echo "OLMo3-32B-Step656k + IFEval (urial)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k ifeval urial seed=1

# # OLMo3-32B-Step656k + IFEval + LM Head
# echo "OLMo3-32B-Step656k + IFEval (lm-head)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo3-32b-step656k ifeval lm-head seed=1

# # OLMo3-32B-Step656k + IFEval + BLoRA (QLoRA)
# for seed in "${SEEDS[@]}"; do
#   for scales_lr in "${SCALES_LR[@]}"; do
#     echo "OLMo3-32B-Step656k + IFEval (blora, seed=$seed, scales_lr=$scales_lr, r=1, prune_rank=False)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-32b-step656k ifeval blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=1 algorithm.strategy.prune_rank=False algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#
#     echo "OLMo3-32B-Step656k + IFEval (blora, seed=$seed, scales_lr=$scales_lr, r=2, prune_rank=True)"
#     sbatch --gres=gpu:a100l:1 --time=48:00:00 slurm/run-experiment.sh olmo3-32b-step656k ifeval blora seed=$seed algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=2 algorithm.strategy.prune_rank=True algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4
#   done
# done

echo ""
echo "All jobs submitted!"
