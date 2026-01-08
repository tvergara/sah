#!/bin/bash

# Generates job list for parallel execution
# Output: jobs.txt with one command per line

SEEDS=(1)
LR_ONLINE_CODING=(1e-4 1e-5)
MAX_EXAMPLES=(1024 2048 4096 8192 16384 32768)

OUTPUT_FILE="jobs.txt"
> $OUTPUT_FILE

# From submit-checkpoint-experiments.sh - IFEval + Online Coding only

models=(
  "smollm3"
  "smollm3-step40k"
  "smollm3-step1720k"
  "smollm3-stage2"
  "smollm3-stage3"
  "olmo3-7b-step0"
  "olmo3-7b-step707k"
  "olmo3-7b-stage2-step6k"
  "olmo3-7b-stage2-step12k"
  "olmo3-7b-stage2-step24k"
  "olmo3-7b-stage2-step48k"
  "olmo3-7b-instruct-step200"
  "olmo3-7b-instruct-step400"
  "olmo3-7b-instruct-final"
  "olmo3-1025-7b"
)

for model in "${models[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for lr in "${LR_ONLINE_CODING[@]}"; do
      for max_ex in "${MAX_EXAMPLES[@]}"; do
        echo "python sah/main.py experiment=finetune-with-strategy-online-coding algorithm/model=$model algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex" >> $OUTPUT_FILE
      done
    done
  done
done

# From submit-pretraining-experiments.sh - IFEval + Online Coding only

pretraining_models=(
  "smollm3-stage1"
  "olmo3-7b-step1414k"
)

for model in "${pretraining_models[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for lr in "${LR_ONLINE_CODING[@]}"; do
      for max_ex in "${MAX_EXAMPLES[@]}"; do
        echo "python sah/main.py experiment=finetune-with-strategy-online-coding algorithm/model=$model algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=$seed algorithm.strategy.lr=$lr algorithm.max_examples=$max_ex" >> $OUTPUT_FILE
      done
    done
  done
done

# OLMo3-32B with QLoRA
for seed in "${SEEDS[@]}"; do
  for max_ex in "${MAX_EXAMPLES[@]}"; do
    echo "python sah/main.py experiment=finetune-with-strategy-online-coding algorithm/model=olmo3-32b-step656k algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=$seed algorithm.strategy.lr=1e-5 algorithm.max_examples=$max_ex algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4" >> $OUTPUT_FILE
  done
done

echo "Generated $(wc -l < $OUTPUT_FILE) jobs in $OUTPUT_FILE"
