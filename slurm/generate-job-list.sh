#!/bin/bash

# Generates job list for parallel execution
# Output: jobs.txt with one command per line

SEEDS=(1)
SCALES_LR=(1e-2 2e-3)

OUTPUT_FILE="jobs.txt"
> $OUTPUT_FILE

# Missing BLoRA jobs for smollm3-stage1 + metamath
echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=smollm3-stage1 algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=0.01 algorithm.strategy.r=2 algorithm.strategy.prune_rank=True" >> $OUTPUT_FILE
echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=smollm3-stage1 algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=0.002 algorithm.strategy.r=1 algorithm.strategy.prune_rank=False" >> $OUTPUT_FILE
echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=smollm3-stage1 algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=0.002 algorithm.strategy.r=2 algorithm.strategy.prune_rank=True" >> $OUTPUT_FILE

# Missing BLoRA jobs for smollm3-stage1 + ifeval (all 4)
for scales_lr in 0.01 0.002; do
  for r in 1 2; do
    if [ $r -eq 1 ]; then
      prune_rank="False"
    else
      prune_rank="True"
    fi
    echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=smollm3-stage1 algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=$r algorithm.strategy.prune_rank=$prune_rank" >> $OUTPUT_FILE
  done
done

# Missing BLoRA jobs for smollm3-stage1 + flores (all 4)
for scales_lr in 0.01 0.002; do
  for r in 1 2; do
    if [ $r -eq 1 ]; then
      prune_rank="False"
    else
      prune_rank="True"
    fi
    echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=smollm3-stage1 algorithm/dataset=flores algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=$r algorithm.strategy.prune_rank=$prune_rank" >> $OUTPUT_FILE
  done
done

# Missing BLoRA jobs for olmo3-7b-step1414k + metamath (2 missing)
echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=olmo3-7b-step1414k algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=0.01 algorithm.strategy.r=1 algorithm.strategy.prune_rank=False" >> $OUTPUT_FILE
echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=olmo3-7b-step1414k algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=0.002 algorithm.strategy.r=1 algorithm.strategy.prune_rank=False" >> $OUTPUT_FILE

# Missing BLoRA jobs for olmo3-7b-step1414k + ifeval (all 4)
for scales_lr in 0.01 0.002; do
  for r in 1 2; do
    if [ $r -eq 1 ]; then
      prune_rank="False"
    else
      prune_rank="True"
    fi
    echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=olmo3-7b-step1414k algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=$r algorithm.strategy.prune_rank=$prune_rank" >> $OUTPUT_FILE
  done
done

# Missing BLoRA jobs for olmo3-7b-step1414k + flores (all 4)
for scales_lr in 0.01 0.002; do
  for r in 1 2; do
    if [ $r -eq 1 ]; then
      prune_rank="False"
    else
      prune_rank="True"
    fi
    echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=olmo3-7b-step1414k algorithm/dataset=flores algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=$r algorithm.strategy.prune_rank=$prune_rank" >> $OUTPUT_FILE
  done
done

# Missing BLoRA jobs for olmo3-32b-step656k + metamath (all 4)
for scales_lr in 0.01 0.002; do
  for r in 1 2; do
    if [ $r -eq 1 ]; then
      prune_rank="False"
    else
      prune_rank="True"
    fi
    echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=olmo3-32b-step656k algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=$r algorithm.strategy.prune_rank=$prune_rank algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4" >> $OUTPUT_FILE
  done
done

# Missing BLoRA jobs for olmo3-32b-step656k + ifeval (all 4)
for scales_lr in 0.01 0.002; do
  for r in 1 2; do
    if [ $r -eq 1 ]; then
      prune_rank="False"
    else
      prune_rank="True"
    fi
    echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=olmo3-32b-step656k algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=$r algorithm.strategy.prune_rank=$prune_rank algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4" >> $OUTPUT_FILE
  done
done

# Missing BLoRA jobs for olmo3-32b-step656k + flores (all 4)
for scales_lr in 0.01 0.002; do
  for r in 1 2; do
    if [ $r -eq 1 ]; then
      prune_rank="False"
    else
      prune_rank="True"
    fi
    echo "python sah/main.py experiment=finetune-with-strategy-blora algorithm/model=olmo3-32b-step656k algorithm/dataset=flores algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.scales_lr=$scales_lr algorithm.strategy.r=$r algorithm.strategy.prune_rank=$prune_rank algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4" >> $OUTPUT_FILE
  done
done

# Missing LoRA jobs for olmo3-32b-step656k + metamath
echo "python sah/main.py experiment=finetune-with-strategy-lora algorithm/model=olmo3-32b-step656k algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.lr=0.0001 algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4" >> $OUTPUT_FILE
echo "python sah/main.py experiment=finetune-with-strategy-lora algorithm/model=olmo3-32b-step656k algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' seed=1 algorithm.strategy.lr=1e-05 algorithm.strategy.ft_strategy=qlora algorithm.batch_size=2 trainer.accumulate_grad_batches=4" >> $OUTPUT_FILE

echo "Generated $(wc -l < $OUTPUT_FILE) jobs in $OUTPUT_FILE"
