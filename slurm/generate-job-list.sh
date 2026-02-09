#!/bin/bash

# Generates job list for parallel execution
# Output: jobs.txt with one command per line

OUTPUT_FILE="jobs.txt"
> $OUTPUT_FILE

# smollm3-step0 + metamath: baseline, icl, urial, online_coding (both lrs)
for max_ex in 1024 2048 4096 8192 16384 32768; do
  echo "python sah/main.py experiment=finetune-with-strategy-online-coding algorithm/model=smollm3-step0 algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1 algorithm.strategy.lr=0.0001 algorithm.max_examples=$max_ex" >> $OUTPUT_FILE
done
for max_ex in 1024 2048 4096 8192 16384 32768; do
  echo "python sah/main.py experiment=finetune-with-strategy-online-coding algorithm/model=smollm3-step0 algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1 algorithm.strategy.lr=1e-05 algorithm.max_examples=$max_ex" >> $OUTPUT_FILE
done
echo "python sah/main.py experiment=finetune-with-strategy-icl algorithm/model=smollm3-step0 algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1 algorithm.batch_size=1" >> $OUTPUT_FILE
echo "python sah/main.py experiment=finetune-with-strategy-urial algorithm/model=smollm3-step0 algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1" >> $OUTPUT_FILE
echo "python sah/main.py experiment=finetune-with-strategy-baseline algorithm/model=smollm3-step0 algorithm/dataset=metamath algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1" >> $OUTPUT_FILE

# smollm3-step0 + flores (nllb): icl, online_coding (both lrs)
for max_ex in 1024 2048 4096 8192 16384 32768; do
  echo "python sah/main.py experiment=finetune-with-strategy-online-coding algorithm/model=smollm3-step0 algorithm/dataset=flores algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1 algorithm.strategy.lr=0.0001 algorithm.max_examples=$max_ex" >> $OUTPUT_FILE
done
for max_ex in 1024 2048 4096 8192 16384 32768; do
  echo "python sah/main.py experiment=finetune-with-strategy-online-coding algorithm/model=smollm3-step0 algorithm/dataset=flores algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1 algorithm.strategy.lr=1e-05 algorithm.max_examples=$max_ex" >> $OUTPUT_FILE
done
echo "python sah/main.py experiment=finetune-with-strategy-icl algorithm/model=smollm3-step0 algorithm/dataset=flores algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1 algorithm.batch_size=1" >> $OUTPUT_FILE

# smollm3-step0 + ifeval: baseline, icl, urial, online_coding (only lr=1e-05)
for max_ex in 1024 2048 4096 8192 16384 32768; do
  echo "python sah/main.py experiment=finetune-with-strategy-online-coding algorithm/model=smollm3-step0 algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1 algorithm.strategy.lr=1e-05 algorithm.max_examples=$max_ex" >> $OUTPUT_FILE
done
echo "python sah/main.py experiment=finetune-with-strategy-icl algorithm/model=smollm3-step0 algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1 algorithm.batch_size=1" >> $OUTPUT_FILE
echo "python sah/main.py experiment=finetune-with-strategy-urial algorithm/model=smollm3-step0 algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1" >> $OUTPUT_FILE
echo "python sah/main.py experiment=finetune-with-strategy-baseline algorithm/model=smollm3-step0 algorithm/dataset=ifeval algorithm.result_file='\${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null seed=1" >> $OUTPUT_FILE

echo "Generated $(wc -l < $OUTPUT_FILE) jobs in $OUTPUT_FILE"
