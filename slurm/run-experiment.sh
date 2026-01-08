#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o ~/scratch/slurm-logs/slurm-%j.out

# Usage: sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath adam
# Usage: sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh qwen metamath adam algorithm.batch_size=4

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

export WANDB_MODE=offline

MODEL=${1:-smollm3-stage1}
DATASET=${2:-metamath}
STRATEGY=${3:-icl}
shift 3
EXTRA_ARGS="$@"

python sah/main.py experiment=finetune-with-strategy-$STRATEGY algorithm/model=$MODEL algorithm/dataset=$DATASET algorithm.result_file='${hydra:runtime.output_dir}/final-results.jsonl' trainer.limit_val_batches=null $EXTRA_ARGS
