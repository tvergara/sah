#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=6:00:00
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out

# Usage: sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath adam
# Usage: sbatch --gres=gpu:a100l:2 slurm/run-experiment.sh qwen metamath adam algorithm.batch_size=4

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

MODEL=${1:-qwen}
DATASET=${2:-metamath}
STRATEGY=${3:-adam}
shift 3
EXTRA_ARGS="$@"

python sah/main.py experiment=finetune-with-strategy-$STRATEGY algorithm/model=$MODEL algorithm/dataset=$DATASET $EXTRA_ARGS
