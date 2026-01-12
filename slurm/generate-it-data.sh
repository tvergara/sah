#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=80G
#SBATCH --time=5:00:00
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%A_%a.out

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

NUM_SPLITS=${1:-50}

if [ "$NUM_SPLITS" -gt 1 ]; then
    python scripts/generate_it_data.py --num_splits $NUM_SPLITS --split_index $SLURM_ARRAY_TASK_ID
else
    python scripts/generate_it_data.py
fi
