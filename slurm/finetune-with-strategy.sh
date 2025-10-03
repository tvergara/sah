#!/bin/bash

#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 2 CPUs
#SBATCH --gres=gpu:a100l:1                                     # Ask for 1 GPU
#SBATCH --mem=80G                                        # Ask for 10 GB of RAM
#SBATCH --time=72:00:00                                   # The job will run for 5 hours
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out  # Write the log on scratch

# Usage: sbatch slurm/finetune-with-strategy.sh [STRATEGY]
# Example: sbatch slurm/finetune-with-strategy.sh compressed
# Example: sbatch slurm/finetune-with-strategy.sh lora

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

# Get strategy from command line argument, default to compressed
STRATEGY=${1:-compressed}

python sah/main.py experiment=finetune-with-strategy-$STRATEGY
