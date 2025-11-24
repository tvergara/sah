#!/bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

python scripts/generate_it_data.py
