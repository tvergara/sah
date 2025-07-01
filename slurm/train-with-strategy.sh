#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=1                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=20G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out  # Write the log on scratch

source ~/.bashrc

cd ~/sah

. .venv/bin/activate


tau=(0.1 0.5 1.0 10.0)
alpha=(0.0 0.5 1.0 1.5)

for t in "${tau[@]}"; do
  for a in "${alpha[@]}"; do
    python sah/main.py experiment=train-with-strategy \
      algorithm.general_config.tau=$t \
      algorithm.general_config.alpha=$a \
      algorithm.general_config.strategy=proposed
  done
done
