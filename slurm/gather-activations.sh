#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=1                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out  # Write the log on scratch

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

for i in {1..8}; do
  REV=$(( i * 160000 ))
  echo "=== running revision step-${REV} ==="
  python sah/main.py \
    experiment=alpaca-inference \
    algorithm.network_config.revision=step-${REV}
done

for i in {1..31}; do
  RAW=$(( i * 50 ))
  REV=$(printf "%05d" "${RAW}")
  echo "=== running revision step-${REV} ==="
  python sah/main.py \
    experiment=alpaca-inference \
    algorithm.general_config.local_checkpoint=true \
    algorithm.general_config.checkpoint_path=$SCRATCH/hydra-runs/alpaca/checkpoints \
    algorithm.general_config.revision=step_${REV}
done
