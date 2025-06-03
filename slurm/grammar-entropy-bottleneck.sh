#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=1                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=6:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out  # Write the log on scratch

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

echo "=== running unconditional estimate ==="
python sah/main.py \
  experiment=entropy-with-bottleneck \
  algorithm.general_config.unconditional_estimate=true

for i in {1..26}; do
  RAW=$(( i * 100 ))
  REV=$(printf "%05d" "${RAW}")
  echo "=== running revision step-${REV} ==="
  python sah/main.py \
    experiment=entropy-with-bottleneck \
    algorithm.checkpoint_config.path=/network/scratch/b/brownet/hydra-runs/grammar/checkpoints/pretraining_step_${REV}
done

for i in {1..13}; do
  RAW=$(( i * 10 ))
  REV=$(printf "%05d" "${RAW}")
  echo "=== running revision step-${REV} ==="
  python sah/main.py \
    experiment=entropy-with-bottleneck \
    algorithm.checkpoint_config.path=/network/scratch/b/brownet/hydra-runs/finetune-grammar/checkpoints/finetuning_step_${REV}
done
