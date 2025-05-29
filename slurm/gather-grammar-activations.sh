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

for i in {1..26}; do
  RAW=$(( i * 100 ))
  REV=$(printf "%05d" "${RAW}")
  echo "=== running revision step-${REV} ==="
  python sah/main.py \
    experiment=activations-grammar \
    algorithm.general_config.revision=pretraining_step_${REV} \
    algorithm.general_config.checkpoint_dir=/network/scratch/b/brownet/hydra-runs/grammar/checkpoints/
done

for i in {1..10}; do
  RAW=$(( i * 10 ))
  REV=$(printf "%05d" "${RAW}")
  echo "=== running revision step-${REV} ==="
  python sah/main.py \
    experiment=activations-grammar \
    algorithm.general_config.revision=finetuning_step_${REV} \
    algorithm.general_config.checkpoint_dir=/network/scratch/b/brownet/hydra-runs/finetune-grammar/checkpoints/
done
