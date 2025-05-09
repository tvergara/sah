#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=1                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=30G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out  # Write the log on scratch

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

steps=()
for i in {1..8}; do
  steps+=("step-$(( i * 160000 ))")
done
for i in {1..4}; do
  RAW=$(( i * 50 ))
  REV=$(printf "%05d" "${RAW}")
  steps+=(REV)
done

for in_dir in "${steps[@]}"; do
  for out_dir in "${steps[@]}"; do
    echo "=== running with input_dir=${in_dir} out_dir=${out_dir} ==="
    python sah/main.py \
      experiment=alpaca-entropy \
      algorithm.activations_config.input_dir="${in_dir}" \
      algorithm.activations_config.output_dir="${out_dir}"
    done
done
