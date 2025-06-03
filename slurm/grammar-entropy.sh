#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=1                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=18:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out  # Write the log on scratch

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

steps=()
# for i in {10..26}; do
#   RAW=$(( i * 100 ))
#   REV=$(printf "%05d" "${RAW}")
#   steps+=("pretraining_step_${REV}")
# done
for i in {1..26}; do
  RAW=$(( i * 100 ))
  REV=$(printf "%05d" "${RAW}")
  steps+=("finetuning_step_${REV}")
done

for i in "${!steps[@]}"; do
  first_input="${steps[i]}"
  # for (( j=i; j<${#steps[@]}; j++ )); do
    second_input="${steps[i + 1]}"
    # if [ "$first_input" = "$second_input" ]; then
      echo "=== running with first_input=${first_input} second_input=null ==="
      python sah/main.py experiment=grammar-entropy \
        algorithm.activations_config.first_input="${first_input}" \
        algorithm.activations_config.second_input=null \
        algorithm.activations_config.base_path=/network/scratch/b/brownet/synthetic-data/activations/subgrammar-43 \
        algorithm.general_config.result_file=/network/scratch/b/brownet/hydra-runs/grammar-entropy/grammar-43-entropies.csv
    # else
      echo "=== running with input_dir=${first_input} second_input=${second_input} ==="
      python sah/main.py \
        experiment=grammar-entropy \
        algorithm.activations_config.first_input="${first_input}" \
        algorithm.activations_config.second_input="${second_input}" \
        algorithm.activations_config.base_path=/network/scratch/b/brownet/synthetic-data/activations/subgrammar-43 \
        algorithm.general_config.result_file=/network/scratch/b/brownet/hydra-runs/grammar-entropy/grammar-43-entropies.csv
    # fi
  # done
done
