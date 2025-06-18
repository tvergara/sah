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


states_replaced=(0 1 2 5 10)
gamma=(0.5 0.8 1.0 1.2 1.4)
noise=(0.1 0.3)

for s in "${states_replaced[@]}"; do
  for g in "${gamma[@]}"; do
    for n in "${noise[@]}"; do
      id=$(echo "${s}_${g}_${n}" | sha1sum | cut -c1-8)
      echo "Running with states_replaced=$s, gamma=$g, noise=$n, id=$id"
      python -m sah.scripts.generate_automata_variation_data --config-path ../../sah/configs/scripts --config-name generate_automata_variation_data.yaml \
        mod.remove_states="$s" \
        mod.add_states="$s" \
        mod.gamma_token="$g" \
        mod.gamma_trans="$g" \
        mod.new_state_noise="$n" \
        dataset.out_dir="/network/scratch/b/brownet/synthetic-data/automata-variations/$id"
    done
  done
done
