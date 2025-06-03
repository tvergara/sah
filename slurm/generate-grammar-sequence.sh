#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=1                                # Ask for 2 CPUs
#SBATCH --gres=gpu:0                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=2:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out  # Write the log on scratch

source ~/.bashrc

cd ~/sah

. .venv/bin/activate

python -m sah.scripts.generate_grammar_data --config-path ../../sah/configs/scripts --config-name generate_grammar_data.yaml dataset.save_dir=/network/scratch/b/brownet/synthetic-data/grammar-0 dataset.n=5000 grammar.n_nonterms=40 grammar.n_terms=20 grammar.n_prods_per_nonterm=3 grammar.avg_branch=4 dataset.max_depth=12


for i in {1..10}; do
  python -m sah.scripts.generate_subgrammar_data --config-path ../../sah/configs/scripts --config-name generate_subgrammar_data.yaml dataset.out_dir=/network/scratch/b/brownet/synthetic-data/grammar-${i} dataset.n=5000 dataset.max_depth=12 initial_grammar.path=/network/scratch/b/brownet/synthetic-data/grammar-$((i - 1)) trim.keep_prob=0.8
done
