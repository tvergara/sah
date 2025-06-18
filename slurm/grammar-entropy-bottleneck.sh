#!/bin/bash

#SBATCH --partition=main                           # Ask for unkillable job
#SBATCH --cpus-per-task=1                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --time=10:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/b/brownet/slurm-logs/slurm-%j.out  # Write the log on scratch

source ~/.bashrc

cd ~/sah

. .venv/bin/activate


for i in {0..10}; do
  SIZE=$(( 2 ** i ))

  echo "=== running unconditional estimate, size ${SIZE} ==="
  python sah/main.py \
    experiment=entropy-with-bottleneck \
    algorithm.general_config.unconditional_estimate=true \
    algorithm.grammar_config.base_path=/network/scratch/b/brownet/synthetic-data/subgrammar-extended \
    algorithm.grammar_config.size=${SIZE} \
    algorithm.general_config.result_file=/network/scratch/b/brownet/hydra-runs/entropy-bottleneck/grammar-entropies.csv
    # trainer.limit_train_batches=0.0

  for j in {1..27}; do
    RAW=$(( j * 100 ))
    REV=$(printf "%05d" "${RAW}")

    if (( j < 18 && i < 7 )); then
      continue
    fi

    echo "=== running revision step-${REV}, size ${SIZE} ==="
    python sah/main.py \
      experiment=entropy-with-bottleneck \
      algorithm.grammar_config.base_path=/network/scratch/b/brownet/synthetic-data/subgrammar-extended \
      algorithm.checkpoint_config.path=/network/scratch/b/brownet/hydra-runs/grammar/checkpoints/pretraining_step_${REV} \
    algorithm.grammar_config.size=${SIZE} \
      algorithm.general_config.result_file=/network/scratch/b/brownet/hydra-runs/entropy-bottleneck/grammar-entropies.csv
    # trainer.limit_train_batches=0.0
  done
done

# for i in {1..13}; do
#   RAW=$(( i * 10 ))
#   REV=$(printf "%05d" "${RAW}")
#   echo "=== running revision step-${REV} ==="
#   python sah/main.py \
#     experiment=entropy-with-bottleneck \
#     algorithm.checkpoint_config.path=/network/scratch/b/brownet/hydra-runs/finetune-grammar/checkpoints/finetuning_step_${REV} \
#     algorithm.general_config.result_file=/network/scratch/b/brownet/hydra-runs/entropy-bottleneck/grammar-entropies-gradual.csv
#   # trainer.limit_train_batches=0.0
# done
