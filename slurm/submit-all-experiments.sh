#!/bin/bash

echo "Submitting all experiment combinations"
echo ""

# echo "Job 1: Qwen + MetaMath (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath adam

# echo "Job 2: Qwen + MetaMath (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath icl algorithm.batch_size=1

# echo "Job 3: Qwen + MMLU (1 GPU, batch_size=1, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen mmlu adam

# echo "Job 4: Qwen + MMLU (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen mmlu icl algorithm.batch_size=1

# echo "Job 5: SmolLM + MetaMath (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath adam

# echo "Job 6: SmolLM + MetaMath (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath icl algorithm.batch_size=1

# echo "Job 7: SmolLM + MMLU (1 GPU, batch_size=1, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm mmlu adam

# echo "Job 8: SmolLM + MMLU (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm mmlu icl algorithm.batch_size=1

# echo "Job 9: SmolLM360 + MetaMath (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 metamath adam

# echo "Job 10: SmolLM360 + MetaMath (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 metamath icl algorithm.batch_size=1

# echo "Job 11: SmolLM360 + MMLU (1 GPU, batch_size=1, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 mmlu adam

# echo "Job 12: SmolLM360 + MMLU (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 mmlu icl algorithm.batch_size=1

# echo "Job 13: Qwen + MetaMath (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen metamath lora

# echo "Job 14: Qwen + MMLU (1 GPU, batch_size=1, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen mmlu lora

# echo "Job 15: SmolLM + MetaMath (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm metamath lora

# echo "Job 16: SmolLM + MMLU (1 GPU, batch_size=1, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm mmlu lora

# echo "Job 17: SmolLM360 + MetaMath (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 metamath lora

# echo "Job 18: SmolLM360 + MMLU (1 GPU, batch_size=1, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 mmlu lora

# echo "Job 19: Qwen + FLORES (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen flores adam

# echo "Job 20: Qwen + FLORES (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen flores icl algorithm.batch_size=1

# echo "Job 21: Qwen + FLORES (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh qwen flores lora

# echo "Job 22: SmolLM + FLORES (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm flores adam

# echo "Job 23: SmolLM + FLORES (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm flores icl algorithm.batch_size=1

# echo "Job 24: SmolLM + FLORES (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm flores lora

# echo "Job 25: SmolLM360 + FLORES (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 flores adam

# echo "Job 26: SmolLM360 + FLORES (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 flores icl algorithm.batch_size=1

# echo "Job 27: SmolLM360 + FLORES (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-360 flores lora

# echo "Job 28: OLMo2-1B-step10k + MetaMath (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step10k metamath adam

# echo "Job 29: OLMo2-1B-step10k + MetaMath (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step10k metamath icl algorithm.batch_size=1

# echo "Job 30: OLMo2-1B-step10k + MetaMath (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step10k metamath lora

# echo "Job 31: OLMo2-1B-step10k + FLORES (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step10k flores adam

# echo "Job 32: OLMo2-1B-step10k + FLORES (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step10k flores icl algorithm.batch_size=1

# echo "Job 33: OLMo2-1B-step10k + FLORES (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step10k flores lora

# echo "Job 34: OLMo2-1B-step20k + MetaMath (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step20k metamath adam

# echo "Job 35: OLMo2-1B-step20k + MetaMath (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step20k metamath icl algorithm.batch_size=1

# echo "Job 36: OLMo2-1B-step20k + MetaMath (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step20k metamath lora

# echo "Job 37: OLMo2-1B-step20k + FLORES (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step20k flores adam

# echo "Job 38: OLMo2-1B-step20k + FLORES (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step20k flores icl algorithm.batch_size=1

# echo "Job 39: OLMo2-1B-step20k + FLORES (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step20k flores lora

# echo "Job 40: OLMo2-1B-step30k + MetaMath (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step30k metamath adam

# echo "Job 41: OLMo2-1B-step30k + MetaMath (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step30k metamath icl algorithm.batch_size=1

# echo "Job 42: OLMo2-1B-step30k + MetaMath (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step30k metamath lora

# echo "Job 43: OLMo2-1B-step30k + FLORES (1 GPU, batch_size=8, strategy=adam)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step30k flores adam

# echo "Job 44: OLMo2-1B-step30k + FLORES (1 GPU, batch_size=1, strategy=icl)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step30k flores icl algorithm.batch_size=1

# echo "Job 45: OLMo2-1B-step30k + FLORES (1 GPU, batch_size=8, strategy=lora)"
# sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh olmo2-1b-step30k flores lora

echo "Job 46: SmolLM-step2750k + MetaMath (1 GPU, batch_size=8, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k metamath adam

echo "Job 47: SmolLM-step2750k + MetaMath (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k metamath icl algorithm.batch_size=1

echo "Job 48: SmolLM-step2750k + MetaMath (1 GPU, batch_size=8, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k metamath lora

echo "Job 49: SmolLM-step2750k + FLORES (1 GPU, batch_size=8, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k flores adam

echo "Job 50: SmolLM-step2750k + FLORES (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k flores icl algorithm.batch_size=1

echo "Job 51: SmolLM-step2750k + FLORES (1 GPU, batch_size=8, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step2750k flores lora

echo "Job 52: SmolLM-step4125k + MetaMath (1 GPU, batch_size=8, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k metamath adam

echo "Job 53: SmolLM-step4125k + MetaMath (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k metamath icl algorithm.batch_size=1

echo "Job 54: SmolLM-step4125k + MetaMath (1 GPU, batch_size=8, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k metamath lora

echo "Job 55: SmolLM-step4125k + FLORES (1 GPU, batch_size=8, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k flores adam

echo "Job 56: SmolLM-step4125k + FLORES (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k flores icl algorithm.batch_size=1

echo "Job 57: SmolLM-step4125k + FLORES (1 GPU, batch_size=8, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4125k flores lora

echo "Job 58: SmolLM-step4875k + MetaMath (1 GPU, batch_size=8, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k metamath adam

echo "Job 59: SmolLM-step4875k + MetaMath (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k metamath icl algorithm.batch_size=1

echo "Job 60: SmolLM-step4875k + MetaMath (1 GPU, batch_size=8, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k metamath lora

echo "Job 61: SmolLM-step4875k + FLORES (1 GPU, batch_size=8, strategy=adam)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k flores adam

echo "Job 62: SmolLM-step4875k + FLORES (1 GPU, batch_size=1, strategy=icl)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k flores icl algorithm.batch_size=1

echo "Job 63: SmolLM-step4875k + FLORES (1 GPU, batch_size=8, strategy=lora)"
sbatch --gres=gpu:a100l:1 slurm/run-experiment.sh smollm-step4875k flores lora

echo ""
echo "All jobs submitted! Check status with: squeue -u $USER"
