import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/results.csv')

# Display name mappings
model_display_names = {
    'smollm': 'SmolLM2-1.7B',
    # 'HuggingFaceTB/SmolLM2-360M': 'SmolLM2-360M',
    'olmo2-1b-step10k': 'OLMo2-1B (10k steps)',
    'olmo2-1b-step20k': 'OLMo2-1B (20k steps)',
    'olmo2-1b-step30k': 'OLMo2-1B (30k steps)',
    'qwen': 'Qwen2.5-1.5B',
}

dataset_display_names = {
    'meta-math/MetaMathQA': 'GSM8K',
    'cais/mmlu': 'MMLU',
    'allenai/nllb': 'NLLB',
}

dataset_metric_names = {
    'meta-math/MetaMathQA': 'Accuracy',
    'cais/mmlu': 'Accuracy',
    'allenai/nllb': 'BLEU',
}

method_display_names = {
    'icl': 'ICL',
    'lora': 'LoRA',
    'adam': 'Online Coding',
}

def compute_pareto_frontier(df):
    """
    Compute Pareto frontier: for each performance level, minimum bits needed.

    A point is on the Pareto frontier if no other point has both:
    - fewer or equal bits AND better or equal performance
    (with at least one strict inequality)

    Creates step function by adding intermediate points to avoid interpolation.
    """
    # Sort by bits ascending
    df_sorted = df.sort_values('bits').copy()

    # Track maximum performance seen so far
    pareto_points = []
    max_performance = -np.inf

    for idx, row in df_sorted.iterrows():
        if row['performance'] >= max_performance:
            # Add a point with same bits but previous performance to create step function
            if len(pareto_points) > 0:
                step_point = row.copy()
                step_point['performance'] = max_performance
                step_point['bits'] -= 0.001
                pareto_points.append(step_point)

            # This point is on the frontier
            pareto_points.append(row)
            max_performance = row['performance']

    return pd.DataFrame(pareto_points)

# dataset_name = 'meta-math/MetaMathQA'
# dataset_name = 'cais/mmlu'
dataset_name = 'allenai/nllb'
model_name = 'olmo2-7b-step464k'
# model_name = 'olmo2-1b-step30k'
# model_name = 'HuggingFaceTB/SmolLM2-360M'
# model_name = 'qwen'

# Filter by dataset and model only - gather ALL points
filtered_df = df[df['dataset_name'] == dataset_name]
filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

# Get performance at 0 bits before filtering
zero_bits_performance = filtered_df[filtered_df['bits'] == 0]['performance'].iloc[0]

# Filter out 0 bits points (can't display on log scale anyway)
filtered_df = filtered_df[filtered_df['bits'] > 0].copy()

plt.figure(figsize=(10, 6))

# Get unique experiment names and assign colors
experiment_names = filtered_df['experiment_name'].unique()
colors = plt.cm.tab10(range(len(experiment_names)))

# Plot points colored by experiment_name
for i, exp_name in enumerate(experiment_names):
    exp_data = filtered_df[filtered_df['experiment_name'] == exp_name]
    exp_display = method_display_names.get(exp_name, exp_name)
    plt.scatter(exp_data['bits'], exp_data['performance'],
                marker='o', alpha=0.6, s=50, color=colors[i], label=exp_display)

# Compute and plot single Pareto frontier across all experiments
pareto_df = compute_pareto_frontier(filtered_df)
pareto_df = pareto_df.sort_values('bits')

plt.plot(pareto_df['bits'], pareto_df['performance'],
         linewidth=2,
         label='Pareto Frontier', color='gray', linestyle='--')

plt.axhline(y=zero_bits_performance, color='red', linestyle=':', linewidth=2, label='0-bit performance')

model_20_params_bits = 20 * 32
plt.axvline(x=model_20_params_bits, color='cyan', linestyle=':', linewidth=2, label='20-param model')

tweet_bits = 280 * 8
plt.axvline(x=tweet_bits, color='orange', linestyle=':', linewidth=2, label='Tweet')

imagenet_image_bits = 224 * 224 * 3 * 8
plt.axvline(x=imagenet_image_bits, color='blue', linestyle=':', linewidth=2, label='ImageNet image')

bert_base_bits = 110_000_000 * 32
plt.axvline(x=bert_base_bits, color='green', linestyle=':', linewidth=2, label='BERT-base')

python_100_lines_bits = 100 * 50 * 8
plt.axvline(x=python_100_lines_bits, color='purple', linestyle=':', linewidth=2, label='100 lines Python')

plt.xlabel(r'$C(T, \delta \mid P)$', fontsize=14)
metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
plt.ylabel(metric_name, fontsize=14)
model_display = model_display_names.get(model_name, model_name)
dataset_display = dataset_display_names.get(dataset_name, dataset_name)
plt.title(f'{model_display} on {dataset_display}')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.savefig('tmp.png')
