import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/results.csv')

# Display name mappings
model_display_names = {
    'smollm': 'SmolLM2-1.7B',
    'HuggingFaceTB/SmolLM2-360M': 'SmolLM2-360M',
    'olmo2-1b-step10k': 'OLMo2-1B (10k steps)',
    'olmo2-1b-step20k': 'OLMo2-1B (20k steps)',
    'olmo2-1b-step30k': 'OLMo2-1B (30k steps)',
    'smollm-step2750k': 'SmolLM2-1.7B (2750k steps)',
    'smollm-step4125k': 'SmolLM2-1.7B (4125k steps)',
    'smollm-step4875k': 'SmolLM2-1.7B (4875k steps)',
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

dataset_name = 'meta-math/MetaMathQA'
# dataset_name = 'cais/mmlu'
# dataset_name = 'allenai/nllb'
# dataset_name = 'ybisk/piqa'
# models = ['olmo2-1b-step10k', 'olmo2-1b-step20k', 'olmo2-1b-step30k']
models = ['smollm-step2750k','smollm-step4125k','smollm-step4875k', 'smollm']
# models = ['hubble-1b-500b-standard','hubble-1b-500b-perturbed']

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

# Create the plot
plt.figure(figsize=(12, 7))

# Filter by dataset
filtered_df = df[df['dataset_name'] == dataset_name].copy()

# Filter out 0 bits points (can't display on log scale anyway)
filtered_df = filtered_df[filtered_df['bits'] > 0].copy()

# Define colors for models
model_colors = plt.cm.tab10(range(len(models)))

# Plot each model's data and Pareto frontier
for i, model_name in enumerate(models):
    model_df = filtered_df[filtered_df['model_name'] == model_name].copy()

    # Plot all points for this model (lighter)
    plt.scatter(model_df['bits'], model_df['performance'],
                marker='o', alpha=0.3, s=30, color=model_colors[i])

    # Compute and plot Pareto frontier
    pareto_df = compute_pareto_frontier(model_df)

    # Sort pareto points by bits for line plotting
    pareto_df = pareto_df.sort_values('bits')

    model_display = model_display_names.get(model_name, model_name)
    plt.plot(pareto_df['bits'], pareto_df['performance'],
             marker='o', linewidth=2, markersize=8,
             label=model_display, color=model_colors[i])

plt.xlabel(r'$C(T, \delta \mid P)$', fontsize=14)
metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
plt.ylabel(metric_name, fontsize=14)
dataset_display = dataset_display_names.get(dataset_name, dataset_name)
plt.title(f'{dataset_display}', fontsize=14)
plt.xscale('log')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tmp.png', dpi=150)
print("Saved plot to tmp.png")
