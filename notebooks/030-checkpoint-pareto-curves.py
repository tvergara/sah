import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/results.csv')

model_display_names = {
    'smollm': 'SmolLM2-1.7B',
    'smollm-step2750k': 'SmolLM2-1.7B (2750k steps)',
    'smollm-step4125k': 'SmolLM2-1.7B (4125k steps)',
    'smollm-step4875k': 'SmolLM2-1.7B (4875k steps)',
}

dataset_display_names = {
    'meta-math/MetaMathQA': 'GSM8K',
    'allenai/nllb': 'NLLB',
}

dataset_metric_names = {
    'meta-math/MetaMathQA': 'Accuracy',
    'allenai/nllb': 'BLEU',
}

def compute_pareto_frontier(df):
    df_sorted = df.sort_values('bits').copy()
    pareto_points = []
    max_performance = -np.inf

    for idx, row in df_sorted.iterrows():
        if row['performance'] >= max_performance:
            if len(pareto_points) > 0:
                step_point = row.copy()
                step_point['performance'] = max_performance
                step_point['bits'] -= 0.001
                pareto_points.append(step_point)
            pareto_points.append(row)
            max_performance = row['performance']

    return pd.DataFrame(pareto_points)

models = ['smollm-step2750k', 'smollm-step4125k', 'smollm-step4875k', 'smollm']
datasets = ['meta-math/MetaMathQA', 'allenai/nllb']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

all_handles = []
all_labels = []

for idx, dataset_name in enumerate(datasets):
    ax = axes[idx]

    filtered_df = df[df['dataset_name'] == dataset_name].copy()
    filtered_df = filtered_df[filtered_df['bits'] > 0].copy()

    model_colors = plt.cm.tab10(range(len(models)))

    for i, model_name in enumerate(models):
        model_df = filtered_df[filtered_df['model_name'] == model_name].copy()

        ax.scatter(model_df['bits'], model_df['performance'],
                   marker='o', alpha=0.3, s=30, color=model_colors[i])

        pareto_df = compute_pareto_frontier(model_df)
        pareto_df = pareto_df.sort_values('bits')

        model_display = model_display_names.get(model_name, model_name)
        ax.plot(pareto_df['bits'], pareto_df['performance'],
                marker='o', linewidth=2, markersize=8,
                label=model_display, color=model_colors[i])

    ax.set_xlabel(r'$C(T, \delta \mid P)$', fontsize=14)
    metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
    ax.set_ylabel(metric_name, fontsize=14)
    dataset_display = dataset_display_names.get(dataset_name, dataset_name)
    ax.set_title(f'{dataset_display}', fontsize=16)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    if idx == 0:
        handles, labels = ax.get_legend_handles_labels()
        all_handles = handles
        all_labels = labels

fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12)
plt.tight_layout()
plt.savefig('checkpoint_pareto_curves.png', bbox_inches='tight', dpi=300)
