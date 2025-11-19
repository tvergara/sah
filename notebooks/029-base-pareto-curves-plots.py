import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/results.csv')

model_display_names = {
    'smollm': 'SmolLM2-1.7B',
    'qwen': 'Qwen2.5-1.5B',
    'smollm3': 'SmolLM3-3B',
}

dataset_display_names = {
    'meta-math/MetaMathQA': 'GSM8K',
    'allenai/nllb': 'NLLB',
}

dataset_metric_names = {
    'meta-math/MetaMathQA': 'Accuracy',
    'allenai/nllb': 'BLEU',
}

method_display_names = {
    'icl': 'ICL',
    'lora': 'LoRA',
    'adam': 'Online Coding',
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

models = ['smollm', 'qwen', 'smollm3']
datasets = ['meta-math/MetaMathQA', 'allenai/nllb']

model_20_params_bits = 20 * 32
tweet_bits = 280 * 8
imagenet_image_bits = 224 * 224 * 3 * 8
bert_base_bits = 110_000_000 * 32
python_100_lines_bits = 100 * 50 * 8
english_wikipedia_bits = 80 * 1024 * 1024 * 1024 * 8

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()

all_handles = []
all_labels = []

for idx, (model_name, dataset_name) in enumerate([(m, d) for m in models for d in datasets]):
    ax = axes[idx]

    filtered_df = df[df['dataset_name'] == dataset_name]
    filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

    zero_bits_performance = filtered_df[filtered_df['bits'] == 0]['performance'].iloc[0]
    filtered_df = filtered_df[filtered_df['bits'] > 0].copy()

    experiment_names = filtered_df['experiment_name'].unique()
    colors = plt.cm.tab10(range(len(experiment_names)))

    for i, exp_name in enumerate(experiment_names):
        exp_data = filtered_df[filtered_df['experiment_name'] == exp_name]
        exp_display = method_display_names.get(exp_name, exp_name)
        ax.scatter(exp_data['bits'], exp_data['performance'],
                   marker='o', alpha=0.6, s=50, color=colors[i], label=exp_display)

    pareto_df = compute_pareto_frontier(filtered_df)
    pareto_df = pareto_df.sort_values('bits')

    ax.plot(pareto_df['bits'], pareto_df['performance'],
            linewidth=2, label='Pareto Frontier', color='gray', linestyle='--')

    ax.axhline(y=zero_bits_performance, color='red', linestyle=':', linewidth=2, label='0-bit performance')
    ax.axvline(x=model_20_params_bits, color='cyan', linestyle=':', linewidth=2, label='20-param model')
    ax.axvline(x=tweet_bits, color='orange', linestyle=':', linewidth=2, label='Tweet')
    ax.axvline(x=imagenet_image_bits, color='blue', linestyle=':', linewidth=2, label='ImageNet image')
    ax.axvline(x=bert_base_bits, color='green', linestyle=':', linewidth=2, label='BERT-base')
    ax.axvline(x=python_100_lines_bits, color='purple', linestyle=':', linewidth=2, label='100 lines Python')
    ax.axvline(x=english_wikipedia_bits, color='brown', linestyle=':', linewidth=2, label='English Wikipedia')

    ax.set_xlabel(r'$C(T, \delta \mid P)$', fontsize=14)
    metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
    ax.set_ylabel(metric_name, fontsize=14)
    model_display = model_display_names.get(model_name, model_name)
    dataset_display = dataset_display_names.get(dataset_name, dataset_name)
    ax.set_title(f'{model_display} on {dataset_display}', fontsize=16)
    ax.set_xscale('log')
    ax.grid(True)

    if idx == 0:
        handles, labels = ax.get_legend_handles_labels()
        all_handles = handles
        all_labels = labels

fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=12)
plt.tight_layout()
plt.savefig('pareto_curves_all.png', bbox_inches='tight', dpi=300)
