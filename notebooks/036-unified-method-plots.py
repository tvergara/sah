import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_json('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results.jsonl', lines=True)
model_display_names = {
    'smollm': 'SmolLM2-1.7B',
    'qwen': 'Qwen2.5-1.5B',
    'smollm3': 'SmolLM3-3B',
    'smollm3-stage1': 'SmolLM3 (Pre-trained)',
    'olmo3-7b-step1414k': 'Olmo3 7B (Pre-trained)',
    'olmo3-32b-step656k': 'Olmo3 32B (Pre-trained)',
}

dataset_display_names = {
    'meta-math/MetaMathQA': 'GSM8K',
    'allenai/nllb': 'FLORES',
}

dataset_metric_names = {
    'meta-math/MetaMathQA': 'Accuracy',
    'allenai/nllb': 'BLEU',
}

method_display_names = {
    'icl': 'ICL',
    'lora': 'LoRA',
    'adam': 'Online Coding',
    'online_coding': 'Subset Training',
    'full_ft': 'Full Fine-tuning',
    'urial': 'URIAL',
    'lm_head': 'Linear Logits',
    'phase-one': 'Phase One',
    'phase-two': 'Phase Two',
}

def compute_pareto_frontier(model_df, zero_bit_perf=None):
    df_sorted = model_df.sort_values('bits').copy()
    pareto_points = []
    max_performance = zero_bit_perf if zero_bit_perf is not None else -np.inf

    for idx, row in df_sorted.iterrows():
        if row['performance'] >= max_performance:
            pareto_points.append({'bits': row['bits'], 'performance': row['performance']})
            max_performance = row['performance']

    bits = []
    performance = []
    for i, point in enumerate(pareto_points):
        if i > 0:
            bits.append(point['bits'])
            performance.append(pareto_points[i-1]['performance'])

        bits.append(point['bits'])
        performance.append(point['performance'])

    return bits, performance

models = ['smollm3-stage1', 'olmo3-7b-step1414k']
datasets = ['meta-math/MetaMathQA', 'allenai/nllb']

model_20_params_bits = 20 * 32
tweet_bits = 280 * 8
imagenet_image_bits = 224 * 224 * 3 * 8
bert_base_bits = 110_000_000 * 32
python_100_lines_bits = 100 * 50 * 8
english_wikipedia_bits = 80 * 1024 * 1024 * 1024 * 8

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()

color_map = {}
all_experiment_names = df['experiment_name'].unique()
for name in all_experiment_names:
    if name == 'phase-one':
        color_map[name] = 'blue'
    elif name == 'phase-two':
        color_map[name] = 'red'
    else:
        color_map[name] = 'gray'

all_handles = []
all_labels = []

for idx, (model_name, dataset_name) in enumerate([(m, d) for m in models for d in datasets]):
    ax = axes[idx]

    filtered_df = df[df['dataset_name'] == dataset_name]
    filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

    zero_bits_performance = filtered_df[filtered_df['bits'] == 0]['performance'].iloc[0]
    filtered_df = filtered_df[filtered_df['bits'] > 0].copy()

    experiment_names = filtered_df['experiment_name'].unique()

    for exp_name in experiment_names:
        exp_data = filtered_df[filtered_df['experiment_name'] == exp_name]
        exp_display = method_display_names.get(exp_name, exp_name)
        ax.scatter(exp_data['bits'], exp_data['performance'],
                   marker='o', alpha=0.6, s=50, color=color_map[exp_name], label=exp_display)

    bits, performance = compute_pareto_frontier(filtered_df, zero_bits_performance)

    ax.plot(bits, performance,
            linewidth=2, label='Pareto Frontier', color='gray')

    reference_lines = [
        (model_20_params_bits, '20-param model'),
        (tweet_bits, 'Tweet'),
        (python_100_lines_bits, '100 lines Python'),
        (imagenet_image_bits, 'ImageNet image'),
        (bert_base_bits, 'BERT-base'),
        (english_wikipedia_bits, 'English Wikipedia')
    ]
    reference_lines.sort(key=lambda x: x[0])

    gradient_colors = plt.cm.viridis(np.linspace(0, 1, len(reference_lines)))

    for i, (bits_value, label) in enumerate(reference_lines):
        ax.axvline(x=bits_value, color=gradient_colors[i], linestyle='--', linewidth=2, label=label)

    ax.set_xlabel('Program Length (bits)', fontsize=18)
    metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
    ax.set_ylabel(metric_name, fontsize=18)
    model_display = model_display_names.get(model_name, model_name)
    dataset_display = dataset_display_names.get(dataset_name, dataset_name)
    ax.set_title(f'{model_display} on {dataset_display}', fontsize=20)
    ax.set_xscale('log')
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True)

    current_xlim = ax.get_xlim()
    ax.set_xlim(left=current_xlim[0], right=1e12)

    if len(bits) > 0:
        leftmost_bits = bits[0]
        leftmost_perf = performance[0]

        ax.plot([leftmost_bits, leftmost_bits],
               [zero_bits_performance, leftmost_perf],
               linewidth=2, color='gray')

        ax.plot([current_xlim[0], leftmost_bits],
               [zero_bits_performance, zero_bits_performance],
               linewidth=2, color='gray')

        rightmost_bits = bits[-1]
        rightmost_perf = performance[-1]
        ax.plot([rightmost_bits, 1e12],
               [rightmost_perf, rightmost_perf],
               linewidth=2, color='gray')

    handles, labels = ax.get_legend_handles_labels()
    for h, lbl in zip(handles, labels):
        if lbl not in all_labels:
            all_handles.append(h)
            all_labels.append(lbl)

fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=18)
plt.tight_layout()
plt.savefig('pareto_curves_unified_methods.png', bbox_inches='tight', dpi=300)
