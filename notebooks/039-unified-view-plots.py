import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_json('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results-filtered.jsonl', lines=True)
SCRIPT_SIZE = 3704
df.loc[df['experiment_name'].isin(['urial', 'icl']), 'bits'] += SCRIPT_SIZE
BASELINE_SCRIPT_SIZE = 2952
df.loc[df['experiment_name'].isin(['baseline']), 'bits'] += BASELINE_SCRIPT_SIZE
ONLINE_CODING_SCRIPT_SIZE = 5904
df.loc[df['experiment_name'].isin(['online_coding']), 'bits'] += ONLINE_CODING_SCRIPT_SIZE

model_display_names = {
    'smollm': 'SmolLM2-1.7B',
    'qwen': 'Qwen2.5-1.5B',
    'smollm3': 'SmolLM3-3B',
    'smollm3-stage1': 'SmolLM3 3B',
    'olmo3-7b-step1414k': 'OLMo3 7B',
    'olmo3-32b-step656k': 'OLMo3 32B',
}

dataset_display_names = {
    'meta-math/MetaMathQA': 'GSM8K',
    'allenai/nllb': 'FLORES',
    'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl': 'IFEval',
}

dataset_metric_names = {
    'meta-math/MetaMathQA': 'Accuracy (%)',
    'allenai/nllb': 'BLEU',
    'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl': 'Score (%)',
}

percentage_datasets = {'meta-math/MetaMathQA', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl'}

method_display_names = {
    'icl': 'ICL',
    'lora': 'LoRA',
    'adam': 'Online Coding',
    'online_coding': 'Subset Training',
    'full_ft': 'Full Fine-tuning',
    'urial': 'URIAL',
    'lm_head': 'Linear Logits',
    'phase-one': 'Phase One',
    'phase-two': 'Unified View',
    'baseline': 'Baseline',
}

method_colors = {
    'icl': '#1f77b4',
    'lora': '#ff7f0e',
    'adam': '#2ca02c',
    'online_coding': '#d62728',
    'full_ft': '#9467bd',
    'urial': '#8c564b',
    'lm_head': '#e377c2',
    'phase-one': '#7f7f7f',
    'phase-two': '#1E88E5',
    'baseline': '#17becf',
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
datasets = ['meta-math/MetaMathQA', 'allenai/nllb', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl']

fig, axes = plt.subplots(2, 3, figsize=(14, 6))

all_handles = []
all_labels = []

for model_idx, model_name in enumerate(models):
    for dataset_idx, dataset_name in enumerate(datasets):
        ax = axes[model_idx, dataset_idx]

        filtered_df = df[df['dataset_name'] == dataset_name]
        filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

        zero_bits_df = filtered_df[filtered_df['bits'] == 0]
        zero_bits_performance = zero_bits_df['performance'].iloc[0] if len(zero_bits_df) > 0 else 0
        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
        if dataset_name in percentage_datasets:
            filtered_df['performance'] = filtered_df['performance'] * 100
            zero_bits_performance = zero_bits_performance * 100

        phase_two_data = filtered_df[filtered_df['experiment_name'] == 'phase-two']
        if len(phase_two_data) > 0:
            ax.scatter(phase_two_data['bits'], phase_two_data['performance'],
                       marker='o', alpha=0.7, s=60, color=method_colors['phase-two'],
                       label=method_display_names['phase-two'])

        bits, performance = compute_pareto_frontier(filtered_df, zero_bits_performance)

        ax.plot(bits, performance, linewidth=4, label='Pareto Frontier', color='gray', linestyle='-')

        if model_idx == 1:
            ax.set_xlabel('Program Length (bits)', fontsize=18)
        metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
        ax.set_ylabel(metric_name, fontsize=18)
        if model_idx == 0:
            dataset_display = dataset_display_names.get(dataset_name, dataset_name)
            ax.set_title(dataset_display, fontsize=20)
        if dataset_idx == 0:
            model_display = model_display_names.get(model_name, model_name)
            ax.annotate(model_display, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        fontsize=20, ha='right', va='center')
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
                   linewidth=4, color='gray')

            ax.plot([current_xlim[0], leftmost_bits],
                   [zero_bits_performance, zero_bits_performance],
                   linewidth=4, color='gray')

            rightmost_bits = bits[-1]
            rightmost_perf = performance[-1]
            ax.plot([rightmost_bits, 1e12],
                   [rightmost_perf, rightmost_perf],
                   linewidth=4, color='gray')

        handles, labels = ax.get_legend_handles_labels()
        for h, lbl in zip(handles, labels):
            if lbl not in all_labels:
                all_handles.append(h)
                all_labels.append(lbl)

fig.legend(all_handles, all_labels, loc='center left',
           bbox_to_anchor=(1.01, 0.5), fontsize=14)
plt.tight_layout()
plt.savefig('unified_view_pareto_curves.pdf', bbox_inches='tight', dpi=300)
