import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_bytes(bits):
    bytes_val = bits / 8
    if bytes_val >= 1e9:
        return f'{bytes_val / 1e9:.0f}GB'
    elif bytes_val >= 1e6:
        return f'{bytes_val / 1e6:.0f}MB'
    elif bytes_val >= 1e3:
        return f'{bytes_val / 1e3:.0f}KB'
    else:
        return f'{bytes_val:.0f}B'


df = pd.read_json('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results.jsonl', lines=True)
SCRIPT_SIZE = 3704
df.loc[df['experiment_name'].isin(['urial', 'icl']), 'bits'] += SCRIPT_SIZE
BASELINE_SCRIPT_SIZE = 2952
df.loc[df['experiment_name'].isin(['baseline']), 'bits'] += BASELINE_SCRIPT_SIZE
ONLINE_CODING_SCRIPT_SIZE = 5704
df.loc[df['experiment_name'].isin(['online_coding']), 'bits'] += ONLINE_CODING_SCRIPT_SIZE
LORA_SCRIPT_SIZE = 2832
df.loc[df['experiment_name'].isin(['lora']), 'bits'] += LORA_SCRIPT_SIZE
BLORA_SCRIPT_SIZE = 8376
df.loc[df['experiment_name'].isin(['blora']), 'bits'] += BLORA_SCRIPT_SIZE

model_display_names = {
    'smollm': 'SmolLM2-1.7B',
    'qwen': 'Qwen2.5-1.5B',
    'smollm3': 'SmolLM3-3B',
    'smollm3-stage1': 'SmolLM3',
    'olmo3-7b-step1414k': 'Olmo3 7B',
    'olmo3-32b-step656k': 'Olmo3 32B',
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

dataset_task_names = {
    'meta-math/MetaMathQA': 'Math',
    'allenai/nllb': 'Translation',
    'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl': 'Instruction',
}

percentage_datasets = {'meta-math/MetaMathQA', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl'}

def compute_pareto_frontier(model_df):
    df_sorted = model_df.sort_values('bits').copy()
    pareto_points = []
    max_performance = -np.inf

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

fig, axes = plt.subplots(3, 2, figsize=(10, 9))

all_handles = []
all_labels = []

for dataset_idx, dataset_name in enumerate(datasets):
    for model_idx, model_name in enumerate(models):
        ax = axes[dataset_idx, model_idx]

        filtered_df = df[df['dataset_name'] == dataset_name]
        filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
        if dataset_name in percentage_datasets:
            filtered_df['performance'] = filtered_df['performance'] * 100

        lm_head_data = filtered_df[filtered_df['experiment_name'] == 'lm_head']
        if len(lm_head_data) > 0:
            best_lm_head = lm_head_data.loc[lm_head_data['performance'].idxmax()]
            ax.scatter(best_lm_head['bits'], best_lm_head['performance'],
                       marker='o', alpha=0.8, s=80, color='#1E88E5', label='Linear Projection', zorder=10)

        bits, performance = compute_pareto_frontier(filtered_df)

        ax.plot(bits, performance,
                linewidth=4, label='Pareto Frontier', color='gray', zorder=5)

        if dataset_idx == len(datasets) - 1:
            ax.set_xlabel('Program Length (bits)', fontsize=18)
        metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)
        task_name = dataset_task_names.get(dataset_name, '')
        if model_idx == 0:
            ax.set_ylabel(f'{task_name}\n{metric_name}', fontsize=20, rotation=0, ha='center', va='center', labelpad=70)
        ax.set_xscale('log')
        if dataset_idx == 0:
            model_display = model_display_names.get(model_name, model_name)
            ax.set_title(model_display, fontsize=20)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, linewidth=0.5)

        ax.set_xlim(left=BASELINE_SCRIPT_SIZE // 2, right=1e12)

        tick_bits = [1e3, 1e5, 1e7, 1e9, 1e11]
        ax.set_xticks(tick_bits)
        ax.set_xticklabels([f'$10^{{{int(np.log10(b))}}}$' for b in tick_bits])

        if len(bits) > 0:
            rightmost_bits = bits[-1]
            rightmost_perf = performance[-1]
            ax.plot([rightmost_bits, 1e12],
                   [rightmost_perf, rightmost_perf],
                   linewidth=4, color='gray', zorder=5)

        handles, labels = ax.get_legend_handles_labels()
        for h, lbl in zip(handles, labels):
            if lbl not in all_labels:
                all_handles.append(h)
                all_labels.append(lbl)

for dataset_idx in range(len(datasets)):
    row_axes = [axes[dataset_idx, model_idx] for model_idx in range(len(models))]
    y_mins = [ax.get_ylim()[0] for ax in row_axes]
    y_maxs = [ax.get_ylim()[1] for ax in row_axes]
    shared_ylim = (min(y_mins), max(y_maxs))
    for ax in row_axes:
        ax.set_ylim(shared_ylim)

fig.legend(all_handles, all_labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=14, title='Methods', title_fontsize=16)
plt.tight_layout()
plt.savefig('lm_head_plot.pdf', bbox_inches='tight', dpi=300)

fig2, ax2 = plt.subplots(figsize=(10, 6))

metamath_dataset = 'meta-math/MetaMathQA'
model_name = 'olmo3-7b-step1414k'

filtered_df = df[df['dataset_name'] == metamath_dataset]
filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
filtered_df['performance'] = filtered_df['performance'] * 100

lm_head_data = filtered_df[filtered_df['experiment_name'] == 'lm_head']
if len(lm_head_data) > 0:
    best_lm_head = lm_head_data.loc[lm_head_data['performance'].idxmax()]
    ax2.scatter(best_lm_head['bits'], best_lm_head['performance'],
               marker='o', alpha=0.8, s=200, color='#1E88E5', label='Linear Projection', zorder=10)

bits, performance = compute_pareto_frontier(filtered_df)

ax2.plot(bits, performance,
        linewidth=6, label='Pareto Frontier', color='gray', zorder=5)

ax2.set_xlabel('Program Length (bits)', fontsize=28)
ax2.set_ylabel('Accuracy (%)', fontsize=28)
ax2.set_xscale('log')
ax2.tick_params(axis='both', labelsize=24)
ax2.grid(True)

ax2.set_xlim(left=BASELINE_SCRIPT_SIZE // 2, right=8e10)

tick_bits = [1e3, 1e5, 1e7, 1e9, 1e11]
ax2.set_xticks(tick_bits)
ax2.set_xticklabels([f'$10^{{{int(np.log10(b))}}}$' for b in tick_bits])

if len(bits) > 0:
    rightmost_bits = bits[-1]
    rightmost_perf = performance[-1]
    ax2.plot([rightmost_bits, 8e10],
           [rightmost_perf, rightmost_perf],
           linewidth=6, color='gray', zorder=5)

ax2.legend(fontsize=22, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)
plt.tight_layout()
plt.savefig('lm_head_metamath_plot.pdf', bbox_inches='tight', dpi=300)
