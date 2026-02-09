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
    'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl': 'Instructions',
}

percentage_datasets = {'meta-math/MetaMathQA', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl'}

method_display_names = {
    'icl': 'ICL',
    'lora': 'LoRA',
    'adam': 'Online Coding',
    'online_coding': 'Subset Training',
    # 'full_ft': 'Full Fine-tuning',
    'urial': 'URIAL',
    # 'lm_head': 'Linear Logits',
    'blora': 'Bayesian LoRA',
    'baseline': 'Base Model',
}

def compute_pareto_frontier(model_df):
    df_sorted = model_df.sort_values('bits').copy()
    pareto_points = []
    pareto_indices = []
    max_performance = -np.inf

    for idx, row in df_sorted.iterrows():
        if row['performance'] >= max_performance:
            pareto_points.append({'bits': row['bits'], 'performance': row['performance']})
            pareto_indices.append(idx)
            max_performance = row['performance']

    bits = []
    performance = []
    for i, point in enumerate(pareto_points):
        if i > 0:
            bits.append(point['bits'])
            performance.append(pareto_points[i-1]['performance'])

        bits.append(point['bits'])
        performance.append(point['performance'])

    return bits, performance, pareto_indices

models = ['smollm3-stage1', 'olmo3-7b-step1414k', 'olmo3-32b-step656k']
datasets = ['meta-math/MetaMathQA', 'allenai/nllb', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl']

olmo3_32b_bits = 32233522176 * 16
olmo3_7b_bits = 7298011136 * 16
smollm3_bits = 3075098624 * 16

model_bits = {
    'smollm3-stage1': smollm3_bits,
    'olmo3-7b-step1414k': olmo3_7b_bits,
    'olmo3-32b-step656k': olmo3_32b_bits,
}

imagenet_image_bits = 224 * 224 * 3 * 8
bert_base_bits = 110_000_000 * 32
python_100_lines_bits = 100 * 50 * 8

fig, axes = plt.subplots(3, 3, figsize=(14, 9))

color_map = {
    'baseline': '#004D40',
    'icl': '#D81B60',
    'urial': '#5E35B1',
    'lora': '#00ACC1',
    'blora': '#FFC107',
    'adam': '#FE6100',
    'online_coding': '#1E88E5',
}

marker_map = {
    'baseline': 'o',
    'icl': 's',
    'urial': '^',
    'lora': 'P',
    'blora': 'D',
    'adam': 'X',
    'online_coding': 'v',
}

method_handles = []
method_labels = []
reference_handles = []
reference_labels = []

pareto_data = {}
reference_data = {}

for dataset_idx, dataset_name in enumerate(datasets):
    for model_idx, model_name in enumerate(models):
        filtered_df = df[df['dataset_name'] == dataset_name]
        filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

        online_coding_df = filtered_df[filtered_df['experiment_name'] == 'online_coding']
        online_coding_df = online_coding_df[online_coding_df['strategy_hparams'].apply(lambda x: x.get('max_examples') is None)]
        dataset_size_bits = online_coding_df['bits'].max()

        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
        filtered_df = filtered_df[filtered_df['experiment_name'].isin(method_display_names.keys())].copy()
        if dataset_name in percentage_datasets:
            filtered_df['performance'] = filtered_df['performance'] * 100

        bits, performance, pareto_indices = compute_pareto_frontier(filtered_df)
        pareto_df = filtered_df.loc[pareto_indices]

        ax = axes[dataset_idx, model_idx]

        experiment_names = pareto_df['experiment_name'].unique()
        for exp_name in experiment_names:
            if exp_name not in method_display_names:
                continue
            exp_data = pareto_df[pareto_df['experiment_name'] == exp_name]
            exp_display = method_display_names.get(exp_name, exp_name)
            ax.scatter(exp_data['bits'], exp_data['performance'],
                       marker=marker_map[exp_name], alpha=0.8, s=80, color=color_map[exp_name], label=exp_display, zorder=10)

        ax.plot(bits, performance,
                linewidth=4, label='Pareto Frontier', color='gray', zorder=5)

        if dataset_idx == len(datasets) - 1:
            ax.set_xlabel('Program Length (bits)', fontsize=18)
        metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)
        task_name = dataset_task_names.get(dataset_name, '')
        if model_idx == 0:
            ax.set_ylabel(f'{dataset_display}\n{metric_name}\n\n({task_name})', fontsize=20, rotation=0, ha='center', va='center', labelpad=70)
        ax.set_xscale('log')
        if dataset_idx == 0:
            model_display = model_display_names.get(model_name, model_name)
            ax.set_title(model_display, fontsize=20)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, linewidth=0.5)

        xlim_left = BASELINE_SCRIPT_SIZE // 2
        xlim_right = 1e12
        ax.set_xlim(left=xlim_left, right=xlim_right)

        tick_bits = [1e3, 1e5, 1e7, 1e9, 1e11]
        ax.set_xticks(tick_bits)
        ax.set_xticklabels([f'$10^{{{int(np.log10(b))}}}$' for b in tick_bits])

        reference_lines = [
            (python_100_lines_bits, '100 lines Python'),
            (imagenet_image_bits, 'ImageNet image'),
            (bert_base_bits, 'BERT-base'),
            (model_bits[model_name], 'Model Size'),
            (dataset_size_bits, 'Dataset Size'),
        ]
        if model_idx == 0 and dataset_idx == 0:
            print(reference_lines)
        reference_lines.sort(key=lambda x: x[0])
        pareto_data[(dataset_idx, model_idx)] = (bits, performance)
        reference_data[(dataset_idx, model_idx)] = reference_lines

        if len(bits) > 0:
            rightmost_bits = bits[-1]
            rightmost_perf = performance[-1]
            ax.plot([rightmost_bits, xlim_right],
                   [rightmost_perf, rightmost_perf],
                   linewidth=4, color='gray', zorder=5)

        handles, labels = ax.get_legend_handles_labels()
        for h, lbl in zip(handles, labels):
            if lbl not in method_labels:
                method_handles.append(h)
                method_labels.append(lbl)

for dataset_idx in range(len(datasets)):
    row_axes = [axes[dataset_idx, model_idx] for model_idx in range(len(models))]
    y_mins = [ax.get_ylim()[0] for ax in row_axes]
    y_maxs = [ax.get_ylim()[1] for ax in row_axes]
    shared_ylim = (min(y_mins), max(y_maxs))
    for ax in row_axes:
        ax.set_ylim(shared_ylim)

for dataset_idx, dataset_name in enumerate(datasets):
    for model_idx, model_name in enumerate(models):
        ax = axes[dataset_idx, model_idx]
        bits, performance = pareto_data[(dataset_idx, model_idx)]
        reference_lines = reference_data[(dataset_idx, model_idx)]
        ylim = ax.get_ylim()
        ref_colors = ['#7986CB', '#4DB6AC', '#FF8A65', '#BA68C8', '#A1887F']
        for i, (bits_value, label) in enumerate(reference_lines):
            pareto_y = performance[0] if len(performance) > 0 else ylim[1]
            for j in range(len(bits)):
                if bits[j] <= bits_value:
                    pareto_y = performance[j]
            line = ax.plot([bits_value, bits_value], [ylim[0], pareto_y], color=ref_colors[i], linestyle='--', linewidth=2, label=label)
            if label not in reference_labels:
                reference_handles.append(line[0])
                reference_labels.append(label)

fig.legend(method_handles, method_labels, loc='center left', bbox_to_anchor=(1.01, 0.6), fontsize=14, title='Methods', title_fontsize=16)
fig.legend(reference_handles, reference_labels, loc='center left', bbox_to_anchor=(1.01, 0.3), fontsize=14, title='References', title_fontsize=16)
plt.tight_layout()
plt.savefig('pareto_estimation.pdf', bbox_inches='tight', dpi=300)

fig2, axes2 = plt.subplots(3, 3, figsize=(14, 9))

method_handles2 = []
method_labels2 = []

for dataset_idx, dataset_name in enumerate(datasets):
    for model_idx, model_name in enumerate(models):
        filtered_df = df[df['dataset_name'] == dataset_name]
        filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
        filtered_df = filtered_df[filtered_df['experiment_name'].isin(method_display_names.keys())].copy()
        if dataset_name in percentage_datasets:
            filtered_df['performance'] = filtered_df['performance'] * 100

        bits, performance, pareto_indices = compute_pareto_frontier(filtered_df)

        ax = axes2[dataset_idx, model_idx]

        experiment_names = filtered_df['experiment_name'].unique()
        for exp_name in experiment_names:
            if exp_name not in method_display_names:
                continue
            exp_data = filtered_df[filtered_df['experiment_name'] == exp_name]
            exp_display = method_display_names.get(exp_name, exp_name)
            ax.scatter(exp_data['bits'], exp_data['performance'],
                       marker=marker_map[exp_name], alpha=0.8, s=80, color=color_map[exp_name], label=exp_display, zorder=10)

        ax.plot(bits, performance,
                linewidth=4, label='Pareto Frontier', color='gray', zorder=5)

        if dataset_idx == len(datasets) - 1:
            ax.set_xlabel('Program Length (bits)', fontsize=18)
        metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)
        task_name = dataset_task_names.get(dataset_name, '')
        if model_idx == 0:
            ax.set_ylabel(f'{dataset_display}\n{metric_name}\n\n({task_name})', fontsize=20, rotation=0, ha='center', va='center', labelpad=70)
        ax.set_xscale('log')
        if dataset_idx == 0:
            model_display = model_display_names.get(model_name, model_name)
            ax.set_title(model_display, fontsize=20)
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(True, linewidth=0.5)

        xlim_left = BASELINE_SCRIPT_SIZE // 2
        xlim_right = 1e12
        ax.set_xlim(left=xlim_left, right=xlim_right)

        tick_bits = [1e3, 1e5, 1e7, 1e9, 1e11]
        ax.set_xticks(tick_bits)
        ax.set_xticklabels([f'$10^{{{int(np.log10(b))}}}$' for b in tick_bits])

        if len(bits) > 0:
            rightmost_bits = bits[-1]
            rightmost_perf = performance[-1]
            ax.plot([rightmost_bits, xlim_right],
                   [rightmost_perf, rightmost_perf],
                   linewidth=4, color='gray', zorder=5)

        handles, labels = ax.get_legend_handles_labels()
        for h, lbl in zip(handles, labels):
            if lbl not in method_labels2:
                method_handles2.append(h)
                method_labels2.append(lbl)

for dataset_idx in range(len(datasets)):
    row_axes = [axes2[dataset_idx, model_idx] for model_idx in range(len(models))]
    y_mins = [ax.get_ylim()[0] for ax in row_axes]
    y_maxs = [ax.get_ylim()[1] for ax in row_axes]
    shared_ylim = (min(y_mins), max(y_maxs))
    for ax in row_axes:
        ax.set_ylim(shared_ylim)

fig2.legend(method_handles2, method_labels2, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=14, title='Methods', title_fontsize=16)
plt.tight_layout()
plt.savefig('pareto_estimation_appendix.pdf', bbox_inches='tight', dpi=300)
