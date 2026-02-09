import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

olmo3_32b_bits = 32233522176 * 16
olmo3_7b_bits = 7298011136 * 16
smollm3_bits = 3075098624 * 16

model_bits = {
    'smollm3-stage1': smollm3_bits,
    'olmo3-7b-step1414k': olmo3_7b_bits,
    'olmo3-32b-step656k': olmo3_32b_bits,
}

df = pd.read_json('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results-filtered.jsonl', lines=True)
SCRIPT_SIZE = 3704
df.loc[df['experiment_name'].isin(['urial', 'icl']), 'bits'] += SCRIPT_SIZE
online_coding_gsm8k = df[(df['experiment_name'] == 'online_coding') &
                         (df['dataset_name'] == 'meta-math/MetaMathQA') &
                         (df['model_name'].isin(['smollm3-stage1', 'olmo3-7b-step1414k'])) &
                         (df['strategy_hparams'].apply(lambda x: x.get('max_examples') is None))]
online_coding_gsm8k = online_coding_gsm8k.sort_values(['model_name', 'bits'], ascending=[True, False])

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
    'meta-math/MetaMathQA': 'Accuracy',
    'allenai/nllb': 'BLEU',
    'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl': 'Score',
}

method_display_names = {
    'icl': 'ICL',
    'lora': 'LoRA',
    'adam': 'Online Coding',
    'online_coding': 'Subset Training',
    # 'full_ft': 'Full Fine-tuning',
    'urial': 'URIAL',
    # 'lm_head': 'Linear Logits',
    'blora': 'Bayesian LoRA',
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

models = ['smollm3-stage1', 'olmo3-7b-step1414k', 'olmo3-32b-step656k']
datasets = ['meta-math/MetaMathQA', 'allenai/nllb', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl']

model_200_params_bits = 200 * 32
tweet_bits = 280 * 8
imagenet_image_bits = 224 * 224 * 3 * 8
bert_base_bits = 110_000_000 * 32
python_100_lines_bits = 100 * 50 * 8
english_wikipedia_bits = 80 * 1024 * 1024 * 1024 * 8

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

reference_handles = []
reference_labels = []

for idx, (model_name, dataset_name) in enumerate([(m, d) for m in models for d in datasets]):
    ax = axes[idx]

    filtered_df = df[df['dataset_name'] == dataset_name]
    filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

    online_coding_df = filtered_df[filtered_df['experiment_name'] == 'online_coding']
    online_coding_df = online_coding_df[online_coding_df['strategy_hparams'].apply(lambda x: x.get('max_examples') is None)]
    dataset_size_bits = online_coding_df['bits'].max()

    zero_bits_performance = filtered_df[filtered_df['bits'] == 0]['performance'].iloc[0]
    filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
    filtered_df = filtered_df[filtered_df['experiment_name'].isin(method_display_names.keys())].copy()
    filtered_df = filtered_df[filtered_df['performance'] >= zero_bits_performance * 0.9].copy()

    bits, performance = compute_pareto_frontier(filtered_df, zero_bits_performance)

    ax.plot(bits, performance,
            linewidth=2, label='Pareto Frontier', color='gray')

    reference_lines = [
        (model_200_params_bits, '200-param model'),
        (tweet_bits, 'Tweet'),
        (python_100_lines_bits, '100 lines Python'),
        (imagenet_image_bits, 'ImageNet image'),
        (bert_base_bits, 'BERT-base'),
        (english_wikipedia_bits, 'English Wikipedia'),
        (model_bits[model_name], 'Model Size'),
        (dataset_size_bits, 'Dataset Size'),
    ]
    reference_lines.sort(key=lambda x: x[0])

    ref_colors = [plt.cm.tab10(i) for i in range(len(reference_lines))]

    for i, (bits_value, label) in enumerate(reference_lines):
        ax.axvline(x=bits_value, color=ref_colors[i], linestyle='--', linewidth=2, label=label)

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
    reference_line_labels = [label for _, label in reference_lines] + ['Pareto Frontier']
    for h, lbl in zip(handles, labels):
        if lbl in reference_line_labels:
            if lbl not in reference_labels:
                reference_handles.append(h)
                reference_labels.append(lbl)

fig.legend(reference_handles, reference_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=18)
plt.tight_layout()
plt.savefig('pareto_curves_references.png', bbox_inches='tight', dpi=300)
