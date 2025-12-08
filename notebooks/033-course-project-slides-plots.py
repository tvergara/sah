import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_json('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/results-backup-full.csv', lines=True)

model_display_names = {
    'smollm': 'SmolLM2-1.7B',
    'qwen': 'Qwen2.5-1.5B',
    'smollm3': 'SmolLM3-3B',
    'smollm3-stage1': 'SmolLM3 (Pre-trained)',
    'olmo3-7b-step1414k': 'Olmo3 (Pre-trained)',
    'olmo3-7b-step0': 'Olmo3 Pre-training (0 steps)',
    'olmo3-7b-step707k': 'Olmo3 Pre-training (707k steps)',
    'olmo3-7b-stage2-step6k': 'Olmo3 Post-training (6k steps)',
    'olmo3-7b-stage2-step12k': 'Olmo3 Post-training (12k steps)',
    'olmo3-7b-stage2-step24k': 'Olmo3 Post-training (24k steps)',
    'olmo3-7b-stage2-step48k': 'Olmo3 Post-training (48k steps)',
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
    'online_coding': 'Online Coding',
    'full_ft': 'Full Fine-tuning',
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

models = ['olmo3-7b-step1414k']
datasets = ['meta-math/MetaMathQA', 'allenai/nllb']

model_20_params_bits = 20 * 32
tweet_bits = 280 * 8
imagenet_image_bits = 224 * 224 * 3 * 8
bert_base_bits = 110_000_000 * 32
python_100_lines_bits = 100 * 50 * 8
english_wikipedia_bits = 80 * 1024 * 1024 * 1024 * 8

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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

    ax.set_xlabel('Message Length', fontsize=14)
    metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
    ax.set_ylabel(metric_name, fontsize=14)
    model_display = model_display_names.get(model_name, model_name)
    dataset_display = dataset_display_names.get(dataset_name, dataset_name)
    ax.set_title(f'{model_display} on {dataset_display}', fontsize=16)
    ax.set_xscale('log')
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

    if idx == 0:
        handles, labels = ax.get_legend_handles_labels()
        all_handles = handles
        all_labels = labels

fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=12)
plt.tight_layout()
plt.savefig('pareto_curves_all.png', bbox_inches='tight', dpi=300)

olmo_pretraining_models = ['olmo3-7b-step0', 'olmo3-7b-step707k', 'olmo3-7b-step1414k']
olmo_posttraining_models = ['olmo3-7b-stage2-step6k', 'olmo3-7b-stage2-step12k', 'olmo3-7b-stage2-step24k', 'olmo3-7b-stage2-step48k']
olmo_all_models = olmo_pretraining_models + olmo_posttraining_models

max_pretrain = len(olmo_pretraining_models)
max_posttrain = len(olmo_posttraining_models)

pretraining_colors = plt.cm.Reds(np.linspace(0.5, 0.8, max_pretrain))
posttraining_colors = plt.cm.Blues(np.linspace(0.5, 0.8, max_posttrain))

olmo_model_colors = {}
for i, model in enumerate(olmo_pretraining_models):
    olmo_model_colors[model] = pretraining_colors[i]
for i, model in enumerate(olmo_posttraining_models):
    olmo_model_colors[model] = posttraining_colors[i]

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

for dataset_idx, dataset_name in enumerate(datasets):
    ax = axes2[dataset_idx]

    filtered_df = df[df['dataset_name'] == dataset_name].copy()
    filtered_df = filtered_df[filtered_df['bits'] > 0].copy()

    pareto_data = {}
    zero_bit_data = {}

    for i, model_name in enumerate(olmo_all_models):
        model_df = filtered_df[filtered_df['model_name'] == model_name].copy()

        zero_bit_df = df[(df['dataset_name'] == dataset_name) &
                         (df['model_name'] == model_name) &
                         (df['bits'] == 0)]
        zero_bit_perf = zero_bit_df['performance'].iloc[0] if not zero_bit_df.empty else None
        if zero_bit_perf is not None:
            zero_bit_data[model_name] = zero_bit_perf

        bits, performance = compute_pareto_frontier(model_df, zero_bit_perf)
        pareto_data[model_name] = (bits, performance)

        model_display = model_display_names.get(model_name, model_name)
        ax.plot(bits, performance,
                linewidth=2,
                label=model_display, color=olmo_model_colors[model_name])

    ax.set_xlabel('Message Length', fontsize=14)
    metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
    ax.set_ylabel(metric_name, fontsize=14)
    dataset_display = dataset_display_names.get(dataset_name, dataset_name)
    ax.set_title(f'{dataset_display} - Olmo3 Checkpoints', fontsize=16)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    current_xlim = ax.get_xlim()
    ax.set_xlim(left=current_xlim[0], right=1e8)

    for model_name in olmo_all_models:
        if model_name in pareto_data:
            bits, performance = pareto_data[model_name]

            if model_name in zero_bit_data:
                zero_bit_perf = zero_bit_data[model_name]
                leftmost_bits = bits[0]
                leftmost_perf = performance[0]

                ax.plot([leftmost_bits, leftmost_bits],
                       [zero_bit_perf, leftmost_perf],
                       linewidth=2, color=olmo_model_colors[model_name], linestyle='-')

                ax.plot([current_xlim[0], leftmost_bits],
                       [zero_bit_perf, zero_bit_perf],
                       linewidth=2, color=olmo_model_colors[model_name], linestyle='-')

            rightmost_bits = bits[-1]
            rightmost_perf = performance[-1]
            ax.plot([rightmost_bits, 1e8],
                   [rightmost_perf, rightmost_perf],
                   linewidth=2, color=olmo_model_colors[model_name], linestyle='-')

    if dataset_idx == 1:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('checkpoint_pareto_curves_olmo.png', bbox_inches='tight', dpi=300)
