import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_json('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results.jsonl', lines=True)
SCRIPT_SIZE = 3704
df.loc[df['experiment_name'].isin(['urial', 'icl']), 'bits'] += SCRIPT_SIZE

dataset_name = 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl'
model_name = 'smollm3-stage3'

model_display_names = {
    'smollm': 'SmolLM2-1.7B',
    'smollm-step2750k': 'SmolLM2-1.7B (2750k steps)',
    'smollm-step4125k': 'SmolLM2-1.7B (4125k steps)',
    'smollm-step4875k': 'SmolLM2-1.7B (4875k steps)',
    'smollm3-step0': 'SmolLM3 Pre-training (0 steps)',
    'smollm3-step40k': 'SmolLM3 Pre-training (40k steps)',
    'smollm3-step1720k': 'SmolLM3 Pre-training (1720k steps)',
    'smollm3-stage1': 'SmolLM3 Pre-training (3440k steps)',
    'smollm3-stage2': 'SmolLM3 Post-training (Stage 2)',
    'smollm3-stage3': 'SmolLM3 Post-training (Stage 3)',
    'smollm3': 'SmolLM3 Post-training (Final Model)',
    'olmo3-7b-step0': 'Olmo3 Pre-training (0 steps)',
    'olmo3-7b-step707k': 'Olmo3 Pre-training (707k steps)',
    'olmo3-7b-step1414k': 'Olmo3 Pre-training (1414k steps)',
    'olmo3-7b-stage2-step6k': 'Olmo3 Mid-training (6k steps)',
    'olmo3-7b-stage2-step12k': 'Olmo3 Mid-training (12k steps)',
    'olmo3-7b-stage2-step24k': 'Olmo3 Mid-training (24k steps)',
    'olmo3-7b-stage2-step48k': 'Olmo3 Mid-training (48k steps)',
    'olmo3-1025-7b': 'Olmo3 Mid-training (Final)',
    'olmo3-7b-instruct-step400': 'Olmo3 Post-training (Final)',
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

model_families = {
    'SmolLM3': {
        'pretraining': ['smollm3-step0', 'smollm3-step1720k', 'smollm3-stage1'],
        'posttraining': ['smollm3-stage2', 'smollm3-stage3', 'smollm3']
    },
    'Olmo3 7B': {
        'pretraining': ['olmo3-7b-step0', 'olmo3-7b-step707k', 'olmo3-7b-step1414k'],
        'posttraining': ['olmo3-7b-stage2-step6k', 'olmo3-7b-stage2-step24k', 'olmo3-1025-7b', 'olmo3-7b-instruct-step400']
    }
}

datasets = ['meta-math/MetaMathQA', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl', 'allenai/nllb']

fig, axes = plt.subplots(2, 3, figsize=(24, 10))

all_legends = {}

for family_idx, (family_name, family_models) in enumerate(model_families.items()):
    pretraining_models = family_models['pretraining']
    posttraining_models = family_models['posttraining']
    models = pretraining_models + posttraining_models

    max_pretrain = max(len(m['pretraining']) for m in model_families.values())
    max_posttrain = max(len(m['posttraining']) for m in model_families.values())

    pretraining_colors = plt.cm.Reds(np.linspace(0.5, 0.8, max_pretrain))
    posttraining_colors = plt.cm.Blues(np.linspace(0.5, 0.8, max_posttrain))

    model_colors = {}
    for i, model in enumerate(pretraining_models):
        model_colors[model] = pretraining_colors[i]
    for i, model in enumerate(posttraining_models):
        model_colors[model] = posttraining_colors[i]

    for dataset_idx, dataset_name in enumerate(datasets):
        ax = axes[family_idx, dataset_idx]

        filtered_df = df[df['dataset_name'] == dataset_name].copy()
        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()

        pareto_data = {}
        zero_bit_data = {}

        for i, model_name in enumerate(models):
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
                    label=model_display, color=model_colors[model_name])

        ax.set_xlabel('Program Length (bits)', fontsize=18)
        metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
        ax.set_ylabel(metric_name, fontsize=18)
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)
        ax.set_title(f'{family_name} on {dataset_display}', fontsize=20)
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3)

        current_xlim = ax.get_xlim()
        ax.set_xlim(left=current_xlim[0], right=1e8)

        for model_name in models:
            if model_name in pareto_data:
                bits, performance = pareto_data[model_name]

                if model_name in zero_bit_data:
                    zero_bit_perf = zero_bit_data[model_name]
                    leftmost_bits = bits[0]
                    leftmost_perf = performance[0]

                    ax.plot([leftmost_bits, leftmost_bits],
                           [zero_bit_perf, leftmost_perf],
                           linewidth=2, color=model_colors[model_name], linestyle='-')

                    ax.plot([current_xlim[0], leftmost_bits],
                           [zero_bit_perf, zero_bit_perf],
                           linewidth=2, color=model_colors[model_name], linestyle='-')


                rightmost_bits = bits[-1]
                rightmost_perf = performance[-1]
                ax.plot([rightmost_bits, 1e8],
                       [rightmost_perf, rightmost_perf],
                       linewidth=2, color=model_colors[model_name], linestyle='-')

        if dataset_idx == 2:
            handles, labels = ax.get_legend_handles_labels()
            all_legends[family_name] = (handles, labels)

for family_idx, (family_name, (handles, labels)) in enumerate(all_legends.items()):
    axes[family_idx, 2].legend(handles, labels, loc='lower right', fontsize=16)

plt.tight_layout(h_pad=3.0)
plt.savefig('checkpoint_pareto_curves.png', bbox_inches='tight', dpi=300)
