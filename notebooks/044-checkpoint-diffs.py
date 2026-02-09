import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_json('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results.jsonl', lines=True)
SCRIPT_SIZE = 3704
df.loc[df['experiment_name'].isin(['urial', 'icl']), 'bits'] += SCRIPT_SIZE

model_display_names = {
    'smollm3-step40k': 'SmolLM3 (0 steps)',
    'smollm3-stage1': 'SmolLM3 Pre-trained',
    'smollm3': 'SmolLM3 Post-trained',
    'olmo3-7b-step0': 'Olmo3 (0 steps)',
    'olmo3-7b-step1414k': 'Olmo3 Pre-trained',
    'olmo3-7b-instruct-step400': 'Olmo3 Post-trained',
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

def eval_pareto_at_bits(pareto_bits, pareto_perf, query_bits, zero_bit_perf):
    result = []
    for qb in query_bits:
        perf = zero_bit_perf
        for b, p in zip(pareto_bits, pareto_perf):
            if b <= qb:
                perf = p
        result.append(perf)
    return np.array(result)

model_families = {
    'SmolLM3': {
        'zero': 'smollm3-step0',
        'pretrained': 'smollm3-stage1',
        'posttrained': 'smollm3'
    },
    'Olmo3 7B': {
        'zero': 'olmo3-7b-step0',
        'pretrained': 'olmo3-7b-step1414k',
        'posttrained': 'olmo3-7b-instruct-step400'
    }
}

datasets = ['meta-math/MetaMathQA', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl', 'allenai/nllb']

fig, axes = plt.subplots(2, 3, figsize=(24, 10))

all_legends = {}

for family_idx, (family_name, family_models) in enumerate(model_families.items()):
    zero_model = family_models['zero']
    pretrained_model = family_models['pretrained']
    posttrained_model = family_models['posttrained']

    for dataset_idx, dataset_name in enumerate(datasets):
        ax = axes[family_idx, dataset_idx]

        filtered_df = df[df['dataset_name'] == dataset_name].copy()
        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()

        pareto_data = {}
        zero_bit_data = {}

        for model_name in [zero_model, pretrained_model, posttrained_model]:
            model_df = filtered_df[filtered_df['model_name'] == model_name].copy()

            zero_bit_df = df[(df['dataset_name'] == dataset_name) &
                             (df['model_name'] == model_name) &
                             (df['bits'] == 0)]
            zero_bit_perf = zero_bit_df['performance'].iloc[0] if not zero_bit_df.empty else 0
            zero_bit_data[model_name] = zero_bit_perf

            bits, performance = compute_pareto_frontier(model_df, zero_bit_perf)
            pareto_data[model_name] = (bits, performance)

        all_bits = set()
        for model_name in [zero_model, pretrained_model, posttrained_model]:
            bits, _ = pareto_data[model_name]
            all_bits.update(bits)
        all_bits = sorted(all_bits)
        xlim_left = 1e2
        xlim_right = 1e8
        query_bits = np.logspace(np.log10(xlim_left), np.log10(xlim_right), 500)

        zero_perf = eval_pareto_at_bits(*pareto_data[zero_model], query_bits, zero_bit_data[zero_model])
        pretrained_perf = eval_pareto_at_bits(*pareto_data[pretrained_model], query_bits, zero_bit_data[pretrained_model])
        posttrained_perf = eval_pareto_at_bits(*pareto_data[posttrained_model], query_bits, zero_bit_data[posttrained_model])

        diff_pretrained = pretrained_perf - zero_perf
        diff_posttrained = posttrained_perf - pretrained_perf

        ax.plot(query_bits, diff_pretrained, linewidth=3, label='Pre-training gain', color='tab:red')
        ax.plot(query_bits, diff_posttrained, linewidth=3, label='Post-training gain', color='tab:blue')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

        ax.set_xlabel('Program Length (bits)', fontsize=18)
        ax.set_ylabel(r'$\Delta$ Performance', fontsize=18)
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)
        ax.set_title(f'{family_name} on {dataset_display}', fontsize=20)
        ax.set_xscale('log')
        ax.set_xlim(left=xlim_left, right=xlim_right)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True, alpha=0.3)

        if dataset_idx == 2:
            handles, labels = ax.get_legend_handles_labels()
            all_legends[family_name] = (handles, labels)

for family_idx, (family_name, (handles, labels)) in enumerate(all_legends.items()):
    axes[family_idx, 2].legend(handles, labels, loc='upper right', fontsize=16)

plt.tight_layout(h_pad=3.0)
plt.savefig('diffs.png', bbox_inches='tight', dpi=300)
