import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def format_bytes(bits):
    bytes_val = bits / 8
    if bytes_val >= 1e9:
        return f'{bytes_val / 1e9:.1f} GB'
    elif bytes_val >= 1e6:
        return f'{bytes_val / 1e6:.1f} MB'
    elif bytes_val >= 1e3:
        return f'{bytes_val / 1e3:.1f} KB'
    else:
        return f'{bytes_val:.0f} B'

df = pd.read_json('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results.jsonl', lines=True)
SCRIPT_SIZE = 3704
df.loc[df['experiment_name'].isin(['urial', 'icl']), 'bits'] += SCRIPT_SIZE
BASELINE_SCRIPT_SIZE = 2952
df.loc[df['experiment_name'].isin(['baseline']), 'bits'] += BASELINE_SCRIPT_SIZE

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

model_families = {
    'SmolLM3': {
        'pretrained': 'smollm3-stage1',
        'posttrained': 'smollm3'
    },
    'Olmo3 7B': {
        'pretrained': 'olmo3-7b-step1414k',
        'posttrained': 'olmo3-7b-instruct-step400'
    }
}

datasets = ['meta-math/MetaMathQA', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl', 'allenai/nllb']
model_family_names = list(model_families.keys())

results = {}

for family_name, family_models in model_families.items():
    pretrained_model = family_models['pretrained']
    posttrained_model = family_models['posttrained']

    for dataset_name in datasets:
        pretrained_df = df[(df['model_name'] == pretrained_model) & (df['dataset_name'] == dataset_name)]
        posttrained_df = df[(df['model_name'] == posttrained_model) & (df['dataset_name'] == dataset_name)]

        max_pretrained = pretrained_df['performance'].max()
        max_posttrained = posttrained_df['performance'].max()

        results[(family_name, dataset_name)] = {
            'max_pretrained': max_pretrained,
            'max_posttrained': max_posttrained,
            'delta': max_posttrained - max_pretrained,
        }

fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for row_idx, dataset_name in enumerate(datasets):
    row_max = max(results[(fn, dataset_name)]['max_posttrained'] for fn in model_family_names)

    for col_idx, family_name in enumerate(model_family_names):
        ax = axes[row_idx, col_idx]

        data = results[(family_name, dataset_name)]
        max_pretrained = data['max_pretrained']
        delta = data['delta']

        ax.bar(0, max_pretrained, width=0.6, label='Pre-trained', color='tab:red', alpha=0.8)
        ax.bar(0, delta, width=0.6, bottom=max_pretrained, label='Post-training gain', color='tab:blue', alpha=0.8)

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylim(0, row_max * 1.15)

        for spine in ax.spines.values():
            spine.set_visible(False)

        metric_name = dataset_metric_names.get(dataset_name, 'Performance')
        if col_idx == 0:
            dataset_display = dataset_display_names.get(dataset_name, dataset_name)
            ax.set_ylabel(f'{dataset_display}\n({metric_name})', fontsize=14)
        else:
            ax.set_yticklabels([])

        if row_idx == 0:
            ax.set_title(f'{family_name}', fontsize=16)

        if row_idx == 0 and col_idx == 1:
            ax.legend(fontsize=12, loc='lower right')

        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        ax.text(0, max_pretrained / 2, f'{max_pretrained:.2f}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        if delta > 0.01:
            ax.text(0, max_pretrained + delta / 2, f'+{delta:.2f}', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('task_complexity_reduction.png', bbox_inches='tight', dpi=300)

fig2, axes2 = plt.subplots(3, 2, figsize=(10, 12))

bits_results = {}

for family_name, family_models in model_families.items():
    pretrained_model = family_models['pretrained']
    posttrained_model = family_models['posttrained']

    for dataset_name in datasets:
        pretrained_df = df[(df['model_name'] == pretrained_model) & (df['dataset_name'] == dataset_name)]
        posttrained_df = df[(df['model_name'] == posttrained_model) & (df['dataset_name'] == dataset_name)]

        target_perf = pretrained_df['performance'].max()

        pretrained_above = pretrained_df[pretrained_df['performance'] >= target_perf]
        min_bits_pretrained = pretrained_above['bits'].min()

        posttrained_above = posttrained_df[posttrained_df['performance'] >= target_perf]
        min_bits_posttrained = posttrained_above['bits'].min() if not posttrained_above.empty else np.nan

        bits_results[(family_name, dataset_name)] = {
            'target_perf': target_perf,
            'min_bits_pretrained': min_bits_pretrained,
            'min_bits_posttrained': min_bits_posttrained,
        }

for row_idx, dataset_name in enumerate(datasets):
    for col_idx, family_name in enumerate(model_family_names):
        ax = axes2[row_idx, col_idx]

        data = bits_results[(family_name, dataset_name)]
        min_bits_pretrained = data['min_bits_pretrained']
        min_bits_posttrained = data['min_bits_posttrained']

        log_baseline = np.log10(BASELINE_SCRIPT_SIZE / 2)
        log_pretrained = np.log10(min_bits_pretrained) if not np.isnan(min_bits_pretrained) else log_baseline
        log_posttrained = np.log10(min_bits_posttrained) if not np.isnan(min_bits_posttrained) else log_baseline

        x = np.array([0, 1])
        bars = ax.bar(x, [log_pretrained - log_baseline, log_posttrained - log_baseline], width=0.6,
                      bottom=log_baseline, color=['tab:red', 'tab:blue'], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(['Pre-trained', 'Post-trained'], fontsize=11)
        ax.set_ylim(bottom=log_baseline, top=np.log10(1e12))

        for spine in ax.spines.values():
            spine.set_visible(False)

        if col_idx == 0:
            dataset_display = dataset_display_names.get(dataset_name, dataset_name)
            ax.set_ylabel(f'{dataset_display}\n(Program Size)', fontsize=14)

        if row_idx == 0:
            ax.set_title(f'{family_name}', fontsize=16)

        tick_bits = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
        tick_positions = [np.log10(b) for b in tick_bits if log_baseline <= np.log10(b) <= np.log10(1e12)]
        tick_labels = [format_bytes(b) for b in tick_bits if log_baseline <= np.log10(b) <= np.log10(1e12)]
        ax.set_yticks(tick_positions)
        if col_idx == 0:
            ax.set_yticklabels(tick_labels)
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val, log_val in zip(bars, [min_bits_pretrained, min_bits_posttrained], [log_pretrained, log_posttrained]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, log_val + 0.2, format_bytes(val), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('task_complexity_reduction_bits.png', bbox_inches='tight', dpi=300)
