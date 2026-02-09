import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D as MLine2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker


def format_bytes(bits):
    bytes_val = bits / 8
    if bytes_val >= 1e9:
        return f'{bytes_val / 1e9:.1f} GB'
    elif bytes_val >= 1e6:
        return f'{bytes_val / 1e6:.0f} MB'
    elif bytes_val >= 1e3:
        return f'{bytes_val / 1e3:.0f} KB'
    else:
        return f'{bytes_val:.0f} B'

df = pd.read_json('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results-filtered.jsonl', lines=True)
SCRIPT_SIZE = 3704
df.loc[df['experiment_name'].isin(['urial', 'icl']), 'bits'] += SCRIPT_SIZE
BASELINE_SCRIPT_SIZE = 2952
df.loc[df['experiment_name'].isin(['baseline']), 'bits'] += BASELINE_SCRIPT_SIZE
ONLINE_CODING_SCRIPT_SIZE = 5904
df.loc[df['experiment_name'].isin(['online_coding']), 'bits'] += ONLINE_CODING_SCRIPT_SIZE


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
    'meta-math/MetaMathQA': 'Accuracy (%)',
    'allenai/nllb': 'BLEU',
    'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl': 'Score (%)',
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

model_families = {
    'SmolLM3': {
        'step0': 'smollm3-step0',
        'final_pretrain': 'smollm3-stage1',
        'final_posttrain': 'smollm3'
    },
    'Olmo3 7B': {
        'step0': 'olmo3-7b-step0',
        'final_pretrain': 'olmo3-7b-step1414k',
        'final_posttrain': 'olmo3-7b-instruct-step400'
    }
}

datasets = ['meta-math/MetaMathQA', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl', 'allenai/nllb']

fig, axes = plt.subplots(3, 2, figsize=(10, 12))

stage_colors = {
    'step0': '#888888',
    'final_pretrain': '#D81B60',
    'final_posttrain': '#1E88E5'
}

stage_labels = {
    'step0': 'Randomly initialized',
    'final_pretrain': 'Pre-trained',
    'final_posttrain': 'Post-trained'
}

for dataset_idx, dataset_name in enumerate(datasets):
    for family_idx, (family_name, family_models) in enumerate(model_families.items()):
        models = [family_models['step0'], family_models['final_pretrain'], family_models['final_posttrain']]
        stages = ['step0', 'final_pretrain', 'final_posttrain']

        model_colors = {family_models[stage]: stage_colors[stage] for stage in stages}
        model_stage_labels = {family_models[stage]: stage_labels[stage] for stage in stages}

        ax = axes[dataset_idx, family_idx]

        filtered_df = df[df['dataset_name'] == dataset_name].copy()
        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
        if dataset_name in percentage_datasets:
            filtered_df['performance'] = filtered_df['performance'] * 100

        pareto_data = {}

        for i, model_name in enumerate(models):
            model_df = filtered_df[filtered_df['model_name'] == model_name].copy()

            bits, performance = compute_pareto_frontier(model_df)
            pareto_data[model_name] = (bits, performance)

            model_display = model_stage_labels.get(model_name, model_name)
            ax.plot(bits, performance,
                    linewidth=4,
                    label=model_display, color=model_colors[model_name])

        metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)
        if dataset_idx == len(datasets) - 1:
            ax.set_xlabel(f'Task Complexity\nconditioned on {family_name}', fontsize=18)
        if family_idx == 0:
            ax.set_ylabel(f'{dataset_display}\n{metric_name}', fontsize=18)
        if dataset_idx == 0:
            ax.set_title(family_name, fontsize=20, pad=15)
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=16)
        if family_idx != 0:
            ax.tick_params(axis='y', labelleft=False)
        if dataset_idx != len(datasets) - 1:
            ax.tick_params(axis='x', labelbottom=False)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(left=1e3, right=2e11)
        ax.set_ylim(bottom=0)

        tick_bits = [1e3, 1e5, 1e7, 1e9, 1e11]
        ax.set_xticks(tick_bits)
        ax.set_xticklabels([f'$10^{{{int(np.log10(b))}}}$' for b in tick_bits])

        for model_name in models:
            if model_name in pareto_data:
                bits, performance = pareto_data[model_name]
                leftmost_bits = bits[0]
                leftmost_perf = performance[0]

                ax.plot([leftmost_bits, leftmost_bits],
                       [0, leftmost_perf],
                       linewidth=4, color=model_colors[model_name], linestyle='-')

                rightmost_bits = bits[-1]
                rightmost_perf = performance[-1]
                ax.plot([rightmost_bits, 2e11],
                       [rightmost_perf, rightmost_perf],
                       linewidth=4, color=model_colors[model_name], linestyle='-')

                label_text = f'{rightmost_perf:.1f}'
                ax.text(2e11, rightmost_perf, label_text,
                       ha='left', va='center', fontsize=12,
                       color=model_colors[model_name], fontweight='bold',
                       clip_on=False)

        if dataset_idx == 0 and family_idx == 0:
            handles, labels = ax.get_legend_handles_labels()

for dataset_idx in range(len(datasets)):
    max_ylim = max(axes[dataset_idx, family_idx].get_ylim()[1] for family_idx in range(len(model_families)))
    for family_idx in range(len(model_families)):
        axes[dataset_idx, family_idx].set_ylim(bottom=0, top=max_ylim)

print("\n=== Program size to achieve best accuracy from pre-training ===\n")
for family_name, family_models in model_families.items():
    pretrain_model = family_models['final_pretrain']
    posttrain_model = family_models['final_posttrain']
    print(f"{family_name}:")
    print(f"  Pre-trained: {pretrain_model}")
    print(f"  Post-trained: {posttrain_model}")
    print()
    for dataset_name in datasets:
        filtered_df = df[df['dataset_name'] == dataset_name].copy()
        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
        if dataset_name in percentage_datasets:
            filtered_df['performance'] = filtered_df['performance'] * 100

        pretrain_df = filtered_df[filtered_df['model_name'] == pretrain_model].copy()
        posttrain_df = filtered_df[filtered_df['model_name'] == posttrain_model].copy()
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)

        if not pretrain_df.empty:
            best_pretrain_row = pretrain_df.loc[pretrain_df['performance'].idxmax()]
            best_pretrain_perf = best_pretrain_row['performance']
            best_pretrain_bits = best_pretrain_row['bits']
            print(f"  {dataset_display}:")
            print(f"    Pre-trained best: {best_pretrain_bits:.0f} bits ({format_bytes(best_pretrain_bits)}) -> {best_pretrain_perf:.1f}")

            if not posttrain_df.empty:
                posttrain_above = posttrain_df[posttrain_df['performance'] >= best_pretrain_perf]
                if not posttrain_above.empty:
                    min_bits_row = posttrain_above.loc[posttrain_above['bits'].idxmin()]
                    print(f"    Post-trained to match: {min_bits_row['bits']:.0f} bits ({format_bytes(min_bits_row['bits'])}) -> {min_bits_row['performance']:.1f}")
                    print(f"    Reduction: {best_pretrain_bits / min_bits_row['bits']:.1f}x")
                else:
                    print(f"    Post-trained: cannot reach {best_pretrain_perf:.1f} performance")
    print()

fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), fontsize=14, ncol=3)
plt.tight_layout()
plt.savefig('task_complexity.pdf', bbox_inches='tight', dpi=300)

datasets_no_flores = ['meta-math/MetaMathQA', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl']

fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))

for dataset_idx, dataset_name in enumerate(datasets_no_flores):
    for family_idx, (family_name, family_models) in enumerate(model_families.items()):
        models = [family_models['step0'], family_models['final_pretrain'], family_models['final_posttrain']]
        stages = ['step0', 'final_pretrain', 'final_posttrain']

        model_colors = {family_models[stage]: stage_colors[stage] for stage in stages}
        model_stage_labels = {family_models[stage]: stage_labels[stage] for stage in stages}

        ax = axes2[dataset_idx, family_idx]

        filtered_df = df[df['dataset_name'] == dataset_name].copy()
        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
        if dataset_name in percentage_datasets:
            filtered_df['performance'] = filtered_df['performance'] * 100

        pareto_data = {}

        for i, model_name in enumerate(models):
            model_df = filtered_df[filtered_df['model_name'] == model_name].copy()

            bits, performance = compute_pareto_frontier(model_df)
            pareto_data[model_name] = (bits, performance)

            model_display = model_stage_labels.get(model_name, model_name)
            ax.plot(bits, performance,
                    linewidth=4,
                    label=model_display, color=model_colors[model_name])

        metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)
        if dataset_idx == len(datasets_no_flores) - 1:
            ax.set_xlabel(f'Task Complexity\nconditioned on {family_name}', fontsize=18)
        if family_idx == 0:
            ax.set_ylabel(f'{dataset_display}\n{metric_name}', fontsize=18)
        if dataset_idx == 0:
            ax.set_title(family_name, fontsize=20, pad=15)
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=16)
        if family_idx != 0:
            ax.tick_params(axis='y', labelleft=False)
        if dataset_idx != len(datasets_no_flores) - 1:
            ax.tick_params(axis='x', labelbottom=False)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(left=1e3, right=2e11)
        ax.set_ylim(bottom=0)

        tick_bits = [1e3, 1e5, 1e7, 1e9, 1e11]
        ax.set_xticks(tick_bits)
        ax.set_xticklabels([f'$10^{{{int(np.log10(b))}}}$' for b in tick_bits])

        for model_name in models:
            if model_name in pareto_data:
                bits, performance = pareto_data[model_name]
                leftmost_bits = bits[0]
                leftmost_perf = performance[0]

                ax.plot([leftmost_bits, leftmost_bits],
                       [0, leftmost_perf],
                       linewidth=4, color=model_colors[model_name], linestyle='-')

                rightmost_bits = bits[-1]
                rightmost_perf = performance[-1]
                ax.plot([rightmost_bits, 2e11],
                       [rightmost_perf, rightmost_perf],
                       linewidth=4, color=model_colors[model_name], linestyle='-')

                label_text = f'{rightmost_perf:.1f}'
                ax.text(2e11, rightmost_perf, label_text,
                       ha='left', va='center', fontsize=12,
                       color=model_colors[model_name], fontweight='bold',
                       clip_on=False)

        if dataset_idx == 0 and family_idx == 0:
            handles2, labels2 = ax.get_legend_handles_labels()

for dataset_idx in range(len(datasets_no_flores)):
    max_ylim = max(axes2[dataset_idx, family_idx].get_ylim()[1] for family_idx in range(len(model_families)))
    for family_idx in range(len(model_families)):
        axes2[dataset_idx, family_idx].set_ylim(bottom=0, top=max_ylim)

fig2.legend(handles2, labels2, loc='upper center', bbox_to_anchor=(0.5, -0.02), fontsize=14, ncol=3)
plt.tight_layout()
plt.savefig('task_complexity_no_flores.pdf', bbox_inches='tight', dpi=300)

datasets_flores_only = ['allenai/nllb']

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))

for dataset_idx, dataset_name in enumerate(datasets_flores_only):
    for family_idx, (family_name, family_models) in enumerate(model_families.items()):
        models = [family_models['step0'], family_models['final_pretrain'], family_models['final_posttrain']]
        stages = ['step0', 'final_pretrain', 'final_posttrain']

        model_colors = {family_models[stage]: stage_colors[stage] for stage in stages}
        model_stage_labels = {family_models[stage]: stage_labels[stage] for stage in stages}

        ax = axes3[family_idx]

        filtered_df = df[df['dataset_name'] == dataset_name].copy()
        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()

        pareto_data = {}

        for i, model_name in enumerate(models):
            model_df = filtered_df[filtered_df['model_name'] == model_name].copy()

            bits, performance = compute_pareto_frontier(model_df)
            pareto_data[model_name] = (bits, performance)

            model_display = model_stage_labels.get(model_name, model_name)
            ax.plot(bits, performance,
                    linewidth=4,
                    label=model_display, color=model_colors[model_name])

        metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)
        ax.set_xlabel(f'Task Complexity\nconditioned on {family_name}', fontsize=18)
        if family_idx == 0:
            ax.set_ylabel(f'{dataset_display}\n{metric_name}', fontsize=18)
        ax.set_title(family_name, fontsize=20, pad=15)
        ax.set_xscale('log')
        ax.tick_params(axis='both', labelsize=16)
        if family_idx != 0:
            ax.tick_params(axis='y', labelleft=False)
        ax.grid(True, alpha=0.3)

        ax.set_xlim(left=1e3, right=2e11)
        ax.set_ylim(bottom=0)

        tick_bits = [1e3, 1e5, 1e7, 1e9, 1e11]
        ax.set_xticks(tick_bits)
        ax.set_xticklabels([f'$10^{{{int(np.log10(b))}}}$' for b in tick_bits])

        label_positions = []
        for model_name in models:
            if model_name in pareto_data:
                bits, performance = pareto_data[model_name]
                leftmost_bits = bits[0]
                leftmost_perf = performance[0]

                ax.plot([leftmost_bits, leftmost_bits],
                       [0, leftmost_perf],
                       linewidth=4, color=model_colors[model_name], linestyle='-')

                rightmost_bits = bits[-1]
                rightmost_perf = performance[-1]
                ax.plot([rightmost_bits, 2e11],
                       [rightmost_perf, rightmost_perf],
                       linewidth=4, color=model_colors[model_name], linestyle='-')

                label_positions.append((rightmost_perf, model_name, model_colors[model_name]))

        label_positions.sort(key=lambda x: x[0])
        min_spacing = 2.5
        adjusted_positions = []
        for i, (perf, model_name, color) in enumerate(label_positions):
            adjusted_perf = perf
            if i > 0 and adjusted_perf - adjusted_positions[-1] < min_spacing:
                adjusted_perf = adjusted_positions[-1] + min_spacing
            adjusted_positions.append(adjusted_perf)
            label_text = f'{perf:.1f}'
            ax.text(2e11, adjusted_perf, label_text,
                   ha='left', va='center', fontsize=12,
                   color=color, fontweight='bold',
                   clip_on=False)

        if family_idx == 0:
            handles3, labels3 = ax.get_legend_handles_labels()

max_ylim = max(axes3[family_idx].get_ylim()[1] for family_idx in range(len(model_families)))
for family_idx in range(len(model_families)):
    axes3[family_idx].set_ylim(bottom=0, top=max_ylim)

fig3.legend(handles3, labels3, loc='upper center', bbox_to_anchor=(0.5, -0.02), fontsize=14, ncol=3)
plt.tight_layout()
plt.savefig('task_complexity_flores.pdf', bbox_inches='tight', dpi=300)


fig_single, ax_single = plt.subplots(figsize=(10, 6))

family_name = 'Olmo3 7B'
family_models = model_families[family_name]
dataset_name = 'meta-math/MetaMathQA'

models = [family_models['step0'], family_models['final_pretrain'], family_models['final_posttrain']]
stages = ['step0', 'final_pretrain', 'final_posttrain']

model_colors_single = {family_models[stage]: stage_colors[stage] for stage in stages}
model_stage_labels = {family_models[stage]: stage_labels[stage] for stage in stages}

filtered_df = df[df['dataset_name'] == dataset_name].copy()
filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
filtered_df['performance'] = filtered_df['performance'] * 100

pareto_data = {}

for model_name in models:
    model_df = filtered_df[filtered_df['model_name'] == model_name].copy()
    bits, performance = compute_pareto_frontier(model_df)
    pareto_data[model_name] = (bits, performance)
    ax_single.plot(bits, performance, linewidth=6, color=model_colors_single[model_name])

for model_name in models:
    if model_name in pareto_data:
        bits, performance = pareto_data[model_name]
        leftmost_bits = bits[0]
        leftmost_perf = performance[0]
        ax_single.plot([leftmost_bits, leftmost_bits], [0, leftmost_perf],
                       linewidth=6, color=model_colors_single[model_name], linestyle='-')
        rightmost_bits = bits[-1]
        rightmost_perf = performance[-1]
        ax_single.plot([rightmost_bits, 8e10], [rightmost_perf, rightmost_perf],
                       linewidth=6, color=model_colors_single[model_name], linestyle='-')

ax_single.set_xlabel('Program Length (bits)', fontsize=28)
ax_single.set_ylabel('Accuracy (%)', fontsize=28)
ax_single.set_xscale('log')
ax_single.tick_params(axis='both', labelsize=24)
ax_single.grid(True, alpha=0.3)
ax_single.set_xlim(left=1e3, right=8e10)
ax_single.set_ylim(bottom=0)

tick_bits = [1e3, 1e5, 1e7, 1e9, 1e11]
ax_single.set_xticks(tick_bits)
ax_single.set_xticklabels([f'$10^{{{int(np.log10(b))}}}$' for b in tick_bits])

stage_legend_labels = {
    'step0': ('Randomly Initialized', ''),
    'final_pretrain': ('Pre-trained', '(1.4M steps)'),
    'final_posttrain': ('Post-trained', '(Instruct model)'),
}

legend_artists = []
for stage in stages:
    model_name = family_models[stage]
    color = model_colors_single[model_name]
    main_label, sub_label = stage_legend_labels[stage]
    line_handle = plt.Line2D([0], [0], color=color, linewidth=6)
    text_main = TextArea(main_label, textprops=dict(fontsize=18))
    if sub_label:
        text_sub = TextArea(sub_label, textprops=dict(fontsize=14, color='gray'))
        text_box = VPacker(children=[text_main, text_sub], align='left', pad=0, sep=1)
    else:
        text_box = text_main
    legend_artists.append((line_handle, text_box))

legend_box_children = []
for (line_handle, text_box) in legend_artists:
    da = DrawingArea(30, 20, 0, 0)
    line = MLine2D([0, 25], [10, 10], color=line_handle.get_color(), linewidth=6)
    da.add_artist(line)
    row = HPacker(children=[da, text_box], align='center', pad=0, sep=8)
    legend_box_children.append(row)

legend_vbox = VPacker(children=legend_box_children, align='left', pad=5, sep=12)
anchored_box = AnchoredOffsetbox(loc='lower right', child=legend_vbox, pad=0.5,
                                  frameon=True, bbox_to_anchor=(1, 0),
                                  bbox_transform=ax_single.transAxes, borderpad=0.5)
anchored_box.patch.set_facecolor('white')
anchored_box.patch.set_edgecolor('lightgray')
anchored_box.patch.set_linewidth(1)
ax_single.add_artist(anchored_box)

plt.tight_layout()
plt.savefig('task_complexity_olmo3_gsm8k.pdf', bbox_inches='tight', dpi=300)

fig_single.patch.set_facecolor('none')
ax_single.set_facecolor('none')
anchored_box.patch.set_facecolor('none')
anchored_box.patch.set_edgecolor('none')
plt.savefig('task_complexity_olmo3_gsm8k_transparent.svg', bbox_inches='tight', transparent=True)
