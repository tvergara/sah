import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D as MLine2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker


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
ONLINE_CODING_SCRIPT_SIZE = 5704
df.loc[df['experiment_name'].isin(['online_coding']), 'bits'] += ONLINE_CODING_SCRIPT_SIZE
LORA_SCRIPT_SIZE = 2832
df.loc[df['experiment_name'].isin(['lora']), 'bits'] += LORA_SCRIPT_SIZE
BLORA_SCRIPT_SIZE = 8376
df.loc[df['experiment_name'].isin(['blora']), 'bits'] += BLORA_SCRIPT_SIZE

model_display_names = {
    'smollm3-stage1': 'SmolLM3',
    'olmo3-7b-step1414k': 'Olmo3 7B',
    'olmo3-32b-step656k': 'Olmo3 32B',
}

dataset_display_names = {
    'meta-math/MetaMathQA': 'Math',
    'allenai/nllb': 'Translation',
    'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl': 'Instruction',
}

dataset_metric_names = {
    'meta-math/MetaMathQA': 'Accuracy (%)',
    'allenai/nllb': 'BLEU',
    'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl': 'Score (%)',
}

percentage_datasets = {'meta-math/MetaMathQA', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl'}

method_view = {
    'icl': 'Inference-Control',
    'lora': 'Parametric',
    'adam': 'Parametric',
    'online_coding': 'Data',
    'urial': 'Inference-Control',
    'blora': 'Parametric',
    'baseline': 'Inference-Control',
}

view_colors = {
    'Data': '#1E88E5',
    'Parametric': '#FFC107',
    'Inference-Control': '#D81B60',
}

def compute_pareto_frontier(model_df):
    df_sorted = model_df.sort_values('bits').copy()
    pareto_points = []
    max_performance = -np.inf

    for idx, row in df_sorted.iterrows():
        if row['performance'] >= max_performance:
            pareto_points.append({
                'bits': row['bits'],
                'performance': row['performance'],
                'experiment_name': row['experiment_name']
            })
            max_performance = row['performance']

    return pareto_points

models = ['smollm3-stage1', 'olmo3-7b-step1414k', 'olmo3-32b-step656k']
datasets = ['meta-math/MetaMathQA', 'allenai/nllb', 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl']

fig, axes = plt.subplots(3, 3, figsize=(14, 9))

for dataset_idx, dataset_name in enumerate(datasets):
    for model_idx, model_name in enumerate(models):
        ax = axes[dataset_idx, model_idx]

        filtered_df = df[df['dataset_name'] == dataset_name]
        filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

        filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
        filtered_df = filtered_df[filtered_df['experiment_name'].isin(method_view.keys())].copy()
        if dataset_name in percentage_datasets:
            filtered_df['performance'] = filtered_df['performance'] * 100

        pareto_points = compute_pareto_frontier(filtered_df)

        if len(pareto_points) > 0:
            for i, point in enumerate(pareto_points):
                prev_perf = pareto_points[i - 1]['performance'] if i > 0 else point['performance']
                view = method_view[point['experiment_name']]
                color = view_colors[view]

                ax.plot([point['bits'], point['bits']],
                        [prev_perf, point['performance']],
                        linewidth=4, color=color)

                if i < len(pareto_points) - 1:
                    next_bits = pareto_points[i + 1]['bits']
                else:
                    next_bits = 1e12
                ax.plot([point['bits'], next_bits],
                        [point['performance'], point['performance']],
                        linewidth=4, color=color)

        if dataset_idx == len(datasets) - 1:
            ax.set_xlabel('Program Length (bits)', fontsize=18)
        metric_name = dataset_metric_names.get(dataset_name, r'$\delta$')
        dataset_display = dataset_display_names.get(dataset_name, dataset_name)
        if model_idx == 0:
            ax.set_ylabel(f'{dataset_display}\n{metric_name}', fontsize=20, rotation=0, ha='center', va='center', labelpad=70)
        ax.set_xscale('log')
        if dataset_idx == 0:
            model_display = model_display_names.get(model_name, model_name)
            ax.set_title(model_display, fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(True)

        ax.set_xlim(left=BASELINE_SCRIPT_SIZE // 2, right=1e12)

        tick_bytes = [1e2, 1e4, 1e6, 1e8, 1e10]
        tick_bits = [b * 8 for b in tick_bytes]
        ax.set_xticks(tick_bits)
        ax.set_xticklabels([format_bytes(b) for b in tick_bits])

for dataset_idx in range(len(datasets)):
    row_axes = [axes[dataset_idx, model_idx] for model_idx in range(len(models))]
    y_mins = [ax.get_ylim()[0] for ax in row_axes]
    y_maxs = [ax.get_ylim()[1] for ax in row_axes]
    shared_ylim = (min(y_mins), max(y_maxs))
    for ax in row_axes:
        ax.set_ylim(shared_ylim)

legend_handles = [plt.Line2D([0], [0], color=color, linewidth=4, label=view)
                  for view, color in view_colors.items()]
fig.legend(legend_handles, view_colors.keys(), loc='center left',
           bbox_to_anchor=(1.01, 0.5), fontsize=14, title='Views', title_fontsize=16)
plt.tight_layout()
plt.savefig('pareto_curves_by_view.pdf', bbox_inches='tight', dpi=300)

fig_single, ax_single = plt.subplots(figsize=(10, 6))

model_name = 'olmo3-7b-step1414k'
dataset_name = 'meta-math/MetaMathQA'

filtered_df = df[df['dataset_name'] == dataset_name]
filtered_df = filtered_df[filtered_df['model_name'] == model_name].copy()

filtered_df = filtered_df[filtered_df['bits'] > 0].copy()
filtered_df = filtered_df[filtered_df['experiment_name'].isin(method_view.keys())].copy()
filtered_df['performance'] = filtered_df['performance'] * 100

pareto_points = compute_pareto_frontier(filtered_df)

if len(pareto_points) > 0:
    for i, point in enumerate(pareto_points):
        prev_perf = pareto_points[i - 1]['performance'] if i > 0 else point['performance']
        view = method_view[point['experiment_name']]
        color = view_colors[view]

        ax_single.plot([point['bits'], point['bits']],
                       [prev_perf, point['performance']],
                       linewidth=6, color=color)

        if i < len(pareto_points) - 1:
            next_bits = pareto_points[i + 1]['bits']
        else:
            next_bits = 8e10
        ax_single.plot([point['bits'], next_bits],
                       [point['performance'], point['performance']],
                       linewidth=6, color=color)

ax_single.set_xlabel('Program Length (bits)', fontsize=28)
ax_single.set_ylabel('Accuracy (%)', fontsize=28)
ax_single.set_xscale('log')
ax_single.tick_params(axis='both', labelsize=24)
ax_single.grid(True)
ax_single.set_xlim(left=BASELINE_SCRIPT_SIZE // 2, right=8e10)

tick_bits = [1e3, 1e5, 1e7, 1e9, 1e11]
ax_single.set_xticks(tick_bits)
ax_single.set_xticklabels([f'$10^{{{int(np.log10(b))}}}$' for b in tick_bits])

view_legend_labels = {
    'Data': ('Data View', '(Finetuning on little data)'),
    'Parametric': ('Parametric View', '(PEFT)'),
    'Inference-Control': ('Inference-Control View', '(Well-designed Prompts)'),
}

legend_handles = []
legend_artists = []
for view, color in view_colors.items():
    main_label, sub_label = view_legend_labels[view]
    line_handle = plt.Line2D([0], [0], color=color, linewidth=6)
    text_main = TextArea(main_label, textprops=dict(fontsize=18))
    text_sub = TextArea(sub_label, textprops=dict(fontsize=14, color='gray'))
    text_box = VPacker(children=[text_main, text_sub], align='left', pad=0, sep=1)
    legend_artists.append((line_handle, text_box))
    legend_handles.append(line_handle)

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
plt.savefig('pareto_curves_by_view_single.pdf', bbox_inches='tight', dpi=300)

fig_single.patch.set_facecolor('none')
ax_single.set_facecolor('none')
anchored_box.patch.set_facecolor('none')
anchored_box.patch.set_edgecolor('none')
plt.savefig('pareto_curves_by_view_single_transparent.svg', bbox_inches='tight', transparent=True)
