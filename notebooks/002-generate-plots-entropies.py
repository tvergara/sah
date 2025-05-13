import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('~/scratch/hydra-runs/alpaca/entropies.csv')

df_entropy = df[df['input'].isna()]

df_entropy

df_sorted = df_entropy.sort_values('output')

order = [
    'step-160000',
    'step-320000',
    'step-480000',
    'step-640000',
    'step-800000',
    'step-960000',
    'step-1120000',
    'step-1280000',
    'step_00050',
    'step_00100',
    'step_00150',
    'step_00200',
]

labels = [
    '(pretraining) 160000',
    '(pretraining) 320000',
    '(pretraining) 480000',
    '(pretraining) 640000',
    '(pretraining) 800000',
    '(pretraining) 960000',
    '(pretraining) 1120000',
    '(pretraining) 1280000',
    '(finetuning) 00050',
    '(finetuning) 00100',
    '(finetuning) 00150',
    '(finetuning) 00200',
]
df_ordered = df_entropy.set_index('output').reindex(order).reset_index()

plt.figure()
plt.plot(labels, df_ordered['entropy'])
plt.xticks(rotation=45, ha='right')
plt.ylabel('entropy')
plt.title('Entropy')
plt.tight_layout()
plt.savefig('entropies')


n = len(order)

predictive_infos = []
predictive_labels = []
for i in range(1, n):
    mask = (df['input'] == order[i - 1]) & (df['output'] == order[i])
    cross_entropy = df[mask]['entropy'].item()

    mask = (df['input'].isna()) & (df['output'] == order[i])
    entropy = df[mask]['entropy'].item()

    predictive_info = entropy - cross_entropy
    predictive_infos.append(predictive_info)
    predictive_labels.append(labels[i])

predictive_infos

plt.figure()
plt.plot(predictive_labels, predictive_infos)
plt.xticks(rotation=45, ha='right')
plt.title('predictive info')
plt.tight_layout()
plt.savefig('predictive_info')

back_predictive_infos = []
back_predictive_labels = []
for i in range(1, n):
    mask = (df['input'] == order[i]) & (df['output'] == order[i - 1])
    cross_entropy = df[mask]['entropy'].item()

    mask = (df['input'].isna()) & (df['output'] == order[i - 1])
    entropy = df[mask]['entropy'].item()

    predictive_info = entropy - cross_entropy
    back_predictive_infos.append(predictive_info)
    back_predictive_labels.append(labels[i-1])


plt.figure()
plt.plot(back_predictive_labels, back_predictive_infos)
plt.xticks(rotation=45, ha='right')
plt.title('back predictive info')
plt.tight_layout()
plt.savefig('back_predictive_info')

info_ratio = []
labels_ratio = []
for i in range(1, n):
    mask = (df['input'] == order[i]) & (df['output'] == order[i - 1])
    cross_entropy = df[mask]['entropy'].item()

    mask = (df['input'].isna()) & (df['output'] == order[i - 1])
    entropy = df[mask]['entropy'].item()

    back_predictive_info = entropy - cross_entropy

    mask = (df['input'] == order[i -1]) & (df['output'] == order[i])
    cross_entropy = df[mask]['entropy'].item()

    mask = (df['input'].isna()) & (df['output'] == order[i])
    entropy = df[mask]['entropy'].item()

    forwards_predictive_info = entropy - cross_entropy

    info_ratio.append(forwards_predictive_info / back_predictive_info)
    labels_ratio.append(labels[i-1] + labels[i])


plt.figure()
plt.plot(labels_ratio, info_ratio)
plt.xticks(rotation=45, ha='right')
plt.title('ratio predictive info')
plt.tight_layout()
plt.savefig('ratio_predictive_info')


pretraining_ckps = order[:8]
ft_ckps = order[8:]

plt.figure()
for pretraining in pretraining_ckps:
    values = []

    for ft in ft_ckps:
        mask = (df['input'] == pretraining) & (df['output'] == ft)
        cross_entropy = df[mask]['entropy'].item()

        mask = (df['input'].isna()) & (df['output'] == ft)
        entropy = df[mask]['entropy'].item()

        predictive_info = entropy - cross_entropy
        values.append(predictive_info)

    plt.plot(values, label=pretraining)

plt.tight_layout()
plt.legend()
