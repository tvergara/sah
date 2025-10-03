import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/results.csv')
df = df.dropna()
df = df[df['gsm8k'] > 0.6]

model_bits = df[df['experiment_name'] == 'adam']['bits'].item()

df['percentage_storage'] = df['bits'] / model_bits * 100
df

label_mapping = {
    'lora-1': 'LoRA (1)',
    'lora-4': 'LoRA (4)',
    'lora-8': 'LoRA (8)',
    'compressed-6400': 'Ours',
    'adam': 'Sending\nthe FT',
    'dataset': 'Dataset',
}

plt.figure(figsize=(10, 6))
plt.scatter(df['percentage_storage'], df['gsm8k'], s=100)
plt.xscale('log')
plt.xlim(10 ** (-3.5), None)
plt.xlabel('Percentage of storage', fontsize=14)
plt.ylabel('GSM8K', fontsize=14)
plt.ylim(0, 1)
plt.title('GSM8K Performance vs Message length', fontsize=16)
plt.tick_params(axis='both', labelsize=12)
plt.grid(True, alpha=0.3)

for i, txt in enumerate(df['experiment_name']):
    label = label_mapping.get(txt, txt)  # Use mapping if exists, otherwise use original name
    if txt == 'lora-8':
        plt.annotate(label, (df['percentage_storage'].iloc[i], df['gsm8k'].iloc[i]),
                    ha='center', va='top', fontsize=12, xytext=(8, -7), textcoords='offset points')
    elif txt == 'dataset':
        plt.annotate(label, (df['percentage_storage'].iloc[i], df['gsm8k'].iloc[i]),
                    ha='center', va='top', fontsize=12, xytext=(0, -7), textcoords='offset points')
    elif txt == 'adam':
        plt.annotate(label, (df['percentage_storage'].iloc[i], df['gsm8k'].iloc[i]),
                    ha='right', va='bottom', fontsize=12, xytext=(5, 5), textcoords='offset points')
    else:
        plt.annotate(label, (df['percentage_storage'].iloc[i], df['gsm8k'].iloc[i]),
                    ha='center', va='bottom', fontsize=12, xytext=(0, 5), textcoords='offset points')

plt.savefig('tmp.png')
