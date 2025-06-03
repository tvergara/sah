import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('~/scratch/hydra-runs/grammar-entropy/grammar-43-entropies.csv')

df

order = []

for i in range(1, 27):
    number = i * 100
    padded = f"{number:05d}"
    order.append(f"pretraining_step_{padded}")

for i in range(1, 11):
    number = i * 100
    padded = f"{number:05d}"
    order.append(f"finetuning_step_{padded}")

order



information_gains = []
labels = []
for i in range(len(order) - 1):
    mask = (df['first_input'].isna()) & (df['second_input'].isna())
    entropy = df[mask]['entropy'].item()

    mask = (df['first_input'] == order[i]) & (df['second_input'] == order[i + 1])
    both_entropy = entropy - df[mask]['entropy'].item()

    mask = (df['first_input'] == order[i]) & (df['second_input'].isna())
    first_entropy = entropy - df[mask]['entropy'].item()

    gain = both_entropy - first_entropy
    information_gains.append(gain)
    labels.append(order[i])

labels
plt.figure()
plt.plot(labels, information_gains)
plt.xticks(rotation=45, ha='right')
plt.title('predictive info')
plt.tight_layout()
plt.savefig('info_gain')

information_lost = []
labels = []
for i in range(len(order) - 1):
    mask = (df['first_input'].isna()) & (df['second_input'].isna())
    entropy = df[mask]['entropy'].item()

    mask = (df['first_input'] == order[i]) & (df['second_input'] == order[i + 1])
    both_entropy = entropy - df[mask]['entropy'].item()

    mask = (df['first_input'] == order[i + 1]) & (df['second_input'].isna())
    second_entropy = entropy - df[mask]['entropy'].item()

    loss = both_entropy - second_entropy
    information_lost.append(loss)
    labels.append(order[i])

labels
plt.figure()
plt.plot(labels, information_lost)
plt.xticks(rotation=45, ha='right')
plt.title('info lost')
plt.tight_layout()
plt.savefig('info_lost')
