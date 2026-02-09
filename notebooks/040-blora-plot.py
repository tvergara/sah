# %%
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
blora_bits_dir = Path("../scratch/blora_bits-tamia")
files = sorted(blora_bits_dir.glob("*.json"))

# %%
layer_pattern = re.compile(r"layers\.(\d+)\.")


def extract_layer_bits(json_path):
    with open(json_path) as f:
        data = json.load(f)

    layer_bits = {}
    for key, bits in data.items():
        match = layer_pattern.search(key)
        if match:
            layer_idx = int(match.group(1))
            layer_bits[layer_idx] = layer_bits.get(layer_idx, 0) + bits
    return layer_bits


# %%
all_runs = []
for run_id, json_path in enumerate(files):
    layer_bits = extract_layer_bits(json_path)
    for layer_idx, bits in layer_bits.items():
        all_runs.append({"run_id": run_id, "layer": layer_idx, "bits": bits, "file": json_path.name})

df = pd.DataFrame(all_runs)

# %%
total_bits_per_run = df.groupby("run_id")["bits"].transform("sum")
df["bits_normalized"] = df["bits"] / total_bits_per_run

# %%
pivot = df.pivot(index="run_id", columns="layer", values="bits_normalized")
pivot = pivot.sort_index(axis=1)

# %%
plt.figure(figsize=(16, 10))
sns.heatmap(pivot, cmap="viridis", xticklabels=True, yticklabels=True)
plt.xlabel("Layer")
plt.ylabel("Run ID")
plt.title("Bits per Layer across Runs")
plt.tight_layout()
plt.savefig('blora.png', bbox_inches='tight', dpi=300)

# %%
file_mapping = df[["run_id", "file"]].drop_duplicates().set_index("run_id")["file"].to_dict()
for run_id, filename in file_mapping.items():
    print(f"{run_id}: {filename}")
