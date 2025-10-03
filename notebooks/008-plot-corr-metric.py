import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('~/scratch/hydra-runs/pretrain-then-finetune/grammar-entropies.csv')
entropies = pd.read_csv('~/scratch/hydra-runs/entropy-bottleneck/tmp-entropies.csv')

df
entropies

entropies['metric'] = entropies['H_BA'] - 0.5 * entropies['H_A']

df

# df['unconditional'] = df['unconditional'].astype(bool)

# # 2. Split the two variants into Series keyed by id
# base   = df[df['unconditional']].set_index('id')['entropy']        # unconditional == True
# variant = df[~df['unconditional']].set_index('id')['entropy']      # unconditional == False

# 3. Compute the difference (variant − base) and package it as a tidy DataFrame
# diff_df = (base - variant).rename('entropy_diff').reset_index()

# 4. (Optional) sanity-check that every id had both rows
# assert diff_df['entropy_diff'].notna().all(), "Some ids are missing a True/False pair"

# diff_df


# diff_na_clean = diff_df.dropna(subset=['entropy_diff'])          # or diff_df if that’s your name

# --- 2.  Bring in the matching metric values ---------------------------------
merged = (
    df
      .merge(entropies[['id', 'metric']], on='id', how='inner')   # keep only ids present in both
)
merged

# --- 3.  Compute the Pearson correlation -------------------------------------
pearson_corr = merged['entropy'].corr(merged['metric'])
print(f"Pearson r: {pearson_corr:.4f}")

# (Optional) Spearman rank correlation:
spearman_corr = merged['entropy'].corr(merged['metric'], method='spearman')
print(f"Spearman ρ: {spearman_corr:.4f}")



# ── 1. Basic scatter ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(
    merged['metric'],
    merged['entropy'],
    alpha=0.8,                # a touch of transparency helps see overlaps
)

ax.set_ylabel('Bayesian Information')
ax.set_xlabel('Metric')
ax.set_title('Bayesian Information vs. Metric')

# ── 2. Optional: add a best-fit line (simple least-squares) ───────────────────
m, b = np.polyfit(merged['metric'], merged['entropy'] , 1)
xline = np.linspace(merged['metric'].min(), merged['metric'].max(), 100)
ax.plot(xline, m * xline + b, linewidth=1.5)

# ── 3. Finish up ─────────────────────────────────────────────────────────────
plt.tight_layout()

plt.savefig('corr_metric')
