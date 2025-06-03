import glob
import os

import matplotlib.pyplot as plt
import torch

base_path = '/network/scratch/b/brownet/synthetic-data/activations/subgrammar'
output_dir = 'step_00100'

L = 3
out_glob = os.path.join(base_path, output_dir, f"layer{L}_batch*.pt")

out_paths = sorted(
    glob.glob(out_glob),
    key=lambda p: int(os.path.basename(p).split("_")[1].removeprefix("batch").removesuffix(".pt"))
)[:100]

out_tensors = []
masks = []
for path in out_paths:
    values = torch.load(path)
    activations = values['activations']
    mask = values['mask']

    if activations.shape[1] == 512:
        out_tensors.append(activations)
        masks.append(mask)

data = torch.cat(out_tensors, dim=0)
full_mask = torch.cat(masks, dim=0)

data.shape
full_mask.shape
full_mask

dim = 1

data_in_dim = data[:, :, dim]

data_in_dim.shape
data_in_dim


# ── pretend this is your tensor ───────────────────────────
# ──────────────────────────────────────────────────────────

# 1.   Detach → move to CPU → NumPy
vals = data_in_dim.detach().flatten().cpu()
mask       = full_mask.detach().flatten().cpu().bool()

# 2)  Split by label
vals_zero  = vals[~mask].numpy()   # label 0
vals_one   = vals[mask].numpy()    # label 1

sum(mask)
mask.shape

# 3)  Plot
fig, ax = plt.subplots(figsize=(6, 4))

bins = 50                      # or "auto", or np.linspace(...)
ax.hist(vals_zero, bins=bins, density=True,
        alpha=0.6, edgecolor="black", label="mask = 0")
ax.hist(vals_one,  bins=bins, density=True,
        alpha=0.6, edgecolor="black", label="mask = 1")

# 4)  Cosmetics
ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.set_title("Distributions conditioned on full_mask")
ax.tick_params(which="both", length=3)   # tiny tick bars
ax.legend(frameon=False)

plt.tight_layout()

plt.savefig('distribution-acts.png')
