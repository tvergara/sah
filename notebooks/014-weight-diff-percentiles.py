import gc

import numpy as np
import torch
from transformers import AutoModelForCausalLM

# ----- 0.  setup GPU devices ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device)
print(f"Using primary device: {device}")
print(f"Using secondary device: {device2}")
print(f"Available GPUs: {torch.cuda.device_count()}")

# ----- 1.  load the two models to compute weight differences ---------------
print("Loading deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B...")
model_deepseek = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).eval().to(device)
# model_deepseek = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", torch_dtype=torch.float16).eval().to(device)

print("Loading Qwen/Qwen2.5-Math-1.5B...")
# model_qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.float16).eval().to(device2)
model_qwen = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).eval().to(device2)

# ----- 2.  compute weight differences excluding token embeddings -----
print("Computing weight differences...")
weight_diffs = []

qwen_params = dict(model_qwen.named_parameters())

for name, param_deepseek in model_deepseek.named_parameters():
    # Skip token embedding parameters
    if 'embed_tokens' in name or 'wte' in name or 'word_embeddings' in name:
        continue

    if name in qwen_params:
        param_qwen = qwen_params[name]
        if param_deepseek.shape == param_qwen.shape:
            # Move to same device, compute difference
            param_qwen_device = param_qwen.to(device)
            diff = (param_deepseek - param_qwen_device).cpu()

            # Flatten and add to weight_diffs list
            weight_diffs.append(diff.flatten())

            del param_qwen_device
        else:
            print(f"Shape mismatch for {name}: {param_deepseek.shape} vs {param_qwen.shape}")

    if len(weight_diffs) > 20:
        break

# Concatenate all weight differences into a single tensor
all_weight_diffs = torch.cat(weight_diffs, dim=0)
print(f"Total weight differences: {all_weight_diffs.numel()}")

# Clean up models to free memory
del model_deepseek, model_qwen, qwen_params, weight_diffs
torch.cuda.empty_cache()
gc.collect()

# ----- 3.  compute percentiles in intervals of 10 -------------------------
print("\nComputing percentiles in intervals of 10...")
percentiles = list(range(0, 101, 10))  # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Convert to numpy for percentile computation
weight_diffs_np = all_weight_diffs.detach().numpy()

# Compute percentiles
percentile_values = np.percentile(weight_diffs_np, percentiles)
# Display results
print("\nWeight Difference Percentiles:")
print("=" * 40)
for p, val in zip(percentiles, percentile_values):
    print(f"{p:3d}th percentile: {val:.6f}")

# Additional statistics
print("\nAdditional Statistics:")
print(f"Mean: {weight_diffs_np.mean():.6f}")
print(f"Std:  {weight_diffs_np.std():.6f}")
print(f"Min:  {weight_diffs_np.min():.6f}")
print(f"Max:  {weight_diffs_np.max():.6f}")

# Compute absolute value percentiles as well
abs_weight_diffs_np = np.abs(weight_diffs_np)
abs_percentile_values = np.percentile(abs_weight_diffs_np, percentiles)

print("\nAbsolute Weight Difference Percentiles:")
print("=" * 40)
for p, val in zip(percentiles, abs_percentile_values):
    print(f"{p:3d}th percentile: {val:.6f}")

print("\nAbsolute Value Statistics:")
print(f"Mean: {abs_weight_diffs_np.mean():.6f}")
print(f"Std:  {abs_weight_diffs_np.std():.6f}")
print(f"Min:  {abs_weight_diffs_np.min():.6f}")
print(f"Max:  {abs_weight_diffs_np.max():.6f}")
