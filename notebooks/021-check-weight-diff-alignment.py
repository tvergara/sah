import gc

import torch
from transformers import AutoModelForCausalLM

from sah.algorithms.utils import load_weights_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16)

# from torch import nn
# def replace_linear_layers(model, count=0):
#     for name, module in model.named_children():
#             if isinstance(module, nn.Linear):
#                 count += module.weight.numel()
#                 if module.bias is not None:
#                     count += module.bias.numel()
#             else:
#                 count = replace_linear_layers(module, count=count)
#     return count

# linear_p = replace_linear_layers(base_model)

# total_p = sum(p.numel() for p in base_model.parameters())
# total_p
# linear_p / total_p

# base_model

# Load final checkpoint model
final_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16)
load_weights_from_checkpoint(final_model, '/network/scratch/b/brownet/hydra-runs/distil-on-meta-math/checkpoints/last', model_name='model')

# Compute diff between base model and final checkpoint (keep in memory)
final_diff = {}
for name, base_param in base_model.named_parameters():
    final_param = dict(final_model.named_parameters())[name]
    final_diff[name] = final_param - base_param

print(f"Computed final diff with {len(final_diff)} parameter groups")

# Clean up final model to save memory
del final_model
gc.collect()

# Model for loading checkpoints
model_ckp = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16)
checkpoints = [1000, 2000, 3000, 4000, 5000]

# Store alignment results
alignment_results = {}

for ckp in checkpoints:
    checkpoint = f"checkpoint-step-{ckp}"
    checkpoint_path = f"/network/scratch/b/brownet/hydra-runs/distil-on-meta-math/checkpoints/{checkpoint}"

    # Load checkpoint weights
    load_weights_from_checkpoint(model_ckp, checkpoint_path, model_name='model')

    # Compute diff between base model and current checkpoint
    current_diff = {}
    for name, base_param in base_model.named_parameters():
        ckp_param = dict(model_ckp.named_parameters())[name]
        current_diff[name] = ckp_param - base_param

    # Compute sign alignment with final diff
    total_params = 0
    aligned_params = 0

    for name in final_diff.keys():
        final_diff_vals = final_diff[name]
        current_diff_vals = current_diff[name]

        # Consider parameters with final diff < 0.01 as "any sign works" (always aligned)
        small_diff_mask = torch.abs(final_diff_vals) < 0.01

        final_signs = torch.sign(final_diff_vals)
        current_signs = torch.sign(current_diff_vals)

        # Count parameters with same sign OR where final diff is small
        same_sign = (final_signs == current_signs)
        aligned = same_sign | small_diff_mask

        aligned_params += aligned.sum().item()
        total_params += final_diff_vals.numel()

    alignment_percentage = (aligned_params / total_params) * 100
    alignment_results[ckp] = alignment_percentage

    print(f"Checkpoint {ckp}: {alignment_percentage:.2f}% sign alignment with final checkpoint")

    # Clean up current diff to save memory
    del current_diff
    gc.collect()

print("\nAlignment Results Summary:")
for ckp, alignment in alignment_results.items():
    print(f"Step {ckp}: {alignment:.2f}%")
