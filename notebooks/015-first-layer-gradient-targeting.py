import gc

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ----- 0. Setup GPU device ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- 1. Load the two Llama models ----------------------------------------
print("Loading meta-llama/Llama-2-7b-chat-hf...")
model_chat = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16).eval().to(device)

print("Loading meta-llama/Llama-2-7b-hf...")
model_base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16).eval().to(device)

# ----- 2. Compute weight differences for first layer only -----------------
print("Computing weight differences for first layer...")
g_target = {}
base_params = dict(model_base.named_parameters())

# Target only first layer parameters (layers.0)
for name, param_chat in model_chat.named_parameters():
    # Skip token embeddings and only include first layer
    if 'embed_tokens' in name or 'wte' in name or 'word_embeddings' in name:
        continue
    if not name.startswith('model.layers.0.'):
        continue

    if name in base_params:
        param_base = base_params[name]
        if param_chat.shape == param_base.shape:
            diff = param_chat - param_base
            diff_quantized = torch.where(torch.abs(diff) < 0.001, 0.0,
                                       torch.where(diff > 0, 0.1, -0.1))
            g_target[name] = diff_quantized
        else:
            print(f"Shape mismatch for {name}: {param_chat.shape} vs {param_base.shape}")

print(f"Computed weight differences for {len(g_target)} first layer parameters")

# ----- 3. Setup model for optimization ------------------------------------
model = model_base

model.model.layers = nn.ModuleList([model.model.layers[0]])
# Only enable gradients for first layer parameters to save memory
for name, param in model.named_parameters():
    if name.startswith('model.layers.0.'):
        param.requires_grad_(True)
    else:
        param.requires_grad_(False)

# ----- 4. Create synthetic input and target data --------------------------
seq_len = 70
batch_size = 20
d_model = model.config.hidden_size
vocab_size = model.config.vocab_size

# Generate random input sequences
input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
input_ids[:, 0] = torch.randint(1, vocab_size-1, (batch_size,), device=device)

# Generate remaining tokens greedily using chat model
for i in range(1, seq_len):
    for j in range(0, batch_size, 10):
        with torch.no_grad():
            max_j = min(j + 10, batch_size)
            outputs = model_chat(input_ids[j:max_j, :i])
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_ids[j:max_j, i] = next_token

# Get target hidden states and logits from chat model
with torch.no_grad():
    chat_outputs = model_chat(input_ids, output_hidden_states=True)
    x_star_data = chat_outputs.hidden_states[0]  # First hidden state
    y_star_data = chat_outputs.hidden_states[1]  # Target logits

# Create learnable parameters
x_star = x_star_data.clone().detach().requires_grad_(True)
y_star = y_star_data.clone().detach().requires_grad_(True)

# Clean up chat model
del model_chat, base_params, chat_outputs
torch.cuda.empty_cache()
gc.collect()

print(f"Initialized x_star: {x_star.shape}, y_star: {y_star.shape}")

x_star.shape
# model.model.layers[0](x_star)
# ----- 5. Setup optimization ----------------------------------------------
# Get target parameters (first layer only)
target_params = []
target_param_names = []
for name, param in model.named_parameters():
    if name in g_target:
        target_params.append(param)
        target_param_names.append(name)

print(f"Targeting {len(target_params)} first layer parameters")
# ----- 6. Model forward function from residuals ---------------------------
def f_from_residual(res):
    return model(inputs_embeds=res, output_hidden_states=True).hidden_states[1]

with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    # ----- 7. Main optimization loop ------------------------------------------
    print("Starting optimization...")
    for step in range(1000):
        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            res_stream = f_from_residual(x_star)
            inner_loss = torch.norm(res_stream - y_star)

        # Compute gradients for target parameters
        g_actuals = torch.autograd.grad(inner_loss, target_params, retain_graph=True, create_graph=True)

        # Compute meta-loss (gradient targeting)
        meta_loss = 0.0
        total_opposite_directions = 0
        total_elements = 0

        for name, g_actual in zip(target_param_names, g_actuals):
            g_target_param = g_target[name]

            # L2 loss between actual and target gradients
            # breakpoint()
            param_loss = 1-torch.nn.functional.cosine_similarity(g_actual.reshape(1, -1), g_target_param.reshape(1, -1))
            meta_loss += param_loss
            # if step % 100 == 0 and total_elements == 0:
            #     breakpoint()
            #     pass

            # Track gradient direction alignment
            product = g_actual * g_target_param
            small_magnitude_mask = torch.abs(g_target_param) < 0.001
            opposite_mask = (product > 0) | small_magnitude_mask
            total_opposite_directions += opposite_mask.sum().item()
            total_elements += product.numel()

        # breakpoint()
        # Backward pass
        meta_loss.backward()

        # if step % 100 == 0:
        #     breakpoint()
        #     pass
        # Gradient clipping and manual update
        x_star.grad.data.clamp_(-100, 100)
        y_star.grad.data.clamp_(-100, 100)

        x_star.data -= x_star.grad * 2e-2
        y_star.data -= y_star.grad * 2e-2

        x_star.grad = None
        y_star.grad = None

        # Logging
        if step % 10 == 0:
            opposite_pct = (total_opposite_directions / total_elements) * 100 if total_elements > 0 else 0
            print(f"Step {step:4d}: meta_loss={meta_loss.item():.4e}, opposite_dirs={opposite_pct:.1f}%")

        # Periodic cleanup
        if step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

print("Optimization completed!")
