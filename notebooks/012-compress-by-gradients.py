import gc

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ----- 0.  setup GPU devices ------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else device)
print(f"Using primary device: {device}")
print(f"Using secondary device: {device2}")
print(f"Available GPUs: {torch.cuda.device_count()}")

# Enable mixed precision for memory efficiency
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# ----- 1.  load the two models to compute weight differences ---------------
print("Loading deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B...")
model_deepseek = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", torch_dtype=torch.float16).eval().to(device)

print("Loading Qwen/Qwen2.5-Math-1.5B...")
model_qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", torch_dtype=torch.float16).eval().to(device2)


# ----- 2.  compute weight differences (g_target) excluding token embeddings -----
g_target = {}
qwen_params = dict(model_qwen.named_parameters())

for name, param_deepseek in model_deepseek.named_parameters():
    # Skip token embedding parameters
    if 'embed_tokens' in name or 'wte' in name or 'word_embeddings' in name:
        continue

    if name in qwen_params:
        param_qwen = qwen_params[name]
        if param_deepseek.shape == param_qwen.shape:
            # Move to same device, compute difference, then move to CPU to save GPU memory
            param_qwen_device = param_qwen.to(device)
            diff = (param_deepseek - param_qwen_device)#.cpu()
            # Quantize diff: 0 if magnitude < 0.02, 0.1 if positive, -0.1 if negative
            # diff_quantized = torch.where(torch.abs(diff) < 0.001, 0.0,
            #                            torch.where(diff > 0, 0.1, -0.1))
            g_target[name] = diff
            del param_qwen_device
        else:
            print(f"Shape mismatch for {name}: {param_deepseek.shape} vs {param_qwen.shape}")

print(f"Computed weight differences for {len(g_target)} parameters")


# ----- 3.  use one of the models for the optimization (move to primary device) -----
model = model_qwen.to(device)
for p in model.parameters():
    p.requires_grad_(True)          # we never touch θ

# ----- 4.  make x*, y* learnable -------------------------------------------
seq_len = 60
batch = 10
d_model = model.config.hidden_size
vocab_size = model.config.vocab_size

# Generate sequences: random first token, then greedy generation
input_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
input_ids[:, 0] = torch.randint(1, vocab_size-1, (batch,), device=device)

# Generate remaining tokens greedily
for i in range(1, seq_len):
    with torch.no_grad():
        outputs = model_deepseek(input_ids[:, :i])
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        input_ids[:, i] = next_token

# Get hidden states and logits from DeepSeek model
with torch.no_grad():
    deepseek_outputs = model_deepseek(input_ids, output_hidden_states=True)

    # x_star: first hidden state (after embedding layer)
    x_star_data = deepseek_outputs.hidden_states[0]  # [batch, seq_len, d_model]

    # y_star: logits from the model
    y_star_data = deepseek_outputs.logits  # [batch, seq_len, vocab_size]

# Create learnable parameters initialized with DeepSeek data
x_star = x_star_data.clone().detach().requires_grad_(True)
y_star = y_star_data.clone().detach().requires_grad_(True)

# Clean up temporary model
torch.cuda.empty_cache()
gc.collect()

print(f"Initialized x_star: {x_star.shape}, y_star: {y_star.shape}")

del model_deepseek
del qwen_params
torch.cuda.empty_cache()
gc.collect()

# use all parameters for gradient targeting
target_params = []
target_param_names = []
for name, param in model.named_parameters():
    if name in g_target:
        target_params.append(param)
        target_param_names.append(name)

print(f"Targeting {len(target_params)} parameters for gradient optimization")


opt = torch.optim.Adam([x_star, y_star], lr=1e-2)   # optimize both x* and y*

criterion = nn.KLDivLoss(reduction='batchmean')

# Parameters for batched processing
BATCH_SIZE = 1000  # Process parameters in batches to save memory

# helper to run model from residuals -----------------------
def f_from_residual(res):                    # res: [T, d]
    return model(inputs_embeds=res).logits  # [T, V]


with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    # ----- 5.  outer loop with batched processing ------------------------------------
    for outer_step in range(5000):
        opt.zero_grad()

        # ---- inner loss & its gradient wrt θ -----------------
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = f_from_residual(x_star)                     # [T, V]
            log_probs = torch.log_softmax(logits, dim=-1)
            target_probs = torch.softmax(y_star, dim=-1)
            inner_loss = criterion(log_probs, target_probs)
            # breakpoint()

        total_meta_loss = 0.0
        total_opposite_directions = 0
        total_elements = 0
        l1_norm = 0
        l1_norm_base = 0
        dot = 0

        # Process parameters in batches to save memory
        for batch_start in range(0, len(target_params), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(target_params))
            batch_params = target_params[batch_start:batch_end]
            batch_names = target_param_names[batch_start:batch_end]

            # compute gradients for this batch of parameters
            g_actuals = torch.autograd.grad(inner_loss,
                                           batch_params,
                                           retain_graph=True,
                                           create_graph=True)
            # breakpoint()

            mini_b = 10
            mini_loss =  0.0
            # ---- outer loss (L1, computed per parameter and backpropagated directly) ----
            for i, (name, g_actual) in enumerate(zip(batch_names, g_actuals)):
                g_target_param = g_target[name]#.to(device)  # Move target to GPU only when needed

                # L1 loss for this parameter
                # param_loss = (torch.sqrt(torch.abs(g_actual - g_target_param)) / g_actual.numel()).sum()
                param_loss = ((g_actual - g_target_param)**2  / 10000).sum()
                # param_loss = torch.sqrt(torch.abs(g_actual - g_target_param)).sum() / 1e7
                # breakpoint()

                if i == 0 and batch_start == 0:
                    # breakpoint()
                    pass

                # Count opposite directions (including small magnitude targets)
                product = g_actual * g_target_param
                small_magnitude_mask = torch.abs(g_target_param) < 0.001
                opposite_mask = (product > 0) | small_magnitude_mask
                total_opposite_directions += opposite_mask.sum().item()
                total_elements += product.numel()

                total_meta_loss += param_loss.item()


                g_actual - g_target_param

                # backpropagate immediately for this parameter
                mini_loss += param_loss
                diff = torch.abs(g_actual /10 - g_target_param)
                l1_norm += torch.abs(diff / 1000).sum().item()
                l1_norm_base += torch.abs(torch.abs(g_target_param) / 1000).sum().item()
                dot += (g_actual * g_target_param).sum().item()

                if (i + 1) % mini_b == 0 or (i  + 1) == len(batch_names):
                    # breakpoint()
                    mini_loss.backward(retain_graph=True)
                    mini_loss = 0.0



                # Clean up intermediate tensors
                # del g_target_param
                # del param_loss

            # Clean up batch gradients
            # del g_actuals
            torch.cuda.empty_cache()

        # breakpoint()

        # if outer_step % 5 == 0:
        #     breakpoint()

        x_star.grad.data.clamp_(-100, 100)
        x_star.data -= x_star.grad * 2e-4
        x_star.grad = None
        y_star.grad.data.clamp_(-100, 100)
        y_star.data -= y_star.grad * 2e-2
        y_star.grad = None

        # opt.step()

        if outer_step % 1 == 0:
            opposite_pct = (total_opposite_directions / total_elements) * 100 if total_elements > 0 else 0
            print(f"{outer_step:4d}  meta‑loss={total_meta_loss:.4e}  opposite_dirs={opposite_pct:.1f}%  l1_norm={l1_norm:.4e} l1_norm_base={l1_norm_base:.4e} dot={dot:.4e}", flush=True)

        # Periodic garbage collection
        if outer_step % 100 == 0:
            gc.collect()
