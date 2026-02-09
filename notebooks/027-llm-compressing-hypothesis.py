import gzip

import torch
from transformers import AutoModelForCausalLM

pretrained_model_name_or_path = 'EleutherAI/pythia-160m'

# 10 evenly spread checkpoints from 0 to 143000 steps
checkpoints = [
    'step0', 'step15000', 'step30000', 'step45000', 'step60000',
    'step75000', 'step90000', 'step105000', 'step120000', 'step143000'
]

print("Checkpoint | Compression Ratio")
print("-" * 35)

for checkpoint in checkpoints:
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        revision=checkpoint
    )

    # Flatten all weights into single tensor
    weight_vector = torch.cat([p.flatten().detach() for p in model.parameters()])

    # Convert to bytes and compress
    weight_bytes = weight_vector.cpu().numpy().tobytes()
    compressed_bytes = gzip.compress(weight_bytes, compresslevel=9)

    # Calculate compression ratio
    compression_ratio = len(weight_bytes) / len(compressed_bytes)

    print(f"{checkpoint:12} | {compression_ratio:.4f}x")

    # Free memory
    del model, weight_vector, weight_bytes, compressed_bytes
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
