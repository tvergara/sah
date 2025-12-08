import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sah.algorithms.dataset_handlers.lima import LimaHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "allenai/Olmo-3-1025-7B"

print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    revision="stage1-step1413814"
)
model.eval()

print("Creating LimaHandler...")
handler = LimaHandler(
    tokenizer=tokenizer,
    dataset_name="GAIR/lima",
    block_size=1548,
    max_examples=None,
    generations_dir=None
)

print("Loading validation dataset (IFEval)...")
val_dataset = handler.get_val_dataset()
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"Total validation examples: {len(val_dataset)}")

class MockPLModule:
    def __init__(self, model, max_length=512):
        self.model = model
        self.max_length = max_length

pl_module = MockPLModule(model, max_length=512)

print("\nGenerating responses...")
total_batches = len(val_dataloader)

for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Generating")):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    result = handler.validate_batch(pl_module, batch, batch_idx)

    if batch_idx % 10 == 0:
        print(f"\nBatch {batch_idx}/{total_batches}")
        print(f"Processed {result['total_count']} examples in this batch")

print(f"\n\nTotal generations: {len(handler.generations)}")
print("\n=== Sample generations ===")
for i, gen in enumerate(handler.generations[:3]):
    print(f"\n--- Example {i+1} ---")
    print(f"Prompt: {gen['prompt'][:300]}...")
    print(f"Response: {gen['response'][:600]}...")
    break


handler.generations[0]
