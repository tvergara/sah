import glob
import os

from transformers.models.auto.tokenization_auto import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen2.5-1.5B"
)

save_dir = '/network/scratch/b/brownet/synthetic-data/finetuned-model-data-qwen'

# Find all .txt files in the save directory
txt_files = glob.glob(os.path.join(save_dir, '*.txt'))
print(f"Found {len(txt_files)} .txt files")

# Function to read and decode tokenized data
def read_and_decode_file(file_path, max_samples=5):
    """Read tokenized data from file and convert back to text."""
    print(f"\n=== File: {os.path.basename(file_path)} ===")

    with open(file_path) as f:
        lines = f.readlines()

    print(f"Total lines in file: {len(lines)}")

    for i, line in enumerate(lines[:max_samples]):
        line = line.strip()
        if not line:
            continue

        try:
            # Parse the tokenized data (assuming space-separated integers)
            token_ids = [int(x) for x in line.split()]

            # Decode back to text
            decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)

            print(f"\nSample {i+1}:")
            print(f"Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
            print(f"Decoded text: {decoded_text[:200]}{'...' if len(decoded_text) > 200 else ''}")

        except Exception as e:
            print(f"Error processing line {i+1}: {e}")
            print(f"Raw line: {line[:100]}...")

# Examine the first few files
for file_path in txt_files[:3]:  # Check first 3 files
    read_and_decode_file(file_path)
