import json
from pathlib import Path

input_dir = Path("/network/scratch/b/brownet")
output_file = input_dir / "correct_ifeval_examples_extended_32_clean.jsonl"

num_splits = 50

all_examples = []

for split_idx in range(num_splits):
    split_file = input_dir / f"correct_ifeval_examples_extended_split_{split_idx:03d}.jsonl"

    if not split_file.exists():
        print(f"Warning: {split_file} does not exist, skipping...")
        continue

    print(f"Reading {split_file}...")
    with open(split_file) as f:
        for line in f:
            all_examples.append(json.loads(line))

    print(f"  Loaded {len(all_examples)} examples so far")

print(f"\nTotal examples before deduplication: {len(all_examples)}")

seen = set()
unique_examples = []
for example in all_examples:
    example_str = json.dumps(example, sort_keys=True)
    if example_str not in seen:
        seen.add(example_str)
        unique_examples.append(example)

print(f"Total unique examples after deduplication: {len(unique_examples)}")
print(f"Removed {len(all_examples) - len(unique_examples)} duplicates")

print(f"\nWriting to {output_file}...")
with open(output_file, 'w') as f:
    for example in unique_examples:
        f.write(json.dumps(example) + '\n')

print(f"Done! Aggregated {len(unique_examples)} unique examples into {output_file}")
