import json
from pathlib import Path

input_dir = Path("/network/scratch/b/brownet")
output_file = input_dir / "correct_ifeval_examples_extended_32.jsonl"

num_splits = 150

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

print(f"\nTotal examples: {len(all_examples)}")

print(f"Writing to {output_file}...")
with open(output_file, 'w') as f:
    for example in all_examples:
        f.write(json.dumps(example) + '\n')

print(f"Done! Aggregated {len(all_examples)} examples into {output_file}")
