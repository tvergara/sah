import json
import sys
from pathlib import Path


def load_jsonl(file_path):
    results = []
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist, skipping...")
        return results

    print(f"Reading {file_path}...")
    with open(file_path) as f:
        for line in f:
            results.append(json.loads(line))

    print(f"  Loaded {len(results)} entries")
    return results


def merge_results(source_file, dest_file):
    source_path = Path(source_file)
    dest_path = Path(dest_file)

    source_results = load_jsonl(source_path)
    dest_results = load_jsonl(dest_path)

    all_results = dest_results + source_results

    print(f"\nTotal entries before deduplication: {len(all_results)}")

    seen_eval_ids = set()
    unique_results = []
    for result in all_results:
        eval_run_id = result.get('eval_run_id')
        if eval_run_id not in seen_eval_ids:
            seen_eval_ids.add(eval_run_id)
            unique_results.append(result)

    print(f"Total unique entries after deduplication: {len(unique_results)}")
    print(f"Removed {len(all_results) - len(unique_results)} duplicates")
    print("(Deduplication based on eval_run_id, keeping main file version on collision)")

    backup_path = dest_path.with_suffix('.jsonl.bak')
    if dest_path.exists():
        print(f"\nCreating backup: {backup_path}")
        dest_path.rename(backup_path)

    print(f"\nWriting to {dest_path}...")
    with open(dest_path, 'w') as f:
        for result in unique_results:
            f.write(json.dumps(result) + '\n')

    print(f"Done! Merged {len(unique_results)} unique entries into {dest_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_results.py <source_file> <dest_file>")
        print("Example: python merge_results.py final-results-cluster1.jsonl final-results.jsonl")
        sys.exit(1)

    source_file = sys.argv[1]
    dest_file = sys.argv[2]

    merge_results(source_file, dest_file)
