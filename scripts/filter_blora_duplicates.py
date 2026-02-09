import argparse
import json
from collections import defaultdict


def filter_blora_duplicates(input_file, output_file=None):
    if output_file is None:
        output_file = input_file

    blora_counts = defaultdict(int)
    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get('experiment_name') == 'blora':
                blora_counts[data['experiment_id']] += 1

    duplicate_ids = {k for k, v in blora_counts.items() if v > 1}
    print(f"Found {len(duplicate_ids)} blora experiment_ids with duplicates")

    seen_blora = set()
    kept = 0
    removed = 0
    output_lines = []

    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            if data.get('experiment_name') == 'blora' and data['experiment_id'] in duplicate_ids:
                if data['experiment_id'] in seen_blora:
                    removed += 1
                    continue
                seen_blora.add(data['experiment_id'])
            kept += 1
            output_lines.append(line)

    print(f"Kept: {kept} entries")
    print(f"Removed: {removed} entries")

    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line)

    print(f"Done. File written to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter duplicate blora entries from results file')
    parser.add_argument('input_file', help='Input JSONL file')
    parser.add_argument('--output', '-o', help='Output file (defaults to overwriting input)')
    args = parser.parse_args()

    filter_blora_duplicates(args.input_file, args.output)
