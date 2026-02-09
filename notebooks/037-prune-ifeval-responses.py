import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

results_file = Path("/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results.jsonl")
generations_dir = Path("/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/generations")

df = pd.read_json(results_file, lines=True)

ifeval_mask = df['dataset_name'].str.contains('correct_ifeval_examples_extended_32_clean.jsonl', na=False)
ifeval_df = df[ifeval_mask]
len(ifeval_df)
df.loc[ifeval_mask, 'performance'] = 0.0

df.to_json(results_file, orient='records', lines=True)

print(f"Found {len(ifeval_df)} IFEval runs to check")

pruned_eval_run_ids = []

for idx, row in tqdm(ifeval_df.iterrows()):
    eval_run_id = row['eval_run_id']

    if pd.isna(eval_run_id):
        print(f"Skipping row {idx}: no eval_run_id")
        continue

    responses_file = generations_dir / eval_run_id / "responses.jsonl"

    if not responses_file.exists():
        print(f"Skipping {eval_run_id}: responses file not found")
        continue

    pruned_responses = []
    had_pruning = False

    with open(responses_file) as f:
        for line in f:
            response_obj = json.loads(line)
            if '\ufffd' in response_obj['response']:
                original_length = len(response_obj['response'])
                response_obj['response'] = response_obj['response'].split('\ufffd')[0]
                pruned_length = len(response_obj['response'])
                print(f"  {eval_run_id}: Pruned response from {original_length} to {pruned_length} chars")
                had_pruning = True
            pruned_responses.append(response_obj)

    if had_pruning:
        with open(responses_file, 'w') as f:
            for response_obj in pruned_responses:
                f.write(json.dumps(response_obj) + '\n')

        pruned_eval_run_ids.append(eval_run_id)
        print(f"✓ Pruned and saved {eval_run_id}")

print(f"\nTotal eval_run_ids with pruning: {len(pruned_eval_run_ids)}")

if pruned_eval_run_ids:
    print("\nUpdating final-results.jsonl...")
    df = pd.read_json(results_file, lines=True)

    for eval_run_id in pruned_eval_run_ids:
        mask = df['eval_run_id'] == eval_run_id
        df.loc[mask, 'performance'] = 0.0
        print(f"  Set performance=0.0 for {eval_run_id}")

    df.to_json(results_file, orient='records', lines=True)
    print(f"\n✓ Updated {len(pruned_eval_run_ids)} rows in final-results.jsonl")
