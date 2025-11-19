import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def find_lima_runs_to_evaluate(csv_path: str):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    print(f"Total rows in CSV: {len(df)}")

    lima_mask = df['dataset_name'].str.contains('lima', case=False, na=False)
    lima_df = df[lima_mask]
    print(f"Rows with lima dataset: {len(lima_df)}")

    no_performance_mask = (
        lima_df['performance'].isna() |
        (lima_df['performance'] == 0.0) |
        (lima_df['performance'] == 0)
    )
    lima_no_perf = lima_df[no_performance_mask]

    print(f"Lima rows needing evaluation: {len(lima_no_perf)}")
    print()
    print("Runs to evaluate:")
    print("-" * 80)

    for idx, row in lima_no_perf.iterrows():
        print(f"Row {idx}:")
        print(f"  experiment_name: {row['experiment_name']}")
        print(f"  experiment_id: {row['experiment_id']}")
        print(f"  eval_run_id: {row['eval_run_id']}")
        print(f"  model_name: {row['model_name']}")
        print(f"  bits: {row['bits']}")
        print(f"  performance: {row['performance']}")
        print()

    return lima_no_perf


def run_evaluation(eval_run_id: str, generations_dir: Path, google_research_dir: Path):
    run_dir = generations_dir / eval_run_id
    input_data_file = run_dir / "input_data.jsonl"
    responses_file = run_dir / "responses.jsonl"
    output_dir = run_dir / "eval_output"
    output_dir.mkdir(exist_ok=True)

    cmd = [
        "python3", "-m", "instruction_following_eval.evaluation_main",
        f"--input_data={input_data_file}",
        f"--input_response_data={responses_file}",
        f"--output_dir={output_dir}"
    ]

    print(f"  Running: cd {google_research_dir} && {' '.join(cmd)}")

    subprocess.run(cmd, cwd=google_research_dir, check=True)

    eval_results_file = output_dir / "eval_results_loose.jsonl"

    with open(eval_results_file) as f:
        results = [json.loads(line) for line in f]

    total_instructions = 0
    followed_instructions = 0
    for r in results:
        total_instructions += r['instruction_id_list'].__len__()
        followed_instructions += sum(r['follow_instruction_list'])

    performance = followed_instructions / total_instructions

    return performance


def evaluate_lima_runs(csv_path: str, generations_dir: str, google_research_dir: str):
    csv_path = Path(csv_path)
    generations_dir = Path(generations_dir)
    google_research_dir = Path(google_research_dir)

    lima_no_perf = find_lima_runs_to_evaluate(csv_path)

    print("\nStarting evaluation...")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    for idx, row in lima_no_perf.iterrows():
        eval_run_id = row['eval_run_id']

        if pd.isna(eval_run_id):
            print(f"\nSkipping row {idx}: no eval_run_id")
            continue

        print(f"\nEvaluating row {idx} (eval_run_id: {eval_run_id})...")

        performance = run_evaluation(eval_run_id, generations_dir, google_research_dir)

        print(f"  Performance: {performance:.4f}")
        df.loc[idx, 'performance'] = performance

    df.to_csv(csv_path, index=False)
    print(f"\n✓ Updated {len(lima_no_perf)} rows in {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find and evaluate LIMA dataset runs"
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/results.csv',
        help='Path to results CSV file'
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    generations_dir = csv_path.parent / "generations"
    google_research_dir = Path.home() / "google-research"

    evaluate_lima_runs(args.csv, generations_dir, google_research_dir)


if __name__ == "__main__":
    main()
