from collections import defaultdict

import pandas as pd

JSONL_PATH = '/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results.jsonl'

DATASET_MAPPING = {
    'metamath': 'meta-math/MetaMathQA',
    'flores': 'allenai/nllb',
    'ifeval': 'ifeval:/network/scratch/b/brownet/correct_ifeval_examples_extended_32_clean.jsonl',
}

SEEDS = [1]
LR_ONLINE_CODING = [1e-4, 1e-5]
MAX_EXAMPLES = [1024, 2048, 4096, 8192, 16384, 32768]

df = pd.read_json(JSONL_PATH, lines=True)

expected_jobs = []

models = [
    'smollm3',
    'smollm3-step0',
    'smollm3-step1720k',
    'smollm3-stage2',
    'smollm3-stage3',
    'olmo3-7b-step0',
    'olmo3-7b-step707k',
    'olmo3-7b-stage2-step6k',
    'olmo3-7b-stage2-step12k',
    'olmo3-7b-stage2-step24k',
    'olmo3-7b-stage2-step48k',
    'olmo3-1025-7b',
    'olmo3-7b-instruct-step200',
    'olmo3-7b-instruct-step400'
]

datasets_short = ['metamath', 'flores', 'ifeval']

for model in models:
    for dataset in datasets_short:
        dataset_full = DATASET_MAPPING[dataset]

        for seed in SEEDS:
            for lr in LR_ONLINE_CODING:
                expected_jobs.append({
                    'model': model,
                    'dataset': dataset_full,
                    'strategy': 'online_coding',
                    'seed': seed,
                    'lr': lr,
                    'expected_count': len(MAX_EXAMPLES) * 3,
                })

        for seed in SEEDS:
            expected_jobs.append({
                'model': model,
                'dataset': dataset_full,
                'strategy': 'icl',
                'seed': seed,
                'lr': None,
                'expected_count': 1,
            })

        expected_jobs.append({
            'model': model,
            'dataset': dataset_full,
            'strategy': 'urial',
            'seed': 1,
            'lr': None,
            'expected_count': 1,
        })

        expected_jobs.append({
            'model': model,
            'dataset': dataset_full,
            'strategy': 'baseline',
            'seed': 1,
            'lr': None,
            'expected_count': 1,
        })

print(f"Total expected job configurations: {len(expected_jobs)}")
print(f"Total results in jsonl: {len(df)}")

missing_jobs = []
found_jobs = []
partial_jobs = []

for expected in expected_jobs:
    mask = (df['model_name'] == expected['model']) & \
           (df['dataset_name'] == expected['dataset']) & \
           (df['experiment_name'] == expected['strategy']) & \
           (df['seed'] == expected['seed'])

    if expected.get('lr') is not None:
        mask &= (df['strategy_hparams'].apply(lambda x: abs(x.get('lr', -1) - expected['lr']) < 1e-10))

    matching = df[mask]

    if expected['strategy'] == 'online_coding':
        epochs_per_config = 3
        expected_epochs = set(range(epochs_per_config))

        matching_with_max_ex = matching[
            (matching['strategy_hparams'].apply(lambda x: x.get('max_examples') is not None)) &
            (matching['strategy_hparams'].apply(lambda x: x.get('current_epoch', -1) in expected_epochs))
        ]

        unique_max_examples = set(matching_with_max_ex['strategy_hparams'].apply(
            lambda x: x.get('max_examples')
        ))

        all_epochs_present = True
        for max_ex in unique_max_examples:
            matching_config = matching_with_max_ex[matching_with_max_ex['strategy_hparams'].apply(
                lambda x: x.get('max_examples') == max_ex
            )]
            present_epochs = set(matching_config['strategy_hparams'].apply(lambda x: x.get('current_epoch', -1)).unique())

            if present_epochs != expected_epochs:
                all_epochs_present = False
                break

        actual_count = len(matching_with_max_ex)
        expected_num_configs = len(MAX_EXAMPLES)

        if actual_count == 0:
            missing_jobs.append({**expected, 'actual_count': 0})
        elif not all_epochs_present or len(unique_max_examples) < expected_num_configs or actual_count < expected['expected_count']:
            partial_jobs.append({**expected, 'actual_count': actual_count})
        else:
            found_jobs.append({**expected, 'actual_count': actual_count})
    else:
        actual_count = len(matching)

        if actual_count == 0:
            missing_jobs.append({**expected, 'actual_count': 0})
        elif actual_count < expected['expected_count']:
            partial_jobs.append({**expected, 'actual_count': actual_count})
        else:
            found_jobs.append({**expected, 'actual_count': actual_count})

print(f"\nComplete jobs: {len(found_jobs)}")
print(f"Partial jobs: {len(partial_jobs)}")
print(f"Missing jobs: {len(missing_jobs)}")

if len(missing_jobs) > 0 or len(partial_jobs) > 0:
    print("\n" + "="*80)
    print("MISSING AND PARTIAL JOBS:")
    print("="*80)

    all_incomplete = missing_jobs + partial_jobs
    incomplete_by_config = defaultdict(list)
    for job in all_incomplete:
        key = (job['model'], job['dataset'], job['strategy'])
        incomplete_by_config[key].append(job)

    for (model, dataset, strategy), jobs in sorted(incomplete_by_config.items()):
        print(f"\n{model} + {dataset} + {strategy}:")
        for job in jobs:
            params = f"seed={job['seed']}"
            if job.get('lr') is not None:
                params += f", lr={job['lr']}"
            status = "MISSING" if job['actual_count'] == 0 else f"PARTIAL ({job['actual_count']}/{job['expected_count']})"
            print(f"  {status}: {params}")

print("\n" + "="*80)
print("SUMMARY BY CONFIGURATION:")
print("="*80)

summary_stats = defaultdict(lambda: {'expected': 0, 'found': 0, 'missing': 0, 'partial': 0})

for job in expected_jobs:
    key = (job['model'], job['dataset'], job['strategy'])
    summary_stats[key]['expected'] += 1

for job in found_jobs:
    key = (job['model'], job['dataset'], job['strategy'])
    summary_stats[key]['found'] += 1

for job in missing_jobs:
    key = (job['model'], job['dataset'], job['strategy'])
    summary_stats[key]['missing'] += 1

for job in partial_jobs:
    key = (job['model'], job['dataset'], job['strategy'])
    summary_stats[key]['partial'] += 1

for (model, dataset, strategy), stats in sorted(summary_stats.items()):
    complete_configs = stats['found']
    total_configs = stats['expected']
    completion_rate = 100 * complete_configs / total_configs if total_configs > 0 else 0
    status = "✓" if stats['missing'] == 0 and stats['partial'] == 0 else "✗"

    details = f"{complete_configs}/{total_configs}"
    if stats['partial'] > 0:
        details += f" ({stats['partial']} partial)"
    if stats['missing'] > 0:
        details += f" ({stats['missing']} missing)"

    print(f"{status} {model:30s} + {dataset:60s} + {strategy:15s}: {details:25s} ({completion_rate:5.1f}%)")

print("\n" + "="*80)
print("DETAILED BREAKDOWN FOR ONLINE_CODING:")
print("="*80)

for model in models:
    for dataset in datasets_short:
        dataset_full = DATASET_MAPPING[dataset]
        print(f"\n{model} + {dataset_full}:")

        for seed in SEEDS:
            for lr in LR_ONLINE_CODING:
                mask = (df['model_name'] == model) & \
                       (df['dataset_name'] == dataset_full) & \
                       (df['experiment_name'] == 'online_coding') & \
                       (df['seed'] == seed) & \
                       (df['strategy_hparams'].apply(lambda x: abs(x.get('lr', -1) - lr) < 1e-10))

                matching = df[mask]
                expected_count = len(MAX_EXAMPLES) * 3
                actual_count = len(matching)

                epochs_per_config = 3
                expected_epochs = set(range(epochs_per_config))

                matching_with_max_ex = matching[
                    (matching['strategy_hparams'].apply(lambda x: x.get('max_examples') is not None)) &
                    (matching['strategy_hparams'].apply(lambda x: x.get('current_epoch', -1) in expected_epochs))
                ]

                unique_max_examples = set(matching_with_max_ex['strategy_hparams'].apply(
                    lambda x: x.get('max_examples')
                ))

                all_epochs_present = True
                missing_info = []

                for max_ex in sorted(unique_max_examples):
                    matching_config = matching_with_max_ex[matching_with_max_ex['strategy_hparams'].apply(
                        lambda x: x.get('max_examples') == max_ex
                    )]
                    present_epochs = set(matching_config['strategy_hparams'].apply(lambda x: x.get('current_epoch', -1)).unique())

                    if present_epochs != expected_epochs:
                        all_epochs_present = False
                        missing_epochs = expected_epochs - present_epochs
                        if missing_epochs:
                            missing_info.append(f"max_ex={max_ex}: missing epochs {sorted(missing_epochs)}")

                actual_count_with_max_ex = len(matching_with_max_ex)
                expected_num_configs = len(MAX_EXAMPLES)
                status = "✓" if actual_count_with_max_ex == expected_count and all_epochs_present and len(unique_max_examples) == expected_num_configs else "✗"
                print(f"  {status} seed={seed}, lr={lr}: {actual_count_with_max_ex}/{expected_count} jobs ({len(unique_max_examples)}/{expected_num_configs} configs)")
                if missing_info:
                    for info in missing_info:
                        print(f"      {info}")
