import json

input_path = '/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results-filtered.jsonl'
output_path = '/network/scratch/b/brownet/hydra-runs/finetune-with-strategy/final-results-clean.jsonl'

with open(input_path) as f_in, open(output_path, 'w') as f_out:
    for line in f_in:
        data = json.loads(line)
        if 'strategy_hparams' in data and isinstance(data['strategy_hparams'], dict) and 'prompt' in data['strategy_hparams']:
            del data['strategy_hparams']['prompt']
        if 'strategy_hparams' in data and isinstance(data['strategy_hparams'], dict) and 'dataset_handler' in data['strategy_hparams']:
            del data['strategy_hparams']['dataset_handler']
        if 'strategy_hparams' in data and isinstance(data['strategy_hparams'], dict) and 'diffs' in data['strategy_hparams']:
            del data['strategy_hparams']['diffs']
        if 'strategy_hparams' in data and isinstance(data['strategy_hparams'], dict) and 'lora_params' in data['strategy_hparams']:
            del data['strategy_hparams']['lora_params']
        if 'eval_run_id' in data:
            del data['eval_run_id' ]
        if 'experiment_id' in data:
            del data['experiment_id' ]
        f_out.write(json.dumps(data) + '\n')

print('Done - wrote to', output_path)
