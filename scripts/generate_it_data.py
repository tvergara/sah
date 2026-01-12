import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

import scripts.if_functions as if_functions

parser = argparse.ArgumentParser()
parser.add_argument("--num_splits", type=int, default=1)
parser.add_argument("--split_index", type=int, default=0)
parser.add_argument("--tensor_parallel_size", type=int, default=1)
args = parser.parse_args()

model_path = "allenai/Olmo-3-32B-Think"
revision = None

print(f"Loading model with vLLM from {model_path}...")
llm = LLM(
    model=model_path,
    revision=revision,
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=0.95,
    max_model_len=5000,
)

tokenizer = llm.get_tokenizer()

dataset = load_dataset("allenai/RLVR-IFeval", split="train")

print(f"Total examples: {len(dataset)}")

total_size = len(dataset)
split_size = (total_size + args.num_splits - 1) // args.num_splits
start_idx = args.split_index * split_size
end_idx = min(start_idx + split_size, total_size)

dataset = dataset.select(range(start_idx, end_idx))

print(f"Processing split {args.split_index + 1}/{args.num_splits}")
print(f"Split range: [{start_idx}, {end_idx}), size: {len(dataset)}")
print(f"\nDataset columns: {dataset.column_names}")

num_candidates = 6

sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=3000,
    n=num_candidates,
)

print("\nFormatting prompts...")
formatted_prompts = []
for messages in tqdm(dataset['messages']):
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    formatted_prompts.append(formatted_prompt)

print(f"\nGenerating responses for {len(formatted_prompts)} prompts...")
outputs = llm.generate(formatted_prompts, sampling_params)

print("\nValidating responses...")
correct_examples = []

for idx, output in enumerate(tqdm(outputs)):
    config = json.loads(dataset['ground_truth'][idx])
    func_name = config['func_name']
    kwargs = {k: v for k, v in config.items() if k != 'func_name' and v is not None}
    validation_func = getattr(if_functions, func_name)

    for completion in output.outputs:
        response = completion.text

        if "</think>" not in response:
            continue
        response = response.split("</think>", 1)[1].strip()

        is_correct = validation_func(response, **kwargs)

        if is_correct:
            correct_examples.append({
                'messages': dataset['messages'][idx],
                'response': response
            })

print(f"\nTotal correct examples: {len(correct_examples)}")

if args.num_splits > 1:
    output_file = Path(f"/network/scratch/b/brownet/correct_ifeval_examples_extended_split_{args.split_index:03d}.jsonl")
else:
    output_file = Path("/network/scratch/b/brownet/correct_ifeval_examples_extended.jsonl")

with open(output_file, 'w') as f:
    for example in correct_examples:
        user_message = example['messages'][0]['content']
        formatted_example = {
            "conversations": [user_message, example['response']]
        }
        f.write(json.dumps(formatted_example) + '\n')

print(f"Saved to {output_file}")
