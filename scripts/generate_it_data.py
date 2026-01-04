import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import scripts.if_functions as if_functions

parser = argparse.ArgumentParser()
parser.add_argument("--num_splits", type=int, default=1)
parser.add_argument("--split_index", type=int, default=0)
args = parser.parse_args()

model_path = "allenai/Olmo-3-32B-Think"
revision = None

print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from {model_path}...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    revision=revision,
    quantization_config=quantization_config,
    device_map="auto",
)
model.eval()

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
print("\nFirst example:")

batch_size = 1
num_candidates = 10

device = next(model.parameters()).device

correct_examples = []

for i in tqdm(range(0, len(dataset), batch_size)):
    batch = dataset[i:i + batch_size]
    messages_list = batch['messages']
    ground_truth = batch['ground_truth']

    formatted_prompts = []
    for messages in messages_list:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=2048,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=num_candidates,
            pad_token_id=tokenizer.eos_token_id
        )

    decoding_starts = inputs['input_ids'].shape[1]

    config = json.loads(ground_truth[0])
    func_name = config['func_name']
    kwargs = {k: v for k, v in config.items() if k != 'func_name' and v is not None}
    validation_func = getattr(if_functions, func_name)

    for gen_tokens in generated:
        response = tokenizer.decode(gen_tokens[decoding_starts:], skip_special_tokens=True)

        if "</think>" in response:
            response = response.split("</think>", 1)[1].strip()

        is_correct = validation_func(response, **kwargs)

        if is_correct:
            correct_examples.append({
                'messages': messages_list[0],
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
