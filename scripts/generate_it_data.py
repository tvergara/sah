import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import scripts.if_functions as if_functions

model_path = "allenai/Olmo-3-7B-Instruct"

print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

dataset = load_dataset("allenai/RLVR-IFeval", split="train")

print(f"Total examples: {len(dataset)}")
print(f"\nDataset columns: {dataset.column_names}")
print("\nFirst example:")

batch_size = 4

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
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoding_starts = inputs['input_ids'].shape[1]

    for j, gen_tokens in enumerate(generated):
        response = tokenizer.decode(gen_tokens[decoding_starts:], skip_special_tokens=True)


        config = json.loads(ground_truth[j])
        func_name = config['func_name']

        kwargs = {k: v for k, v in config.items() if k != 'func_name' and v is not None}

        validation_func = getattr(if_functions, func_name)

        is_correct = validation_func(response, **kwargs)

        if is_correct:
            correct_examples.append({
                'messages': messages_list[j],
                'response': response
            })

print(f"\nTotal correct examples: {len(correct_examples)}")


output_file = Path("/network/scratch/b/brownet/correct_ifeval_examples.jsonl")
with open(output_file, 'w') as f:
    for example in correct_examples:
        user_message = example['messages'][0]['content']
        formatted_example = {
            "conversations": [user_message, example['response']]
        }
        f.write(json.dumps(formatted_example) + '\n')

print(f"Saved to {output_file}")
