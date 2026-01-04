import json
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from .base import BaseDatasetHandler, GenerationValDataset


class IFEvalTrainDataset(Dataset):
    def __init__(self, tokenizer, data_path, block_size=1548, max_examples=None):
        self.examples = []

        with open(data_path) as f:
            lines = f.readlines()
            if max_examples:
                lines = lines[:max_examples]

        for line in tqdm(lines):
            data = json.loads(line)
            conversations = data['conversations']
            question = conversations[0]
            answer = conversations[1]

            question_text = f"<|im_start|>user\n{question}\n<|im_start|>assistant"
            answer_text = f"\n{answer}\n<|im_start|>user"

            question_ids = tokenizer.encode(question_text, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

            if tokenizer.bos_token_id is not None:
                full_ids = [tokenizer.bos_token_id] + question_ids + answer_ids
                question_length = 1 + len(question_ids)
            else:
                full_ids = question_ids + answer_ids
                question_length = len(question_ids)

            if len(full_ids) > block_size:
                full_ids = full_ids[:block_size]

            if len(full_ids) > question_length:
                labels = full_ids.copy()
                labels[:question_length] = [-100] * question_length

                self.examples.append({
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class IFEvalHandler(BaseDatasetHandler):
    def __init__(self, tokenizer, dataset_name, block_size=1548, max_examples=None, generations_dir=None):
        super().__init__(tokenizer, dataset_name, block_size, max_examples, generations_dir)
        self.prompt_to_ifeval = {}
        self.used_ifeval_data = []

    def format_example(self, example):
        conversations = example['conversations']
        question = conversations[0]
        answer = conversations[1]

        return {
            "question": f"<|im_start|>user\n{question}\n<|im_start|>assistant",
            "answer": f"\n{answer}\n<|im_start|>user"
        }

    def get_train_dataset(self):
        data_path = self.dataset_name.replace("ifeval:", "")
        return IFEvalTrainDataset(
            self.tokenizer,
            data_path,
            self.block_size,
            self.max_examples
        )

    def _load_ifeval_data(self):
        raw_dataset = load_dataset("google/IFEval", split="train")
        self.ifeval_data = []
        self.prompt_to_ifeval.clear()
        self.used_ifeval_data.clear()

        for item in raw_dataset:
            prompt = item['prompt']
            ifeval_entry = {
                "key": item['key'],
                "prompt": item['prompt'],
                "instruction_id_list": item['instruction_id_list'],
                "kwargs": item['kwargs']
            }
            self.ifeval_data.append(ifeval_entry)
            self.prompt_to_ifeval[prompt] = ifeval_entry

    def get_val_dataset(self):
        self._load_ifeval_data()
        prompts = [f"<|im_start|>user\n{item['prompt']}\n<|im_start|>assistant" for item in self.ifeval_data]

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.block_size,
            padding_side='left',
        )

        return GenerationValDataset(tokenized['input_ids'], tokenized['attention_mask'], prompts)

    def get_raw_val_data(self):
        if self.validation_data is None:
            self._load_ifeval_data()
            prompts = [item['prompt'] for item in self.ifeval_data]

            self.validation_data = [
                {"question": f"<|im_start|>user\n{q}\n<|im_start|>assistant", "expected_answer": "", "raw_prompt": q}
                for q in prompts
            ]

        return self.validation_data

    def validate_batch(self, pl_module, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        prompts = batch.get('question', batch.get('expected_answer'))
        raw_prompts = batch.get('raw_prompt', prompts)

        with torch.no_grad():
            generated = pl_module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=pl_module.max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoding_starts = input_ids.shape[1]
        for i, gen_tokens in enumerate(generated):
            decoded = self.tokenizer.decode(gen_tokens[decoding_starts:], skip_special_tokens=True)
            raw_prompt = raw_prompts[i] if isinstance(raw_prompts, list) else prompts[i]

            if "<im_start>" in decoded:
                decoded = decoded[:decoded.index("<im_start>")]

            self.generations.append({
                "prompt": raw_prompt,
                "response": decoded
            })

            ifeval_entry = self.prompt_to_ifeval[raw_prompt]
            if ifeval_entry not in self.used_ifeval_data:
                self.used_ifeval_data.append(ifeval_entry)

        total_count = len(generated)
        return {"performance": 0.0, "correct_count": 0.0, "total_count": float(total_count)}

    def save_generations(self, run_id):
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        if self.generations_dir is None:
            return

        experiment_dir = Path(self.generations_dir) / run_id
        experiment_dir.mkdir(parents=True, exist_ok=True)

        input_data_file = experiment_dir / "input_data.jsonl"
        with open(input_data_file, 'w') as f:
            for item in self.used_ifeval_data:
                filtered_item = item.copy()
                if 'kwargs' in filtered_item and isinstance(filtered_item['kwargs'], list):
                    filtered_item['kwargs'] = [
                        {k: v for k, v in kwarg.items() if v is not None}
                        for kwarg in filtered_item['kwargs']
                    ]
                f.write(json.dumps(filtered_item) + '\n')

        responses_file = experiment_dir / "responses.jsonl"
        with open(responses_file, 'w') as f:
            for item in self.generations:
                f.write(json.dumps(item) + '\n')

        self.generations.clear()
        self.used_ifeval_data.clear()
