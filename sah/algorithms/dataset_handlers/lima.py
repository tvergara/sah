import json
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset

from .base import BaseDatasetHandler, GenerationValDataset, ProcessedTrainDataset


class LimaHandler(BaseDatasetHandler):
    def __init__(self, tokenizer, dataset_name, block_size=1548, max_examples=None, generations_dir=None):
        super().__init__(tokenizer, dataset_name, block_size, max_examples, generations_dir)
        self.prompt_to_ifeval = {}
        self.used_ifeval_data = []

    def format_example(self, example):
        conversations = example['conversations']
        question = conversations[0]
        answer = conversations[1]

        return {
            "question": f"Instruction: {question}\nAnswer:",
            "answer": f" {answer}"
        }

    def get_train_dataset(self):
        return ProcessedTrainDataset(
            self.tokenizer,
            "GAIR/lima",
            self.format_example,
            self.block_size,
            max_examples=self.max_examples
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
        prompts = [item['prompt'] for item in self.ifeval_data]

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
                {"question": q, "expected_answer": ""}
                for q in prompts
            ]

        return self.validation_data

    def validate_batch(self, pl_module, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        prompts = batch.get('question', batch.get('expected_answer'))

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
            prompt = prompts[i]

            self.generations.append({
                "prompt": prompt,
                "response": decoded
            })

            if prompt in self.prompt_to_ifeval:
                ifeval_entry = self.prompt_to_ifeval[prompt]
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
