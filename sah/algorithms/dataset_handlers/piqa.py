import json

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .base import BaseDatasetHandler, ProcessedTrainDataset


class PiQAValDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels, sol1, sol2, expected_answers):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.sol1 = sol1
        self.sol2 = sol2
        self.expected_answers = expected_answers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': self.labels[idx],
            'sol1': self.sol1[idx],
            'sol2': self.sol2[idx],
            'expected_answer': self.expected_answers[idx],
        }


class PiQAHandler(BaseDatasetHandler):
    def __init__(self, tokenizer, dataset_name="ybisk/piqa", block_size=1548, max_examples=None):
        super().__init__(tokenizer, dataset_name, block_size, max_examples)

    def format_example(self, example):
        goal = example['goal']
        sol1 = example['sol1']
        sol2 = example['sol2']
        label = example['label']

        answer = sol1 if label == 0 else sol2

        return {
            "question": f"Question: {goal}\nAnswer:",
            "answer": f" {answer}"
        }

    def get_train_dataset(self):
        split_str = f"train[:{self.max_examples}]" if self.max_examples else "train"

        return ProcessedTrainDataset(
            self.tokenizer,
            "ybisk/piqa",
            self.format_example,
            self.block_size,
            max_examples=None,
            split_str=split_str,
        )

    def get_val_dataset(self):
        raw_dataset = load_dataset("allegrolab/testset_piqa", split="test")
        prompts = []
        labels = []
        sol1_texts = []
        sol2_texts = []
        expected_answers = []

        for item in raw_dataset:
            meta = json.loads(item['meta'])

            goal = meta['goal']
            sol1 = meta['sol1']
            sol2 = meta['sol2']
            label = meta['label']

            prompt = f"Question: {goal}\nAnswer:"
            prompts.append(prompt)
            labels.append(label)
            sol1_texts.append(sol1)
            sol2_texts.append(sol2)
            expected_answers.append(sol1 if label == 0 else sol2)

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.block_size,
            padding_side='left',
        )

        return PiQAValDataset(
            tokenized['input_ids'],
            tokenized['attention_mask'],
            labels,
            sol1_texts,
            sol2_texts,
            expected_answers
        )

    def get_raw_val_data(self):
        if self.validation_data is None:
            raw_dataset = load_dataset("allegrolab/testset_piqa", split="test")
            prompts = []
            labels = []
            sol1_texts = []
            sol2_texts = []
            expected_answers = []

            for item in raw_dataset:
                meta = json.loads(item['meta'])

                goal = meta['goal']
                sol1 = meta['sol1']
                sol2 = meta['sol2']
                label = meta['label']

                prompt = f"Question: {goal}\nAnswer:"
                prompts.append(prompt)
                labels.append(label)
                sol1_texts.append(sol1)
                sol2_texts.append(sol2)
                expected_answers.append(sol1 if label == 0 else sol2)

            self.validation_data = [
                {"question": q, "label": lbl, "sol1": s1, "sol2": s2, "expected_answer": a}
                for q, lbl, s1, s2, a in zip(prompts, labels, sol1_texts, sol2_texts, expected_answers)
            ]

        return self.validation_data

    def validate_batch(self, pl_module, batch, batch_idx):
        input_ids = batch['input_ids']

        correct_count = 0
        total_count = 0

        with torch.no_grad():
            for i in range(len(batch['label'])):
                prompt_ids = input_ids[i:i+1]

                sol1_text = batch['sol1'][i]
                sol2_text = batch['sol2'][i]
                sol1_ids = torch.tensor(self.tokenizer.encode(" " + sol1_text, add_special_tokens=False), device=prompt_ids.device)
                sol2_ids = torch.tensor(self.tokenizer.encode(" " + sol2_text, add_special_tokens=False), device=prompt_ids.device)

                full_ids_sol1 = torch.cat([prompt_ids[0], sol1_ids])
                full_ids_sol2 = torch.cat([prompt_ids[0], sol2_ids])

                full_ids_sol1 = full_ids_sol1.unsqueeze(0)
                full_ids_sol2 = full_ids_sol2.unsqueeze(0)

                outputs_sol1 = pl_module.model(input_ids=full_ids_sol1, labels=full_ids_sol1)
                outputs_sol2 = pl_module.model(input_ids=full_ids_sol2, labels=full_ids_sol2)

                nll_sol1 = outputs_sol1.loss * len(sol1_ids)
                nll_sol2 = outputs_sol2.loss * len(sol2_ids)

                predicted_label = 0 if nll_sol1 < nll_sol2 else 1
                expected_label = batch['label'][i]

                if predicted_label == expected_label:
                    correct_count += 1
                total_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return {
            "performance": float(accuracy),
            "correct_count": float(correct_count),
            "total_count": float(total_count)
        }
