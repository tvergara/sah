import re

import torch
from datasets import load_dataset

from .base import BaseDatasetHandler, GenerationValDataset, ProcessedTrainDataset


class MMLUHandler(BaseDatasetHandler):
    def __init__(self, tokenizer, dataset_name="cais/mmlu", block_size=1548, max_examples=None):
        super().__init__(tokenizer, dataset_name, block_size, max_examples)

    def format_example(self, example):
        question_text = example['question']
        choices = example['choices']
        answer_idx = example['answer']

        formatted_question = f"{question_text}\n"
        choice_letters = ['A', 'B', 'C', 'D']
        for i, choice in enumerate(choices):
            formatted_question += f"{choice_letters[i]}) {choice}\n"

        answer_letter = choice_letters[answer_idx]

        return {
            "question": formatted_question.strip(),
            "answer": answer_letter
        }

    def get_train_dataset(self):
        split_str = f"auxiliary_train[:{self.max_examples}]" if self.max_examples else "auxiliary_train"

        return ProcessedTrainDataset(
            self.tokenizer,
            self.dataset_name,
            self.format_example,
            self.block_size,
            max_examples=None,
            split_str=split_str,
            config_name="all"
        )

    def get_val_dataset(self):
        raw_dataset = load_dataset(self.dataset_name, "all", split="test")
        prompts = []
        answers = []

        for item in raw_dataset:
            formatted = self.format_example(item)
            prompt = f"Question: {formatted['question']}\nAnswer:"
            prompts.append(prompt)
            answers.append(formatted['answer'])

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.block_size,
            padding_side='left',
        )

        return GenerationValDataset(
            tokenized['input_ids'],
            tokenized['attention_mask'],
            answers
        )

    def get_raw_val_data(self):
        if self.validation_data is None:
            raw_dataset = load_dataset(self.dataset_name, "all", split="test")
            prompts = []
            answers = []

            for item in raw_dataset:
                formatted = self.format_example(item)
                prompt = f"Question: {formatted['question']}\nAnswer:"
                prompts.append(prompt)
                answers.append(formatted['answer'])

            self.validation_data = [
                {"question": q, "expected_answer": a}
                for q, a in zip(prompts, answers)
            ]

        return self.validation_data

    def validate_batch(self, pl_module, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            generated = pl_module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=pl_module.max_length if hasattr(pl_module, 'max_length') else 10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        correct_count = 0
        total_count = 0
        decoding_starts = batch['input_ids'].shape[1]

        for i, gen_tokens in enumerate(generated):
            decoded = self.tokenizer.decode(
                gen_tokens[decoding_starts:],
                skip_special_tokens=True
            ).strip()

            pattern = r'\b([A-D])\b'
            match = re.search(pattern, decoded)
            extracted_answer = match.group(1) if match else ""

            expected_answer = batch['expected_answer'][i]
            is_correct = extracted_answer == expected_answer

            if is_correct:
                correct_count += 1
            total_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return {
            "performance": float(accuracy),
            "correct_count": float(correct_count),
            "total_count": float(total_count)
        }
