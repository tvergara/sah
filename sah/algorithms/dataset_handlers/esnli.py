import re

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from .base import BaseDatasetHandler, GenerationValDataset

DATA_DIR = "/network/scratch/b/brownet/data/esnli"
TRAIN_FILES = [
    f"{DATA_DIR}/esnli_train_1.csv",
    f"{DATA_DIR}/esnli_train_2.csv",
]
TEST_FILE = f"{DATA_DIR}/esnli_test.csv"

LABEL_MAP = {
    "entailment": "entailment",
    "neutral": "neutral",
    "contradiction": "contradiction",
}


class ESNLIHandler(BaseDatasetHandler):
    def format_example(self, example):
        premise = example["Sentence1"]
        hypothesis = example["Sentence2"]
        label = example["gold_label"]
        explanation = example["Explanation_1"]

        question = (
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Does the premise entail, contradict, or is neutral to the hypothesis? "
            f"Explain your reasoning, then answer."
        )
        answer = f" {explanation} Therefore, the answer is {label}."

        return {"question": question, "answer": answer}

    def format_val_example(self, example):
        premise = example["Sentence1"]
        hypothesis = example["Sentence2"]

        question = (
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Does the premise entail, contradict, or is neutral to the hypothesis? "
            f"Explain your reasoning, then answer."
        )
        return question

    def _load_raw_train(self):
        raw = load_dataset("csv", data_files=TRAIN_FILES, split="train")
        if self.max_examples:
            raw = raw.select(range(min(self.max_examples, len(raw))))
        return raw

    def _load_raw_test(self):
        return load_dataset("csv", data_files=TEST_FILE, split="train")

    def get_train_dataset(self):
        raw_dataset = self._load_raw_train()
        examples = []

        for example in tqdm(raw_dataset):
            if example["gold_label"] not in LABEL_MAP:
                continue

            formatted = self.format_example(example)
            question_text = formatted["question"]
            answer_text = formatted["answer"]

            question_ids = self.tokenizer.encode(question_text, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)

            if self.tokenizer.bos_token_id is not None:
                full_ids = [self.tokenizer.bos_token_id] + question_ids + answer_ids
                question_length = 1 + len(question_ids)
            else:
                full_ids = question_ids + answer_ids
                question_length = len(question_ids)

            if len(full_ids) > self.block_size:
                full_ids = full_ids[: self.block_size]

            if len(full_ids) > question_length:
                labels = full_ids.copy()
                labels[:question_length] = [-100] * question_length

                examples.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": [1] * len(full_ids),
                        "labels": labels,
                    }
                )

        return _ListDataset(examples)

    def get_val_dataset(self):
        raw_dataset = self._load_raw_test()
        prompts = []
        answers = []

        for item in raw_dataset:
            if item["gold_label"] not in LABEL_MAP:
                continue
            prompt = self.format_val_example(item)
            prompts.append(prompt)
            answers.append(item["gold_label"])

        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.block_size,
            padding_side="left",
        )

        return GenerationValDataset(tokenized["input_ids"], tokenized["attention_mask"], answers)

    def get_raw_val_data(self):
        if self.validation_data is None:
            raw_dataset = self._load_raw_test()
            prompts = []
            answers = []

            for item in raw_dataset:
                if item["gold_label"] not in LABEL_MAP:
                    continue
                prompt = self.format_val_example(item)
                prompts.append(prompt)
                answers.append(item["gold_label"])

            self.validation_data = [
                {"question": q, "expected_answer": a} for q, a in zip(prompts, answers)
            ]

        return self.validation_data

    def validate_batch(self, pl_module, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.no_grad():
            generated = pl_module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=pl_module.max_length if hasattr(pl_module, "max_length") else 200,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        correct_count = 0
        total_count = 0
        decoding_starts = batch["input_ids"].shape[1]

        for i, gen_tokens in enumerate(generated):
            decoded = self.tokenizer.decode(gen_tokens[decoding_starts:], skip_special_tokens=True)

            pattern = r"the answer is (\w+)"
            match = re.search(pattern, decoded, re.IGNORECASE)
            extracted_answer = match.group(1).lower().rstrip(".") if match else ""

            expected_answer = batch["expected_answer"][i]
            if extracted_answer == expected_answer:
                correct_count += 1
            total_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0
        return {
            "performance": float(accuracy),
            "correct_count": float(correct_count),
            "total_count": float(total_count),
        }


class _ListDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
