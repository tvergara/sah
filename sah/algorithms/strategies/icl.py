import math
import re
from decimal import Decimal

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from sah.algorithms.formatters import get_dataset_formatter
from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.arithmetic_coding import encode


class ICLStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.total_samples = 0
        self.prompt = ""
        self.updated = True

    def setup(self, pl_module, stage):
        pl_module.model = pl_module.model.eval()

    def on_train_start(self, pl_module):
        self.bits = 0
        return super().on_train_start(pl_module)

    def training_step(self, pl_module, batch, batch_idx):
        # sample = batch["input_ids"][0]
        new_prompt = self.prompt + batch[0]
        encoded = pl_module.tokenizer.encode(new_prompt, add_special_tokens=False)
        if len(encoded) > pl_module.max_length:
            return None
        pl_module.log(
            "train/context_length",
            float(len(encoded)),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            reduce_fx="max",
        )
        self.prompt = new_prompt
        self.updated = True
        self.total_samples += 1
        with torch.no_grad():
            compressed_batch = encode(pl_module.model, encoded)
        total_bits = 0
        log_2 = Decimal(2).ln()
        for compressed_value, interval_width in compressed_batch:
            if interval_width > 0:
                bits = math.ceil(float(-(interval_width.ln() / log_2)))
                total_bits += bits
        self.bits = total_bits
        return None

    def validation_step(self, pl_module, batch, batch_idx):
        if not self.updated:
            return
        questions = batch["question"]
        full_prompts = [self.prompt + q for q in questions]
        tokenized = pl_module.tokenizer(
            full_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left",
        )
        input_ids = tokenized["input_ids"].to(pl_module.device)
        attention_mask = tokenized["attention_mask"].to(pl_module.device)
        if input_ids.size(1) > pl_module.max_length:
            pl_module.trainer.should_stop = True
            self.updated = False
            return
        with torch.no_grad():
            max_length = min(pl_module.max_length, input_ids.size(1) + 512)
            generated = pl_module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=False,
                pad_token_id=pl_module.tokenizer.eos_token_id,
            )

        correct_count = 0
        total_count = 0
        context_length = len(self.prompt)
        pattern = r"answer is: ([\d,]+)"

        for i, gen_tokens in enumerate(generated):
            decoded = pl_module.tokenizer.decode(gen_tokens, skip_special_tokens=True)[
                context_length:
            ]

            extracted_answer = ""
            match = re.search(pattern, decoded, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).replace(",", "")

            expected_answer = batch["expected_answer"][i]
            is_correct = extracted_answer == expected_answer

            if is_correct:
                correct_count += 1
            total_count += 1
        pl_module.log(
            "val/correct_count",
            float(correct_count),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="sum",
        )
        pl_module.log(
            "val/total_count",
            float(total_count),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            reduce_fx="sum",
        )

    def train_dataloader(self, pl_module):
        train_dataset = ICLTrainDataset(
            pl_module.tokenizer,
            pl_module.dataset_name,
            max_examples=pl_module.max_examples,
        )
        return DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
        )

    def val_dataloader(self, pl_module):
        dataset = ICLValidationDataset(
            pl_module.tokenizer,
            pl_module.val_dataset_name,
            max_examples=pl_module.max_examples,
        )
        return DataLoader(
            dataset,
            batch_size=pl_module.val_batch_size,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )

    def on_validation_epoch_end(self, pl_module):
        if not self.updated:
            raise StopTrainingError
        self.updated = False
        return super().on_validation_epoch_end(pl_module)


class ICLTrainDataset(Dataset):
    def __init__(self, tokenizer, dataset_name, max_examples=None):
        print("Loading dataset...")
        self.raw_dataset = load_dataset(dataset_name, split="train")
        print(f"Dataset loaded: {len(self.raw_dataset)} examples")  # type: ignore
        print("Processing examples...")
        self.data_length = (
            min(len(self.raw_dataset), max_examples)  # type: ignore
            if max_examples
            else len(self.raw_dataset)  # type: ignore
        )
        self.formatter = get_dataset_formatter(dataset_name)
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        sample = self.formatter(self.raw_dataset[idx])  # type: ignore
        assert sample is not None
        context = (
            "Question: "
            + sample["question"]
            + "\nResponse: "
            + sample["answer"]
            + "\n\n"
        )
        return context


class ICLValidationDataset(Dataset):
    def __init__(self, tokenizer, dataset_name, max_examples=None):
        self.tokenizer = tokenizer
        self.name = dataset_name
        self.answers = []
        # self.prompt = prompt

        raw_dataset = load_dataset(*self.name, split="test")
        prompts = []
        for item in raw_dataset:
            question = item["question"]  # type: ignore
            answer = item["answer"]  # type: ignore

            answer_match = re.search(r"#### ([\d,]+)", answer)
            numerical_answer = (
                answer_match.group(1).replace(",", "") if answer_match else ""
            )

            prompt = f"Question: {question}\nResponse:"
            prompts.append(prompt)
            self.answers.append(numerical_answer)

        self.prompts = prompts

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return {
            "question": self.prompts[idx],
            "expected_answer": self.answers[idx],
        }
