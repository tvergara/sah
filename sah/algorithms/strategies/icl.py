import math
from decimal import Decimal

import torch
from torch.utils.data import DataLoader, Dataset

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.arithmetic_coding import encode


class StopTrainingError(Exception):
    pass


class ICLStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.total_samples = 0
        self.prompt = ""
        self.updated = True

    def setup(self, pl_module, stage):
        super().setup(pl_module, stage)
        pl_module.model = pl_module.model.eval()

    def on_train_start(self, pl_module):
        self.bits = 0
        return super().on_train_start(pl_module)

    def training_step(self, pl_module, batch, batch_idx):
        text = batch["text"][0]
        new_prompt = self.prompt + text
        encoded = pl_module.tokenizer.encode(new_prompt, add_special_tokens=False)
        pl_module.log(
            "train/context_length",
            float(len(encoded)),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            reduce_fx="max",
            batch_size=1,
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

        prepared_batch = {
            **batch,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        metrics = self.dataset_handler.validate_batch(pl_module, prepared_batch, batch_idx)
        batch_size = len(questions)

        for key, value in metrics.items():
            if key in ["correct_count", "total_count"]:
                pl_module.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum", batch_size=batch_size)
            else:
                pl_module.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def train_dataloader(self, pl_module):
        base_dataset = self.dataset_handler.get_train_dataset()
        train_dataset = ICLTrainDataset(pl_module.tokenizer, base_dataset)
        return DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
        )

    def val_dataloader(self, pl_module):
        raw_val_data = self.dataset_handler.get_raw_val_data()
        dataset = ICLValidationDataset(raw_val_data)
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
    def __init__(self, tokenizer, base_dataset):
        self.tokenizer = tokenizer
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        input_ids = item["input_ids"]

        text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        context = text + "\n\n"
        return {"text": context}


class ICLValidationDataset(Dataset):
    def __init__(self, raw_val_data):
        self.data = raw_val_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
