import re

import torch
from torch.utils.data import DataLoader

from sah.algorithms.utils.data_collator import DataCollatorForAnswerOnlyLM
from sah.algorithms.utils.processed_dataset import ProcessedDataset
from sah.algorithms.utils.processed_validation_dataset import ProcessedValidationDataset


class BaseStrategy:
    def __init__(self):
        self.bits = 0

    def setup(self, pl_module, stage):
        pass

    def on_train_start(self, pl_module):
        pass

    def on_train_batch_start(self, pl_module, batch, batch_idx):
        pass

    def training_step(self, pl_module, batch, batch_idx):
        outputs = pl_module.model(**batch)
        loss = outputs.loss
        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, pl_module, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.no_grad():
            generated = pl_module.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=pl_module.max_length,
                do_sample=False,
                pad_token_id=pl_module.tokenizer.eos_token_id
            )

        correct_count = 0
        total_count = 0

        for i, gen_tokens in enumerate(generated):
            decoded = pl_module.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            pattern = r'answer is: ([\d,]+)'

            extracted_answer = ""
            match = re.search(pattern, decoded, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).replace(',', '')

            expected_answer = batch['expected_answer'][i]
            is_correct = extracted_answer == expected_answer

            if is_correct:
                correct_count += 1
            total_count += 1

        pl_module.log("val/correct_count", float(correct_count), on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
        pl_module.log("val/total_count", float(total_count), on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")

    def on_validation_epoch_end(self, pl_module):
        correct = pl_module.trainer.logged_metrics.get("val/correct_count", 0)
        total = pl_module.trainer.logged_metrics.get("val/total_count", 0)

        if total > 0:
            accuracy = correct / total
            pl_module.log("val/accuracy", accuracy, prog_bar=True)
            pl_module.log("val/bits", self.bits, prog_bar=True)

    def configure_optimizers(self, pl_module):
        return None

    def on_train_batch_end(self, pl_module, outputs, batch, batch_idx):
        pass

    def train_dataloader(self, pl_module):
        dataset = ProcessedDataset(pl_module.tokenizer, pl_module.dataset_name, max_examples=pl_module.max_examples)
        data_collator = DataCollatorForAnswerOnlyLM(tokenizer=pl_module.tokenizer)

        return DataLoader(
            dataset,
            batch_size=pl_module.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=data_collator,
        )

    def val_dataloader(self, pl_module):
        dataset = ProcessedValidationDataset(pl_module.tokenizer, pl_module.val_dataset_name, max_examples=pl_module.max_examples)

        return DataLoader(
            dataset,
            batch_size=pl_module.val_batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
