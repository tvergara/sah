
import hydra_zen
import torch
from datasets import load_dataset
from lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.formatters import get_dataset_formatter
from sah.algorithms.llm_finetuning import NetworkConfig, TokenizerConfig


class LimaDataset(Dataset):
    def __init__(self, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size

        raw_dataset = load_dataset("GAIR/lima", split="train")
        formatter = get_dataset_formatter("GAIR/lima")

        self.examples = []
        for example in raw_dataset:
            formatted = formatter(example)
            # Tokenize the text
            tokenized = tokenizer(
                formatted["text"],
                truncation=True,
                padding=False,
                max_length=block_size,
                return_tensors="pt"
            )

            if len(tokenized["input_ids"][0]) > 1:
                self.examples.append({
                    "input_ids": tokenized["input_ids"][0],
                    "attention_mask": tokenized["attention_mask"][0]
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class FinetuneOnLima(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        pretrained_config: NetworkConfig,
        batch_size: int = 8,
        block_size: int = 512
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = hydra_zen.instantiate(tokenizer_config)
        self.model = hydra_zen.instantiate(pretrained_config, torch_dtype=torch.bfloat16)
        self.batch_size = batch_size
        self.block_size = block_size

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def train_dataloader(self):
        dataset = LimaDataset(self.tokenizer, self.block_size)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer
