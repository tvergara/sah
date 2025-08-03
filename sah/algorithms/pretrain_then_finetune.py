from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import pandas as pd
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler

from sah.algorithms.networks.transformer import TransformerConfig
from sah.algorithms.utils import GrammarConfig, TokenizerConfig, collate


@dataclass(frozen=True, unsafe_hash=True)
class GeneralConfig:
    id: str
    result_file: str

class PretrainThenFinetune(LightningModule):
    def __init__(
        self,
        finetuning_config: GrammarConfig,
        pretraining_config: GrammarConfig,
        tokenizer_config: TokenizerConfig,
        transformer_config: TransformerConfig,
        general_config: GeneralConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.grammar_config = finetuning_config
        self.tokenizer_config = tokenizer_config
        self.general_config = general_config
        self.finetuning_dataset = hydra_zen.instantiate(finetuning_config)
        self.pretraining_dataset = hydra_zen.instantiate(pretraining_config)
        self.tokenizer = self.finetuning_dataset.tokenizer
        self.test_dataset = hydra_zen.instantiate(finetuning_config, mode='test', tokenizer=self.tokenizer)
        self.pad  = self.finetuning_dataset.pad_id
        self.transformer = hydra_zen.instantiate(
            transformer_config,
            vocab_size = self.tokenizer.vocab_size,
        )
        self.dataset = ConcatDataset([self.pretraining_dataset, self.finetuning_dataset])

    def training_step(self, batch, batch_idx):
        x, mask = batch                 # x : (B, L)
        inputs  = x[:, :-1]
        targets = x[:, 1:]

        logits = self.transformer(inputs)
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.reshape(-1),
            reduction='mean',
            ignore_index=self.pad
        )

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, mask = batch                 # (B, L)
        inputs  = x[:, :-1]
        targets = x[:, 1:]
        logits = self.transformer(inputs)  # (B, L-1, V)
        target_mask = mask[:, 1:]

        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.reshape(-1),
            reduction='mean',
            ignore_index=self.pad
        )
        self.log("test/loss", loss, prog_bar=True)

        preds    = logits.argmax(-1)
        correct  = ((preds == targets) & (target_mask.bool())).sum()
        n_tokens = target_mask.sum()
        acc = correct.float() / n_tokens
        self.log("accuracy", acc, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        avg_ent = self.trainer.callback_metrics["test/loss"].item()
        self.save_entropy(avg_ent)

    def save_entropy(self, entropy):
        result_path = Path(self.general_config.result_file)
        row = {
            "entropy": entropy,
            "id": self.general_config.id
        }

        def _match(col, value):
            if value is None:
                return df[col].isna()
            return df[col] == value

        if result_path.exists():
            df = pd.read_csv(result_path)

            mask = _match("id", row["id"])

            if mask.any():
                df.loc[mask, "entropy"] = entropy
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])


        df.to_csv(result_path, index=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=32,
            sampler=SequentialSampler(self.dataset),
            collate_fn=collate
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, collate_fn=collate)
