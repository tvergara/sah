
import pickle
from dataclasses import dataclass

import hydra_zen
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader

from sah.algorithms.networks.transformer import TransformerConfig
from sah.algorithms.utils import (
    GrammarConfig,
    TokenizerConfig,
    collate,
    load_weights_from_checkpoint,
)


@dataclass(frozen=True, unsafe_hash=True)
class CheckpointConfig:
    path: str

class GrammarFinetuner(LightningModule):
    def __init__(
        self,
        grammar_config: GrammarConfig,
        tokenizer_config: TokenizerConfig,
        checkpoint_config: CheckpointConfig,
        transformer_config: TransformerConfig,
        vocab_size: int = 50,
        emb_dim: int = 32,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.save_hyperparameters()
        self.grammar_config = grammar_config
        with open(tokenizer_config.out_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.dataset = hydra_zen.instantiate(self.grammar_config, tokenizer=self.tokenizer)
        self.test_dataset = hydra_zen.instantiate(self.grammar_config, mode='test', tokenizer=self.tokenizer)
        self.pad  = self.dataset.pad_id
        self.transformer = hydra_zen.instantiate(
            transformer_config,
            vocab_size = self.tokenizer.vocab_size,
        )
        load_weights_from_checkpoint(self.transformer, checkpoint_config.path, model_name='transformer')

    def training_step(self, batch, batch_idx):
        x, mask = batch                 # x : (B, L)
        inputs  = x[:, :-1]             # drop last token
        targets = x[:, 1:]              # predict this
        target_mask   = mask[:, 1:]

        logits = self.transformer(inputs)
        token_loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.reshape(-1),
            reduction='none'
        )
        loss = (token_loss * target_mask.reshape(-1)).sum() / target_mask.sum()

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, mask = batch                 # (B, L)
        inputs  = x[:, :-1]
        targets = x[:, 1:]
        logits = self.transformer(inputs)  # (B, L-1, V)
        target_mask = mask[:, 1:]

        token_loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.reshape(-1),
            reduction='none'
        )
        loss = (token_loss * target_mask.reshape(-1)).sum() / target_mask.sum()
        self.log("test/loss", loss, prog_bar=True)

        preds    = logits.argmax(-1)
        correct  = ((preds == targets) & (target_mask.bool())).sum()
        n_tokens = target_mask.sum()
        acc = correct.float() / n_tokens
        self.log("accuracy", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, collate_fn=collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, collate_fn=collate)
