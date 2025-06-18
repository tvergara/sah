
import pickle
from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import pandas as pd
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
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

@dataclass(frozen=True, unsafe_hash=True)
class GeneralConfig:
    result_file: str
    probe_start: int
    id: str
    unconditional_estimate: bool = False
    l2_coef: float = 10.0

class GrammarEntropyBottleneck(LightningModule):
    def __init__(
        self,
        grammar_config: GrammarConfig,
        tokenizer_config: TokenizerConfig,
        checkpoint_config: CheckpointConfig,
        transformer_config: TransformerConfig,
        general_config: GeneralConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.grammar_config = grammar_config
        with open(tokenizer_config.out_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.dataset = hydra_zen.instantiate(self.grammar_config, tokenizer=self.tokenizer)
        self.test_dataset = hydra_zen.instantiate(
            self.grammar_config,
            mode='test',
            tokenizer=self.tokenizer,
            size=None
        )
        self.pad  = self.dataset.pad_id
        self.transformer = hydra_zen.instantiate(
            transformer_config,
            vocab_size = self.tokenizer.vocab_size,
        )
        self.general_config = general_config
        self.transformer.freeze_up_to(general_config.probe_start)
        self.checkpoint_config = checkpoint_config
        load_weights_from_checkpoint(self.transformer, checkpoint_config.path, model_name='transformer')
        self._init_params = {
            n: p.detach().clone()
            for n, p in self.named_parameters()
            if p.requires_grad
        }

        if self.general_config.unconditional_estimate:
            self.logits = nn.Parameter(torch.zeros(self.tokenizer.vocab_size))
        # self.test_step = self.validation_step

    def training_step(self, batch, batch_idx):
        x, _ = batch                 # x : (B, L)
        inputs  = x[:, :-1]             # drop last token
        targets = x[:, 1:]              # predict this

        if self.general_config.unconditional_estimate:
            B, L = targets.shape
            logits = self.logits.expand(B, L, -1)
        else:
            logits = self.transformer(inputs)  # (B, L-1, V)

        ce_loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.reshape(-1),
            reduction='mean',
            ignore_index=self.pad
        )

        if self.general_config.l2_coef > 0 and not self.general_config.unconditional_estimate:
            l2_reg = 0.0
            for name, p in self.named_parameters():
                if not p.requires_grad:
                    continue
                init_p = self._init_params[name].to(p.device)
                l2_reg += (p - init_p).pow(2).sum()
            l2_reg = self.general_config.l2_coef * l2_reg
        else:
            l2_reg = 0.0

        loss = ce_loss + l2_reg
        self.log_dict(
            {
                "train/loss": loss,
                "train/ce_loss": ce_loss,
                "train/l2_reg": l2_reg,
            },
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch                 # (B, L)
        inputs  = x[:, :-1]
        targets = x[:, 1:]

        if self.general_config.unconditional_estimate:
            B, L = targets.shape
            logits = self.logits.expand(B, L, -1)
        else:
            logits = self.transformer(inputs)  # (B, L-1, V)

        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.reshape(-1),
            reduction='mean',
            ignore_index=self.pad,
        )

        self.log("test/loss", loss, prog_bar=True)

        preds = logits.argmax(-1)
        correct = ((preds == targets) & (targets != self.pad)).sum()
        n_tokens = (targets != self.pad).sum()
        acc = correct.float() / n_tokens
        self.log("accuracy", acc, prog_bar=True, sync_dist=True)

    def on_test_epoch_end(self):
        avg_ent = self.trainer.callback_metrics["test/loss"].item()
        self.save_entropy(avg_ent)

    def save_entropy(self, entropy):
        result_path = Path(self.general_config.result_file)
        revision = self.checkpoint_config.path.split('/')[-1]

        row = {
            "revision": revision,
            "probe_start": self.general_config.probe_start,
            "entropy": entropy,
            "unconditional": self.general_config.unconditional_estimate,
            "size": self.grammar_config.size,
            "id": self.general_config.id
        }

        def _match(col, value):
            if value is None:
                return df[col].isna()
            return df[col] == value

        if result_path.exists():
            df = pd.read_csv(result_path)

            mask = _match("revision", row["revision"]) & _match("probe_start", row["probe_start"]) & _match("unconditional", row["unconditional"]) & _match("size", row["size"]) & _match("id", row["id"])

            if mask.any():
                df.loc[mask, "entropy"] = entropy
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])


        df.to_csv(result_path, index=False)

    def configure_optimizers(self):
        trainable_params = (p for p in self.parameters() if p.requires_grad)
        return torch.optim.AdamW(trainable_params, lr=1e-3)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, collate_fn=collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, collate_fn=collate)
