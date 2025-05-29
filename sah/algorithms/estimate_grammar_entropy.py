import glob
import os
import pickle
from dataclasses import dataclass

import hydra_zen
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from sah.algorithms.utils import (
    TokenizerConfig,
)


@dataclass(frozen=True, unsafe_hash=True)
class GeneralConfig:
    batch_size: int = 32
    test_size: float = 0.2
    dim_size: int = 50

class ActivationDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        first_input: str | None,
        layer: int | None,
    ):
        super().__init__()

        out_glob = os.path.join(base_path, "tokens/tokens_batch*.pt")
        out_paths = sorted(
            glob.glob(out_glob),
            key=lambda p: int(os.path.basename(p).split("_")[1].removeprefix("batch").removesuffix(".pt"))
        )
        all_tokens = []
        masks = []
        for path in out_paths:
            values = torch.load(path)
            tokens = values['input']
            mask = values['mask']
            all_tokens.append(tokens)
            masks.append(mask)
        self.tokens = torch.cat(all_tokens, dim=0)
        self.masks = torch.cat(masks, dim=0)

        assert self.tokens.shape == self.masks.shape

        if first_input:
            out_glob = os.path.join(base_path, f"{first_input}/layer{layer}_batch*.pt")
            out_paths = sorted(
                glob.glob(out_glob),
                key=lambda p: int(os.path.basename(p).split("_")[1].removeprefix("batch").removesuffix(".pt"))
            )
            all_activations = []
            for path in out_paths:
                values = torch.load(path)
                activations = values['activations']
                all_activations.append(activations)
            self.first_inputs = torch.cat(all_activations, dim=0)
            assert self.first_inputs.shape[:-1] == self.tokens.shape
        else:
            self.first_inputs = torch.zeros_like(self.tokens)


    def __len__(self):
        return self.tokens.size(0)

    def __getitem__(self, idx):
        return self.tokens[idx], self.masks[idx], self.first_inputs[idx]

@hydra_zen.hydrated_dataclass(
    target=ActivationDataset,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class ActivationsConfig:
    base_path: str
    layer: int | None = None
    first_input: str | None = None

class GrammarEntropyEstimator(LightningModule):
    def __init__(
        self,
        activations_config: ActivationsConfig,
        general_config: GeneralConfig,
        tokenizer_config: TokenizerConfig,
    ) -> None:
        super().__init__()
        self.activations_config = activations_config
        self.general_config = general_config
        with open(tokenizer_config.out_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        self.learned_logits = False
        if self.activations_config.first_input:
            self.network = nn.Linear(
                self.general_config.dim_size,
                len(self.tokenizer),
                bias=True
            )
        else:
            self.learned_logits = True
            self.network = nn.Parameter(torch.zeros(self.tokenizer.vocab_size))

    def setup(self, stage: str):
        if stage == "fit":
            full_ds = hydra_zen.instantiate(self.activations_config)
            n = len(full_ds)
            test_n = int(self.general_config.test_size * n)
            train_n = n - test_n
            self.train_ds, self.test_ds = random_split(
                full_ds,
                [train_n, test_n],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.general_config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.general_config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def training_step(self, batch, batch_idx):
        if not self.learned_logits:
            tokens, mask, first_acts = batch           # first_acts: [B, L, D]
            targets     = tokens[:, 1:]                # [B, L-1]
            mask        = mask[:, 1:]                  # [B, L-1]
            first_acts  = first_acts[:, :-1, :]        # [B, L-1, D]

            keep        = mask.bool()
            tgt_flat    = targets[keep]                # (N,)
            feats_flat  = first_acts[keep]             # (N, D)

            logits      = self.network(feats_flat)     # (N, V)
            loss        = F.cross_entropy(logits, tgt_flat, reduction="mean")
        else:
            tokens, mask = batch                       # global-logits path
            targets  = tokens[:, 1:]
            mask     = mask[:, 1:]
            tgt_flat = targets[mask.bool()]

            logits_exp = self.network.unsqueeze(0).expand(tgt_flat.size(0), -1)
            loss       = F.cross_entropy(logits_exp, tgt_flat, reduction="mean")

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        if not self.learned_logits:
            tokens, mask, first_acts = batch
            targets     = tokens[:, 1:]
            mask        = mask[:, 1:]
            first_acts  = first_acts[:, :-1, :]

            keep        = mask.bool()
            tgt_flat    = targets[keep]
            feats_flat  = first_acts[keep]

            log_probs = F.log_softmax(self.network(feats_flat), dim=-1)  # (N, V)
            nll       = -log_probs[torch.arange(log_probs.size(0)), tgt_flat].mean()
        else:
            tokens, mask = batch
            targets  = tokens[:, 1:]
            mask     = mask[:, 1:]
            tgt_flat = targets[mask.bool()]

            log_probs = F.log_softmax(self.network, dim=0)               # (V,)
            nll       = -log_probs[tgt_flat].mean()

        self.log("test/loss", nll, on_epoch=True, prog_bar=True)
        return nll

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
