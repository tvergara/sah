import glob
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import pandas as pd
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
    result_file: str
    batch_size: int = 32
    test_size: float = 0.2
    dim_size: int = 50

class ActivationDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        first_input: str | None,
        second_input: str | None,
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
            glob_pat = os.path.join(base_path, f"{first_input}/layer{layer}_batch*.pt")
            self.first_inputs = _load_acts(glob_pat)                # [N, L, D]
            assert self.first_inputs.shape[:-1] == self.tokens.shape
        else:
            self.first_inputs = torch.zeros_like(self.tokens)

        # ── second input (notice the correct variable) ─────────────
        if second_input:
            glob_pat = os.path.join(base_path, f"{second_input}/layer{layer}_batch*.pt")
            self.second_inputs = _load_acts(glob_pat)               # [N, L, D]
            assert self.second_inputs.shape[:-1] == self.tokens.shape
        else:
            self.second_inputs = torch.zeros_like(self.tokens)

    def __getitem__(self, idx):
        return (self.tokens[idx], self.masks[idx], self.first_inputs[idx], self.second_inputs[idx])

    def __len__(self):
        return self.tokens.size(0)


def _load_acts(glob_pat):
    paths = sorted(
        glob.glob(glob_pat),
        key=lambda p: int(os.path.basename(p).split("_")[1].removeprefix("batch").removesuffix(".pt"))
    )
    return torch.cat([torch.load(p)["activations"] for p in paths], dim=0)

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
    second_input: str | None = None

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
        self.use_first  = self.activations_config.first_input  is not None
        self.use_second = self.activations_config.second_input is not None
        vocab_size      = len(self.tokenizer)

        if self.use_first:                       # activation-conditioned model
            in_dim = self.general_config.dim_size * (2 if self.use_second else 1)
            self.network = nn.Linear(in_dim, vocab_size, bias=True)
            self.learned_logits = False
        else:                                    # global-unigram model
            self.network = nn.Parameter(torch.zeros(vocab_size))
            self.learned_logits = True

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
        tokens, mask, first_acts, second_acts = batch
        targets = tokens[:, 1:]
        mask    = mask[:, 1:]
        keep    = mask.bool()

        if self.learned_logits:                               # unigram baseline
            tgt_flat   = targets[keep]
            logits_exp = self.network.unsqueeze(0).expand(tgt_flat.size(0), -1)
            loss       = F.cross_entropy(logits_exp, tgt_flat, reduction="mean")

        else:                                                 # activation model
            acts = first_acts[:, :-1, :]                      # [B, L-1, D]
            if self.use_second:
                acts = torch.cat([acts, second_acts[:, :-1, :]], dim=-1)  # [B, L-1, 2D]

            feats_flat = acts[keep]       # (N, D or 2D)
            tgt_flat   = targets[keep]    # (N,)
            logits     = self.network(feats_flat)
            loss       = F.cross_entropy(logits, tgt_flat, reduction="mean")

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        tokens, mask, first_acts, second_acts = batch
        targets = tokens[:, 1:]
        mask    = mask[:, 1:]
        keep    = mask.bool()

        if self.learned_logits:
            tgt_flat = targets[keep]
            log_probs = F.log_softmax(self.network, dim=0)          # (V,)
            nll = -log_probs[tgt_flat].mean()

        else:
            acts = first_acts[:, :-1, :]
            if self.use_second:
                acts = torch.cat([acts, second_acts[:, :-1, :]], dim=-1)

            feats_flat = acts[keep]
            tgt_flat   = targets[keep]
            log_probs  = F.log_softmax(self.network(feats_flat), dim=-1)  # (N, V)
            nll = -log_probs[torch.arange(log_probs.size(0)), tgt_flat].mean()

        self.log("test/loss", nll, on_epoch=True, prog_bar=True)
        return nll

    def on_test_epoch_end(self):
        avg_ent = self.trainer.callback_metrics["test/loss"].item()
        self.save_entropy(avg_ent)

    def save_entropy(self, entropy):
        result_path = Path(self.general_config.result_file)
        first_input = self.activations_config.first_input
        second_input = self.activations_config.second_input
        row = {"first_input": first_input, "second_input": second_input, "entropy": entropy}

        def _match(col, value):
            if value is None:
                return df[col].isna()
            return df[col] == value

        if result_path.exists():
            df = pd.read_csv(result_path)

            mask = _match("first_input", row["first_input"]) & _match("second_input", row["second_input"])

            if mask.any():
                df.loc[mask, "entropy"] = entropy
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])


        df.to_csv(result_path, index=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)
