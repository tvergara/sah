# activation_collector.py
import pickle
from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import torch
from lightning import LightningModule
from torch.utils.data import DataLoader

from sah.algorithms.networks.transformer import Transformer
from sah.algorithms.networks.utils import listen_to_hidden_activations
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
    output_dir: str
    checkpoint_path: str
    checkpoint_dir: str
    revision: str

class GrammarActivationCollector(LightningModule):
    def __init__(
        self,
        grammar_config: GrammarConfig,
        tokenizer_config: TokenizerConfig,
        general_config: GeneralConfig,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        with open(tokenizer_config.out_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.test_ds = hydra_zen.instantiate(
            grammar_config, mode="test", tokenizer=self.tokenizer
        )
        self.pad_id = self.test_ds.pad_id
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transformer = Transformer(self.tokenizer.vocab_size)
        load_weights_from_checkpoint(
            self.transformer, general_config.checkpoint_path, model_name="transformer"
        )

        self._acts: dict[int, torch.Tensor] = {}
        self._handles = listen_to_hidden_activations(
            self.transformer, self._acts, detach=True, device="cpu"
        )
        self.out_dir = Path(general_config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch                                           # (B, L)
        valid_mask = (x != self.pad_id).to(torch.bool).cpu()   # (B, L)

        with torch.inference_mode():
            _ = self.transformer(x.to(self.device))

        for layer_idx, tensor in self._acts.items():
            torch.save(
                {
                    "activations": tensor,   # (B, L, d_model)
                    "mask": valid_mask,      # (B, L)  → True = real token
                },
                self.out_dir / f"layer{layer_idx}_batch{batch_idx}.pt",
            )

        self._acts.clear()
        return None

    def train_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda b: collate(b, self.pad_id),
        )

    def teardown(self, stage: str):
        for h in self._handles:
            h.remove()

    def configure_optimizers(self):
        pass
