# activation_collector.py
import pickle
from dataclasses import dataclass

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


class GrammarActivationCollector(LightningModule):
    def __init__(
        self,
        grammar_config: GrammarConfig,
        tokenizer_config: TokenizerConfig,
        checkpoint_config: CheckpointConfig,
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
            self.transformer, checkpoint_config.path, model_name="transformer"
        )

        self._acts: dict[int, torch.Tensor] = {}
        self._handles = listen_to_hidden_activations(
            self.transformer, self._acts, detach=True, device="cpu"
        )

    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, _ = batch
        with torch.inference_mode():
            _ = self.transformer(x.to(self.device))

        batch_acts = {k: v.clone() for k, v in self._acts.items()}
        print(batch_acts)
        self._acts.clear()

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
