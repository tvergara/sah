from pathlib import Path

import hydra_zen
import pandas as pd
from lightning import LightningModule

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.data_classes import NetworkConfig, TokenizerConfig


class FinetuneWithStrategy(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        model_config: NetworkConfig,
        strategy: BaseStrategy,
        dataset_name: str,
        max_examples: int | None,
        batch_size: int,
        result_file: str,
        experiment_name: str,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = hydra_zen.instantiate(tokenizer_config)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = hydra_zen.instantiate(model_config)
        self.max_examples = max_examples
        self.dataset_name = dataset_name
        self.strategy = strategy
        self.batch_size = batch_size
        self.result_file = Path(result_file)
        self.experiment_name = experiment_name

    def setup(self, stage):
        self.strategy.setup(self, stage)
        bits = self.strategy.compute_bits(self)
        print(f"Will be using {bits} bits to communicate these changes")

        row = {"experiment_name": self.experiment_name, "bits": bits}
        if self.result_file.exists():
            df = pd.read_csv(self.result_file)
            mask = df["experiment_name"] == self.experiment_name
            if mask.any():
                df.loc[mask, "bits"] = bits
            else:
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(self.result_file, index=False)

    def training_step(self, batch, batch_idx):
        return self.strategy.training_step(self, batch, batch_idx)

    def on_train_start(self):
        self.strategy.on_train_start(self)

    def train_dataloader(self):
        return self.strategy.train_dataloader(self)

    def configure_optimizers(self):
        return self.strategy.configure_optimizers(self)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.strategy.on_train_batch_end(self, outputs, batch, batch_idx)
