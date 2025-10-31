from pathlib import Path

import hydra_zen
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
        val_dataset_name: list[str],
        val_batch_size: int,
        max_examples: int | None,
        batch_size: int,
        result_file: str,
        experiment_name: str,
        max_length: int
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = hydra_zen.instantiate(tokenizer_config)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = hydra_zen.instantiate(model_config)
        self.max_examples = max_examples
        self.dataset_name = dataset_name
        self.val_dataset_name = val_dataset_name
        self.strategy = strategy
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.result_file = Path(result_file)
        self.experiment_name = experiment_name
        self.max_length = max_length

    def setup(self, stage):
        self.strategy.setup(self, stage)

    def training_step(self, batch, batch_idx):
        return self.strategy.training_step(self, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.strategy.validation_step(self, batch, batch_idx)

    def on_validation_epoch_end(self):
        self.strategy.on_validation_epoch_end(self)

    def on_train_start(self):
        self.strategy.on_train_start(self)

    def on_train_batch_start(self, batch, batch_idx):
        self.strategy.on_train_batch_start(self, batch, batch_idx)

    def train_dataloader(self):
        return self.strategy.train_dataloader(self)

    def val_dataloader(self):
        return self.strategy.val_dataloader(self)

    def configure_optimizers(self):
        return self.strategy.configure_optimizers(self)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.strategy.on_train_batch_end(self, outputs, batch, batch_idx)

    def on_train_end(self):
        self.strategy.on_train_end(self)
