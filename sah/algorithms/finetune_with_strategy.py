import hydra_zen
from lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.data_classes import NetworkConfig, TokenizerConfig
from sah.algorithms.utils.processed_dataset import ProcessedDataset


class FinetuneWithStrategy(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        model_config: NetworkConfig,
        strategy: BaseStrategy,
        dataset_name: str,
        max_examples: int | None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = hydra_zen.instantiate(tokenizer_config)
        self.model = hydra_zen.instantiate(model_config)
        self.dataset = ProcessedDataset(self.tokenizer, dataset_name, max_examples=max_examples)
        self.strategy = strategy

    def setup(self, stage):
        self.strategy.setup(self, stage)

    def training_step(self, batch, batch_idx):
        return self.strategy.training_step(self, batch, batch_idx)

    def train_dataloader(self):
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        return DataLoader(
            self.dataset,
            batch_size=6,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=data_collator,
        )

    def configure_optimizers(self):
        return self.strategy.configure_optimizers(self)
