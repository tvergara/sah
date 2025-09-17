import hydra_zen
import torch
from lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.utils.data_classes import NetworkConfig, TokenizerConfig
from sah.algorithms.utils.processed_dataset import ProcessedDataset


class FinetuneWithStrategy(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        model_config: NetworkConfig,
        dataset_name: str,
        max_examples: int | None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = hydra_zen.instantiate(tokenizer_config)
        self.model = hydra_zen.instantiate(model_config)
        self.dataset = ProcessedDataset(self.tokenizer, dataset_name, max_examples=max_examples)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

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
        return torch.optim.SGD(self.model.parameters(), lr=0.01)
