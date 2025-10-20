from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.utils.processed_dataset import ProcessedDataset


class BaseStrategy:
    def __init__(self):
        pass

    def setup(self, pl_module, stage):
        pass

    def on_train_start(self, pl_module):
        pass

    def training_step(self, pl_module, batch, batch_idx):
        outputs = pl_module.model(**batch)
        loss = outputs.loss
        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self, pl_module):
        return None

    def on_train_batch_end(self, pl_module, outputs, batch, batch_idx):
        pass

    def compute_bits(self, pl_module):
        pass

    def train_dataloader(self, pl_module):
        dataset = ProcessedDataset(pl_module.tokenizer, pl_module.dataset_name, max_examples=pl_module.max_examples)
        data_collator = DataCollatorForLanguageModeling(tokenizer=pl_module.tokenizer, mlm=False)

        return DataLoader(
            dataset,
            batch_size=pl_module.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=data_collator,
        )
