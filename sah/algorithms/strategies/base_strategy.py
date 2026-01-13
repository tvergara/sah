import json
import uuid

import torch.distributed as dist
from torch.utils.data import DataLoader

from sah.algorithms.dataset_handlers import get_dataset_handler
from sah.algorithms.utils.data_collator import DataCollatorForAnswerOnlyLM


class BaseStrategy:
    def __init__(self):
        self.bits = 0

    def setup(self, pl_module, stage):
        self.dataset_handler = get_dataset_handler(
            pl_module.dataset_name,
            pl_module.tokenizer,
            block_size=pl_module.max_length,
            max_examples=pl_module.max_examples,
            generations_dir=pl_module.generations_dir
        )

    def on_train_start(self, pl_module):
        pass

    def on_validation_start(self, pl_module):
        pass

    def on_train_batch_start(self, pl_module, batch, batch_idx):
        pass

    def training_step(self, pl_module, batch, batch_idx):
        outputs = pl_module.model(**batch)
        loss = outputs.loss
        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, pl_module, batch, batch_idx):
        metrics = self.dataset_handler.validate_batch(pl_module, batch, batch_idx)

        for key, value in metrics.items():
            if key in ["correct_count", "total_count"]:
                pl_module.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True, reduce_fx="sum")
            else:
                pl_module.log(f"val/{key}", value, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self, pl_module):
        pl_module.log("val/bits", self.bits, prog_bar=True)

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        performance = pl_module.trainer.logged_metrics.get("val/performance", None)

        if performance is not None:
            experiment_id = pl_module.trainer.logger.experiment.id
            eval_run_id = str(uuid.uuid4())
            experiment_name = pl_module.experiment_name
            result_file = pl_module.result_file
            dataset_name = pl_module.dataset_name
            model_name = pl_module.model_name

            result = {
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "eval_run_id": eval_run_id,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "performance": performance.item() if hasattr(performance, 'item') else performance,
                "bits": self.bits,
                "seed": pl_module.hparams.seed,
                "strategy_hparams": vars(pl_module.hparams.strategy)
            }

            with open(result_file, 'a') as f:
                f.write(json.dumps(result, default=str) + '\n')

            if hasattr(self.dataset_handler, 'save_generations'):
                self.dataset_handler.save_generations(eval_run_id)

    def configure_optimizers(self, pl_module):
        return None

    def on_train_batch_end(self, pl_module, outputs, batch, batch_idx):
        pass

    def on_train_epoch_end(self, pl_module):
        pass

    def on_train_end(self, pl_module):
        pass

    def train_dataloader(self, pl_module):
        dataset = self.dataset_handler.get_train_dataset()
        data_collator = DataCollatorForAnswerOnlyLM(tokenizer=pl_module.tokenizer)

        return DataLoader(
            dataset,
            batch_size=pl_module.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=data_collator,
        )

    def val_dataloader(self, pl_module):
        dataset = self.dataset_handler.get_val_dataset()
        data_collator = DataCollatorForAnswerOnlyLM(tokenizer=pl_module.tokenizer)

        use_collator = 'labels' in dataset[0]

        return DataLoader(
            dataset,
            batch_size=pl_module.val_batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            collate_fn=data_collator if use_collator else None,
        )
