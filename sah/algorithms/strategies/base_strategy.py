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
