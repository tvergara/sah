import torch

from sah.algorithms.strategies.base_strategy import BaseStrategy


class FullFTStrategy(BaseStrategy):
    def __init__(self, lr=1e-4, bits_per_parameter=16):
        super().__init__()
        self.lr = lr
        self.bits_per_parameter = bits_per_parameter
        self.bits = 0

    def setup(self, pl_module, stage):
        super().setup(pl_module, stage)
        if stage == "fit":
            num_params = sum(p.numel() for p in pl_module.model.parameters() if p.requires_grad)
            self.bits = num_params * self.bits_per_parameter

    def configure_optimizers(self, pl_module):
        return torch.optim.AdamW(pl_module.model.parameters(), lr=self.lr)

    def training_step(self, pl_module, batch, batch_idx):
        outputs = pl_module.model(**batch)
        loss = outputs.loss

        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss
