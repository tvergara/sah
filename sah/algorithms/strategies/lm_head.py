import torch

from sah.algorithms.strategies.base_strategy import BaseStrategy


class LMHeadStrategy(BaseStrategy):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr

    def setup(self, pl_module, stage):
        super().setup(pl_module, stage)
        if stage == "fit":
            for param in pl_module.model.parameters():
                param.requires_grad = False

            for param in pl_module.model.lm_head.parameters():
                param.requires_grad = True

            num_params = sum(p.numel() for p in pl_module.model.lm_head.parameters())
            self.bits = 16 * num_params

    def configure_optimizers(self, pl_module):
        return torch.optim.AdamW(pl_module.model.lm_head.parameters(), lr=self.lr)
