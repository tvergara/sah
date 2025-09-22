import torch

from sah.algorithms.strategies.base_strategy import BaseStrategy


class AdamStrategy(BaseStrategy):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr

    def configure_optimizers(self, pl_module):
        return torch.optim.Adam(pl_module.model.parameters(), lr=self.lr)

    def compute_bits(self, pl_module):
      return sum(param.numel() * 16 for param in pl_module.model.parameters())
