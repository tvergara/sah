
import torch

from sah.algorithms.strategies.base_strategy import BaseStrategy


class AdamStrategy(BaseStrategy):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr
        self.bits = 0

    def configure_optimizers(self, pl_module):
        return torch.optim.Adam(pl_module.model.parameters(), lr=self.lr)

    def training_step(self, pl_module, batch, batch_idx):
        tensor = batch['input_ids']
        bits_per_byte = 8
        size_in_bits = tensor.element_size() * bits_per_byte * tensor.numel()
        self.bits += size_in_bits

        return super().training_step(pl_module, batch, batch_idx)
