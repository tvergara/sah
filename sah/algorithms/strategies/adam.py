
import math
from decimal import Decimal

import torch

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.arithmetic_coding import encode


class AdamStrategy(BaseStrategy):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr
        self.bits = 0

    def configure_optimizers(self, pl_module):
        return torch.optim.Adam(pl_module.model.parameters(), lr=self.lr)

    def on_train_batch_start(self, pl_module, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        with torch.no_grad():
                compressed_batch = encode(pl_module.model, input_ids, attention_mask)

        total_bits = 0
        log_2 = Decimal(2).ln()
        for compressed_value, interval_width in compressed_batch:
            if interval_width > 0:
                bits = math.ceil(float(-(interval_width.ln() / log_2)))
                total_bits += bits
        self.bits += total_bits

        return super().on_train_batch_start(pl_module, batch, batch_idx)

    def training_step(self, pl_module, batch, batch_idx):
        return super().training_step(pl_module, batch, batch_idx)
