import math
from decimal import Decimal
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.arithmetic_coding import encode


class PhaseOneStrategy(BaseStrategy):
    def __init__(
        self,
        lr: float,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        diffs_file: str
    ):
        super().__init__()
        self.lr = lr
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.diffs_file = Path(diffs_file)
        self.bits = 0

    def setup(self, pl_module, stage):
        if stage == "fit":
            lora_config = LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            pl_module.model = get_peft_model(pl_module.model, lora_config)
            pl_module.model.print_trainable_parameters()

    def configure_optimizers(self, pl_module):
        return torch.optim.AdamW(pl_module.model.parameters(), lr=self.lr)

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

    def on_train_start(self, pl_module):
        self.lora_params = {name: param.data.clone() for name, param in pl_module.model.named_parameters() if param.requires_grad}
        self.diffs = {name: [] for name in self.lora_params}
        self.diffs['bits'] = []

    def on_train_batch_end(self, pl_module, outputs, batch, batch_idx):
        for name, param in pl_module.model.named_parameters():
            if not param.requires_grad:
                continue
            old_tensor = self.lora_params[name]
            new_tensor = param.data
            diff = new_tensor - old_tensor
            self.diffs[name].append(diff.detach().cpu())
            self.lora_params[name] = new_tensor.clone()
        self.diffs['bits'].append(self.bits)

    def on_train_end(self, pl_module):
        self.diffs_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.diffs, self.diffs_file)
