import torch
from peft import LoraConfig, get_peft_model

from sah.algorithms.strategies.base_strategy import BaseStrategy


class LoRAStrategy(BaseStrategy):
    def __init__(self, lr=1e-4, r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.lr = lr
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

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

    def compute_bits(self, pl_module):
      return sum(param.numel() * 16 for param in pl_module.model.parameters() if param.requires_grad)
