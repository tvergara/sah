import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from sah.algorithms.strategies.base_strategy import BaseStrategy


class LoRAStrategy(BaseStrategy):
    def __init__(self, lr=1e-4, r=8, lora_alpha=32, lora_dropout=0.1, ft_strategy="lora"):
        super().__init__()
        self.lr = lr
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.ft_strategy = ft_strategy

    def setup(self, pl_module, stage):
        super().setup(pl_module, stage)
        if stage == "fit":
            if self.ft_strategy == "qlora":
                pl_module.model = prepare_model_for_kbit_training(pl_module.model)

            lora_config = LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )

            pl_module.model = get_peft_model(pl_module.model, lora_config)
            pl_module.model.print_trainable_parameters()

    def on_train_start(self, pl_module):
        self.bits = self.compute_bits(pl_module)
        return super().on_train_start(pl_module)

    def configure_optimizers(self, pl_module):
        return torch.optim.AdamW(pl_module.model.parameters(), lr=self.lr)

    def compute_bits(self, pl_module):
      return sum(param.numel() * 16 for param in pl_module.model.parameters() if param.requires_grad)
