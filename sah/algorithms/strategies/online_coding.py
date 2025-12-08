import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.arithmetic_coding import compute_bits_from_logits


class OnlineCodingStrategy(BaseStrategy):
    def __init__(self, lr=1e-4, optimizer="adam", use_lora=False, r=8, lora_alpha=32, lora_dropout=0.1):
        super().__init__()
        self.lr = lr
        self.optimizer = optimizer
        self.use_lora = use_lora
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bits = 0
        self.current_epoch = 0

    def setup(self, pl_module, stage):
        super().setup(pl_module, stage)
        if stage == "fit" and self.use_lora:
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

    def configure_optimizers(self, pl_module):
        if self.optimizer == "adam":
            return torch.optim.Adam(pl_module.model.parameters(), lr=self.lr)
        elif self.optimizer == "adamw":
            return torch.optim.AdamW(pl_module.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def training_step(self, pl_module, batch, batch_idx):
        outputs = pl_module.model(**batch)
        loss = outputs.loss

        if self.current_epoch == 0:
            with torch.no_grad():
                total_bits = compute_bits_from_logits(
                    outputs.logits,
                    batch['input_ids'],
                    batch.get('attention_mask')
                )

                if dist.is_initialized():
                    total_bits_tensor = torch.tensor(total_bits, dtype=torch.float32, device=pl_module.device)
                    dist.all_reduce(total_bits_tensor, op=dist.ReduceOp.SUM)
                    total_bits = int(total_bits_tensor.item())

                self.bits += total_bits

        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_train_epoch_end(self, pl_module):
        self.current_epoch += 1
