from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear

from sah.algorithms.strategies.base_strategy import BaseStrategy


class PhaseTwoStrategy(BaseStrategy):
    def __init__(
        self,
        lr: float,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        diffs_file: str,
        grads_in_memory: int
    ):
        super().__init__()
        self.lr = lr
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.diffs_file = Path(diffs_file)
        self.bits = 0
        self.grads_in_memory = grads_in_memory

    def setup(self, pl_module, stage):
        super().setup(pl_module, stage)
        if stage == "fit":
            lora_config = LoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )

            pl_module.model = get_peft_model(pl_module.model, lora_config)
            diffs = torch.load(self.diffs_file)
            replace_lora_layers(pl_module.model, self.grads_in_memory, diffs)

            num_lora_layers = sum(1 for key in diffs.keys() if key.endswith('.lora_A.default.weight'))
            gradient_bits = diffs['bits'][self.grads_in_memory - 1]
            alpha_bits = 32 * num_lora_layers * self.grads_in_memory
            self.bits = gradient_bits + alpha_bits
            print(f'Ratio (gradient_bits / alpha_bits): {gradient_bits / alpha_bits:.4f}')

    def configure_optimizers(self, pl_module):
        return torch.optim.AdamW(pl_module.model.parameters(), lr=self.lr)

    def training_step(self, pl_module, batch, batch_idx):
        outputs = pl_module.model(**batch)
        loss = outputs.loss
        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, reduce_fx="mean")

        alpha_params = [p for n, p in pl_module.model.named_parameters() if 'alphas' in n]
        all_alphas = torch.cat([p.flatten() for p in alpha_params])
        pl_module.log("train/alpha_mean", all_alphas.mean(), on_step=True, on_epoch=False)
        pl_module.log("train/alpha_std", all_alphas.std(), on_step=True, on_epoch=False)

        return loss

def replace_lora_layers(model, grads_in_memory, diffs, prefix=""):
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, Linear):
            lora = ModifiedLoRA(module, grads_in_memory, diffs, full_name)
            setattr(model, name, lora)
        else:
            replace_lora_layers(module, grads_in_memory, diffs, full_name)

class ModifiedLoRA(nn.Module):
    def __init__(self, original_lora, grads_in_memory, diffs, name):
        super().__init__()

        self.original_lora = original_lora
        for param in self.original_lora.parameters():
            param.requires_grad = False

        self.grads_A = nn.Parameter(torch.stack(diffs[f"{name}.lora_A.default.weight"])[:grads_in_memory], requires_grad=False)
        self.grads_B = nn.Parameter(torch.stack(diffs[f"{name}.lora_B.default.weight"])[:grads_in_memory], requires_grad=False)

        self.alphas = nn.Parameter(torch.ones(grads_in_memory), requires_grad=True)


    def forward(self, x):
        result = self.original_lora.base_layer(x)

        lora_A = self.original_lora.lora_A.default.weight + torch.einsum('i,ijk->jk', self.alphas, self.grads_A)
        lora_B = self.original_lora.lora_B.default.weight + torch.einsum('i,ijk->jk', self.alphas, self.grads_B)

        lora_output = torch.nn.functional.linear(torch.nn.functional.linear(x, lora_A), lora_B)
        scaling = self.original_lora.scaling["default"]
        result = result + lora_output * scaling

        return result
