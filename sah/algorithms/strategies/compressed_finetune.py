import torch
import torch.nn as nn
import torch.nn.functional as F

from sah.algorithms.strategies.base_strategy import BaseStrategy


class CompressedFinetuneStrategy(BaseStrategy):
    def __init__(self, lr, grad_accumulation_steps, compress_batches_every):
        super().__init__()
        self.lr = lr
        self.grad_accumulation_steps = grad_accumulation_steps
        self.compress_batches_every = compress_batches_every

    def setup(self, pl_module, stage):
        if stage == "fit":
            replace_linear_layers(pl_module.model, pl_module.batch_size)
            pl_module.automatic_optimization = False

    def on_train_start(self, pl_module):
        self._move_perturbation_tensors_to_device(pl_module)

    def _move_perturbation_tensors_to_device(self, pl_module):
        device = next(pl_module.model.parameters()).device
        for child in pl_module.model.modules():
            if isinstance(child, ModifiedLinear):
                child.weight_perturbations = child.weight_perturbations.to(device)
                if child.original_bias is not None:
                    child.bias_perturbations = child.bias_perturbations.to(device)

    def training_step(self, pl_module, batch, batch_idx):
        steps_since_last_meta_batch = batch_idx % self.compress_batches_every
        is_compressed_step = steps_since_last_meta_batch < self.grad_accumulation_steps
        if is_compressed_step:
            return

        outputs = pl_module.model(**batch)
        loss = outputs.loss
        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self, pl_module):
        pass

class ModifiedLinear(nn.Module):
    def __init__(self, original_linear, batch_size):
        super().__init__()

        self.scale = nn.Parameter(torch.zeros(batch_size), requires_grad=True)

        self.original_weight = original_linear.weight
        self.weight_perturbations = torch.zeros(batch_size, *self.original_weight.shape, dtype=torch.float16)

        self.original_bias = None
        if original_linear.bias is not None:
            self.original_bias = original_linear.bias
            self.bias_perturbations = torch.zeros(batch_size, *self.original_bias.shape, dtype=torch.float16)

    def forward(self, x):
        weight = self.original_weight + torch.einsum('i,ijk->jk', self.scale, self.weight_perturbations)
        bias = None
        if self.original_bias is not None:
            bias = self.original_bias + torch.einsum('i,ij->j', self.scale, self.bias_perturbations)

        return F.linear(x, weight, bias)

def replace_linear_layers(model, batch_size):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            linear = ModifiedLinear(module, batch_size)
            setattr(model, name, linear)
        else:
            replace_linear_layers(module, batch_size)
