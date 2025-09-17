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
        move_perturbation_tensors_to_device(pl_module)

    def training_step(self, pl_module, batch, batch_idx):
        steps_since_last_meta_batch = batch_idx % self.compress_batches_every
        is_compressed_step = steps_since_last_meta_batch < self.grad_accumulation_steps
        if is_compressed_step:
            return

        outputs = pl_module.model(**batch)
        loss = outputs.loss
        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def compressed_batch(self, pl_module, batch, accumulation_step=0):
        compile_perturbations_into_weights(pl_module)

        batch_size = batch["input_ids"].size(0)
        total_loss = 0

        for i in range(batch_size):
            single_batch = {k: v[i:i+1] for k, v in batch.items()}

            outputs = pl_module.model(**single_batch)
            loss = outputs.loss
            total_loss += loss.item()

            original_params = [p for n, p in pl_module.model.named_parameters() if 'original' in n]
            grads = torch.autograd.grad(loss, original_params, retain_graph=False)

            slot_idx = accumulation_step * pl_module.batch_size + i
            param_idx = 0

            for module in pl_module.model.modules():
                if isinstance(module, ModifiedLinear):
                    module.weight_perturbations[slot_idx] = grads[param_idx].data.to(torch.float16)
                    param_idx += 1

                    if module.original_bias is not None:
                        module.bias_perturbations[slot_idx] = grads[param_idx].data.to(torch.float16)
                        param_idx += 1

        initialize_scale_parameters(pl_module, self.lr)

        avg_loss = total_loss / batch_size
        pl_module.log("train/loss", avg_loss, on_step=True, on_epoch=False, prog_bar=True)
        return avg_loss

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


def move_perturbation_tensors_to_device(pl_module):
    device = next(pl_module.model.parameters()).device
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            child.weight_perturbations = child.weight_perturbations.to(device)
            if child.original_bias is not None:
                child.bias_perturbations = child.bias_perturbations.to(device)


def compile_perturbations_into_weights(pl_module):
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            with torch.no_grad():
                weight_update = torch.einsum('i,ijk->jk', child.scale, child.weight_perturbations)
                child.original_weight.data += weight_update

                if child.original_bias is not None:
                    bias_update = torch.einsum('i,ij->j', child.scale, child.bias_perturbations)
                    child.original_bias.data += bias_update

                child.weight_perturbations.zero_()
                child.scale.data.zero_()
                if child.original_bias is not None:
                    child.bias_perturbations.zero_()


def initialize_scale_parameters(pl_module, lr):
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            with torch.no_grad():
                child.scale.data.fill_(-lr)
