import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.strategies.base_strategy import BaseStrategy


class IterativeStrategy(BaseStrategy):
    def __init__(self, lr, grads_in_memory):
        super().__init__()
        self.lr = lr
        self.grads_in_memory = grads_in_memory

    def setup(self, pl_module, stage):
        if stage == "fit":
            replace_linear_layers(self, pl_module.model, self.grads_in_memory)

    def compute_bits(self, pl_module):
        return 0

    def training_step(self, pl_module, batch, batch_idx):
        if batch_idx == 0:
            return self.update_gradients(pl_module, batch)

        outputs = pl_module.model(**batch)
        loss = outputs.loss
        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def update_gradients(self, pl_module, batch):
        batch_size = batch["input_ids"].size(0)
        for i in range(batch_size):
            single_batch = {k: v[i:i+1] for k, v in batch.items()}
            outputs = pl_module.model(**single_batch)

            original_params_dict = {n: p for n, p in pl_module.model.named_parameters() if '.main_' in n}
            grads = torch.autograd.grad(outputs.loss, list(original_params_dict.values()), retain_graph=False)
            grad_dict = dict(zip(original_params_dict.keys(), grads))

            for name, module in pl_module.model.named_modules():
                if isinstance(module, ModifiedLinear):
                    weight_key = f"{name}.main_weight"
                    if weight_key in grad_dict:
                        module.weight_grads[i] = grad_dict[weight_key].data.to(torch.bfloat16)

                    if module.main_bias is not None:
                        bias_key = f"{name}.main_bias"
                        if bias_key in grad_dict:
                            module.bias_grads[i] = grad_dict[bias_key].data.to(torch.bfloat16)

    def configure_optimizers(self, pl_module):
        scale_params = [p for n, p in pl_module.model.named_parameters() if 'scale' in n]
        return torch.optim.Adam(scale_params, lr=self.lr)

    def train_dataloader(self, pl_module):
        data_collator = DataCollatorForLanguageModeling(tokenizer=pl_module.tokenizer, mlm=False)
        sampler = SamplerWithSpecialFirstBatch(
            n=len(pl_module.dataset),
            first_batch_size=self.grads_in_memory,
            batch_size=pl_module.batch_size,
            shuffle=True
        )
        return torch.utils.data.DataLoader(
            pl_module.dataset,
            batch_sampler=sampler,
            num_workers=4,
            persistent_workers=True,
            collate_fn=data_collator,
        )


def replace_linear_layers(pl_module, model, batch_size):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            linear = ModifiedLinear(module, batch_size)
            setattr(model, name, linear)
        else:
            replace_linear_layers(pl_module, module, batch_size)

class ModifiedLinear(nn.Module):
    def __init__(self, original_linear, grads_in_memory):
        super().__init__()

        self.scale = nn.Parameter(torch.zeros(grads_in_memory), requires_grad=True)
        self.activated = False

        self.main_weight = nn.Parameter(original_linear.weight.clone())
        self.weight_grads = torch.zeros(grads_in_memory, *self.main_weight.shape, dtype=torch.bfloat16)

        self.main_bias = None
        if original_linear.bias is not None:
            self.main_bias = nn.Parameter(original_linear.bias.clone())
            self.bias_grads = torch.zeros(grads_in_memory, *self.main_bias.shape, dtype=torch.bfloat16)

    def forward(self, x):
        if not self.activated:
            return F.linear(x, self.main_weight, self.main_bias)

        weight = self.main_weight + torch.einsum('i,ijk->jk', self.scale, self.weight_grads)
        bias = None
        if self.main_bias is not None:
            bias = self.main_bias + torch.einsum('i,ij->j', self.scale, self.bias_grads)

        return F.linear(x, weight, bias)

class SamplerWithSpecialFirstBatch:
    def __init__(self, n, first_batch_size, batch_size, shuffle=True):
        self.n = n
        self.first_batch_size = first_batch_size
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = torch.randperm(self.n).tolist() if self.shuffle else list(range(self.n))
        yield indices[:self.first_batch_size]
        for i in range(self.first_batch_size, self.n, self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return 1 + (self.n - self.first_batch_size + self.batch_size - 1) // self.batch_size
