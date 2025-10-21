import torch
import torch.nn as nn
import torch.nn.functional as F

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.data_collator import DataCollatorForAnswerOnlyLM
from sah.algorithms.utils.processed_dataset import ProcessedDataset


class IterativeStrategy(BaseStrategy):
    def __init__(self, lr, grads_in_memory, in_context_examples=0, merge_grads_every=100):
        super().__init__()
        self.lr = lr
        self.grads_in_memory = grads_in_memory
        self.in_context_examples = in_context_examples
        self.merge_grads_every = merge_grads_every

    def setup(self, pl_module, stage):
        if stage == "fit":
            replace_linear_layers(self, pl_module.model, self.grads_in_memory)
            self.cached_dataset = ProcessedDataset(
                pl_module.tokenizer,
                pl_module.dataset_name,
                max_examples=pl_module.max_examples,
                in_context_examples=self.in_context_examples
            )

    def compute_bits(self, pl_module):
        return 0

    def sample_special_batch(self, pl_module):
        indices = torch.randperm(len(self.cached_dataset))[:self.grads_in_memory].tolist()
        samples = [self.cached_dataset[i] for i in indices]
        data_collator = DataCollatorForAnswerOnlyLM(tokenizer=pl_module.tokenizer)
        batch = data_collator(samples)
        batch = {k: v.to(pl_module.device) for k, v in batch.items()}
        return batch

    def training_step(self, pl_module, batch, batch_idx):
        if pl_module.global_step % self.merge_grads_every == 0:
            special_batch = self.sample_special_batch(pl_module)
            merge_grads_into_weights(pl_module)
            optimizer = pl_module.trainer.optimizers[0]
            optimizer.state.clear()
            self.update_gradients(pl_module, special_batch)
            activate_linear_layers(pl_module)

        outputs = pl_module.model(**batch)
        loss = outputs.loss
        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, reduce_fx="mean")

        scale_params = [p for n, p in pl_module.model.named_parameters() if 'scale' in n]
        all_scales = torch.cat([p.flatten() for p in scale_params])
        pl_module.log("train/scale_mean", all_scales.mean(), on_step=True, on_epoch=False)
        pl_module.log("train/scale_std", all_scales.std(), on_step=True, on_epoch=False)

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
                        module.weight_grads[i] = grad_dict[weight_key].detach().data.to(torch.bfloat16)

                    if module.main_bias is not None:
                        bias_key = f"{name}.main_bias"
                        if bias_key in grad_dict:
                            module.bias_grads[i] = grad_dict[bias_key].detach().data.to(torch.bfloat16)

            del outputs, grads, grad_dict, single_batch

    def configure_optimizers(self, pl_module):
        scale_params = [p for n, p in pl_module.model.named_parameters() if 'scale' in n]
        return torch.optim.AdamW(scale_params, lr=self.lr)

    def train_dataloader(self, pl_module):
        data_collator = DataCollatorForAnswerOnlyLM(tokenizer=pl_module.tokenizer)

        return torch.utils.data.DataLoader(
            self.cached_dataset,
            batch_size=pl_module.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=False,
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
        self.register_buffer('weight_grads', torch.zeros(grads_in_memory, *self.main_weight.shape, dtype=torch.bfloat16))

        self.main_bias = None
        if original_linear.bias is not None:
            self.main_bias = nn.Parameter(original_linear.bias.clone())
            self.register_buffer('bias_grads', torch.zeros(grads_in_memory, *self.main_bias.shape, dtype=torch.bfloat16))

    def forward(self, x):
        if not self.activated:
            return F.linear(x, self.main_weight, self.main_bias)

        weight = self.main_weight + torch.einsum('i,ijk->jk', self.scale, self.weight_grads)
        bias = None
        if self.main_bias is not None:
            bias = self.main_bias + torch.einsum('i,ij->j', self.scale, self.bias_grads)

        return F.linear(x, weight, bias)

def merge_grads_into_weights(pl_module):
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            with torch.no_grad():
                weight_diff = torch.einsum('i,ijk->jk', child.scale, child.weight_grads)
                child.main_weight.data += weight_diff
                child.weight_grads.zero_()

                if child.main_bias is not None:
                    bias_diff = torch.einsum('i,ij->j', child.scale, child.bias_grads)
                    child.main_bias.data += bias_diff
                    child.bias_grads.zero_()

                child.scale.data.zero_()
                child.activated = False

def activate_linear_layers(pl_module):
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            child.activated = True
