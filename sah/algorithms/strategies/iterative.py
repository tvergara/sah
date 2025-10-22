import torch
import torch.nn as nn
import torch.nn.functional as F

from sah.algorithms.strategies.base_strategy import BaseStrategy
from sah.algorithms.utils.data_collator import DataCollatorForAnswerOnlyLM
from sah.algorithms.utils.processed_dataset import ProcessedDataset


class IterativeStrategy(BaseStrategy):
    def __init__(self, lr, grads_in_memory, in_context_examples=0, merge_grads_every=100, default_lr=1e-5):
        super().__init__()
        self.lr = lr
        self.grads_in_memory = grads_in_memory
        self.in_context_examples = in_context_examples
        self.merge_grads_every = merge_grads_every
        self.default_lr = default_lr

    def setup(self, pl_module, stage):
        if stage == "fit":
            replace_linear_layers(self, pl_module.model, self.grads_in_memory, self.default_lr)
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


def replace_linear_layers(pl_module, model, batch_size, default_lr):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            linear = ModifiedLinear(module, batch_size, default_lr)
            setattr(model, name, linear)
        else:
            replace_linear_layers(pl_module, module, batch_size, default_lr)

class ModifiedLinear(nn.Module):
    def __init__(self, original_linear, grads_in_memory, default_lr):
        super().__init__()

        self.scale = nn.Parameter(torch.zeros(grads_in_memory), requires_grad=True)
        self.activated = False
        self.default_lr = default_lr

        self.main_weight = nn.Parameter(original_linear.weight.clone())
        self.register_buffer('weight_grads', torch.zeros(grads_in_memory, *self.main_weight.shape, dtype=torch.bfloat16))
        self.register_buffer('first_moment', torch.zeros(*self.main_weight.shape, dtype=torch.bfloat16))
        self.register_buffer('second_moment', torch.zeros(*self.main_weight.shape, dtype=torch.bfloat16))
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.time = 0
        self.epsilon = 1e-8

        self.main_bias = None
        if original_linear.bias is not None:
            self.main_bias = nn.Parameter(original_linear.bias.clone())
            self.register_buffer('bias_grads', torch.zeros(grads_in_memory, *self.main_bias.shape, dtype=torch.bfloat16))
            self.register_buffer('first_moment_bias', torch.zeros(*self.main_bias.shape, dtype=torch.bfloat16))
            self.register_buffer('second_moment_bias', torch.zeros(*self.main_bias.shape, dtype=torch.bfloat16))

    def forward(self, x):
        if not self.activated:
            return F.linear(x, self.main_weight, self.main_bias)

        grad = torch.einsum('i,ijk->jk', self.scale, self.weight_grads)
        first_moment = self.beta_1 * self.first_moment + (1 - self.beta_1) * grad
        second_moment = self.beta_2 * self.second_moment + (1 - self.beta_2) * (grad ** 2)
        first_moment = first_moment / (1 - self.beta_1 ** self.time)
        second_moment = second_moment / (1 - self.beta_2 ** self.time)

        weight = self.main_weight - self.default_lr * first_moment / (torch.sqrt(second_moment) + self.epsilon)

        bias = None
        if self.main_bias is not None:
            grad_bias = torch.einsum('i,ij->j', self.scale, self.bias_grads)
            first_moment_bias = self.beta_1 * self.first_moment_bias + (1 - self.beta_1) * grad_bias
            second_moment_bias = self.beta_2 * self.second_moment_bias + (1 - self.beta_2) * (grad_bias ** 2)
            first_moment_bias = first_moment_bias / (1 - self.beta_1 ** self.time)
            second_moment_bias = second_moment_bias / (1 - self.beta_2 ** self.time)

            bias = self.main_bias - self.default_lr * first_moment_bias / (torch.sqrt(second_moment_bias) + self.epsilon)

        return F.linear(x, weight, bias)

def merge_grads_into_weights(pl_module):
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            with torch.no_grad():
                grad = torch.einsum('i,ijk->jk', child.scale, child.weight_grads)
                child.first_moment.mul_(child.beta_1).add_(grad, alpha=(1 - child.beta_1))
                child.second_moment.mul_(child.beta_2).addcmul_(grad, grad, value=(1 - child.beta_2))

                first_moment_corrected = child.first_moment / (1 - child.beta_1 ** (child.time + 1))
                second_moment_corrected = child.second_moment / (1 - child.beta_2 ** (child.time + 1))

                child.main_weight.data -= child.default_lr * first_moment_corrected / (torch.sqrt(second_moment_corrected) + child.epsilon)
                child.weight_grads.zero_()

                if child.main_bias is not None:
                    bias_grad = torch.einsum('i,ij->j', child.scale, child.bias_grads)
                    child.first_moment_bias.mul_(child.beta_1).add_(bias_grad, alpha=(1 - child.beta_1))
                    child.second_moment_bias.mul_(child.beta_2).addcmul_(bias_grad, bias_grad, value=(1 - child.beta_2))

                    first_moment_bias_corrected = child.first_moment_bias / (1 - child.beta_1 ** (child.time + 1))
                    second_moment_bias_corrected = child.second_moment_bias / (1 - child.beta_2 ** (child.time + 1))

                    child.main_bias.data -= child.default_lr * first_moment_bias_corrected / (torch.sqrt(second_moment_bias_corrected) + child.epsilon)
                    child.bias_grads.zero_()

                child.scale.data.fill_(1)
                child.activated = False
                child.time += 1

def activate_linear_layers(pl_module):
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            child.activated = True
