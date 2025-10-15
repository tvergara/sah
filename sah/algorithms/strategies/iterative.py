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

    def configure_sharded_model(self, pl_module):
        from functools import partial

        import torch
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        pl_module.model.gradient_checkpointing_enable()

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            pl_module.model = pl_module.model.to(device)

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )

        device_id = torch.cuda.current_device() if torch.cuda.is_available() else None

        return {
            "auto_wrap_policy": auto_wrap_policy,
            "device_id": device_id,
        }

    def get_strategy_modules(self, pl_module):
        return {"scale_modules": self._scale_modules}

    def compute_bits(self, pl_module):
        return 0

    def training_step(self, pl_module, batch, batch_idx):
        if batch_idx == 0:
            merge_grads_into_weights(pl_module)
            optimizer = pl_module.optimizers()
            optimizer.state = {}
            self.update_gradients(pl_module, batch)
            activate_linear_layers(pl_module)
            return torch.tensor(0.0, device=pl_module.device, requires_grad=True)

        outputs = pl_module.model(**batch)
        loss = outputs.loss
        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        all_scales = torch.cat([m.scale.flatten() for m in self._scale_modules])
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
        scale_params = [m.scale for m in self._scale_modules]
        return torch.optim.AdamW(scale_params, lr=self.lr)

    def train_dataloader(self, pl_module):
        data_collator = DataCollatorForLanguageModeling(tokenizer=pl_module.tokenizer, mlm=False)
        world_size = pl_module.trainer.world_size if pl_module.trainer else 1
        rank = pl_module.trainer.global_rank if pl_module.trainer else 0

        sampler = SamplerWithSpecialFirstBatch(
            n=len(pl_module.dataset),
            first_batch_size=self.grads_in_memory,
            batch_size=pl_module.batch_size,
            world_size=world_size,
            rank=rank,
            shuffle=True
        )
        return torch.utils.data.DataLoader(
            pl_module.dataset,
            batch_sampler=sampler,
            num_workers=4,
            persistent_workers=False,
            collate_fn=data_collator,
        )


def replace_linear_layers(strategy, model, batch_size):
    strategy._scale_modules = nn.ModuleList()

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            scale_module = ScaleParameters(batch_size)
            strategy._scale_modules.append(scale_module)
            linear = ModifiedLinear(module, batch_size, scale_module)
            setattr(model, name, linear)
        else:
            replace_linear_layers(strategy, module, batch_size)

class ScaleParameters(nn.Module):
    def __init__(self, grads_in_memory):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(grads_in_memory, dtype=torch.float32), requires_grad=True)


class ModifiedLinear(nn.Module):
    def __init__(self, original_linear, grads_in_memory, scale_module):
        super().__init__()

        self.scale_module = scale_module
        self.activated = False

        self.main_weight = nn.Parameter(original_linear.weight.clone())
        self.weight_grads = nn.Parameter(
            torch.zeros(grads_in_memory, *self.main_weight.shape, dtype=torch.bfloat16),
            requires_grad=False
        )

        self.main_bias = None
        if original_linear.bias is not None:
            self.main_bias = nn.Parameter(original_linear.bias.clone())
            self.bias_grads = nn.Parameter(
                torch.zeros(grads_in_memory, *self.main_bias.shape, dtype=torch.bfloat16),
                requires_grad=False
            )

    def forward(self, x):
        if not self.activated:
            return F.linear(x, self.main_weight, self.main_bias)

        weight = self.main_weight + torch.einsum('i,ijk->jk', self.scale_module.scale, self.weight_grads) / self.scale_module.scale.shape[0]
        bias = None
        if self.main_bias is not None:
            bias = self.main_bias + torch.einsum('i,ij->j', self.scale_module.scale, self.bias_grads) / self.scale_module.scale.shape[0]

        return F.linear(x, weight, bias)

def merge_grads_into_weights(pl_module):
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            with torch.no_grad():
                child.main_weight.data += torch.einsum('i,ijk->jk', child.scale_module.scale, child.weight_grads)
                if child.main_bias is not None:
                    child.main_bias.data += torch.einsum('i,ij->j', child.scale_module.scale, child.bias_grads)

                child.weight_grads.zero_()
                child.scale_module.scale.data.zero_()
                if child.main_bias is not None:
                    child.bias_grads.zero_()
                child.activated = False

def activate_linear_layers(pl_module):
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            child.activated = True

class SamplerWithSpecialFirstBatch:
    def __init__(self, n, first_batch_size, batch_size, world_size=1, rank=0, shuffle=True, seed=0):
        self.n = n
        self.first_batch_size = first_batch_size
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.n, generator=g).tolist()
        else:
            indices = list(range(self.n))

        yield indices[:self.first_batch_size]

        remaining = indices[self.first_batch_size:]
        rank_indices = remaining[self.rank::self.world_size]

        for i in range(0, len(rank_indices), self.batch_size):
            batch = rank_indices[i:i + self.batch_size]
            if batch:
                yield batch

    def __len__(self):
        remaining = self.n - self.first_batch_size
        rank_samples = (remaining + self.world_size - 1) // self.world_size
        return 1 + (rank_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
