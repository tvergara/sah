import torch
import torch.nn as nn
import torch.nn.functional as F

from sah.algorithms.strategies.base_strategy import BaseStrategy


class CompressedFinetuneStrategy(BaseStrategy):
    def __init__(self, lr, scale_lr, grad_accumulation_steps, compress_batches_every):
        super().__init__()
        self.lr = lr
        self.scale_lr = scale_lr
        self.grad_accumulation_steps = grad_accumulation_steps
        self.compress_batches_every = compress_batches_every
        self.scale_optimizer = None
        self.grad_accumulation_counter = 0
        self.linear_layers = 0
        self.tensor_buffer = None

    def setup(self, pl_module, stage):
        if stage == "fit":
            total_slots = pl_module.batch_size * self.grad_accumulation_steps
            replace_linear_layers(self, pl_module.model, total_slots)
            pl_module.automatic_optimization = False

    def compute_bits(self, pl_module):
        total_slots = pl_module.batch_size * self.grad_accumulation_steps
        compression_cycles = len(pl_module.dataset) // (total_slots * self.compress_batches_every)

        total_params = self.linear_layers * total_slots * compression_cycles
        total_tokens = compression_cycles * pl_module.dataset.block_size * total_slots

        return (total_params + total_tokens) * 16

    def _split_model_across_gpus(self, pl_module):
        num_gpus = torch.cuda.device_count()
        layers = pl_module.model.model.layers
        layers_per_gpu = len(layers) // num_gpus

        for gpu_id in range(num_gpus):
            start_idx = gpu_id * layers_per_gpu
            end_idx = (gpu_id + 1) * layers_per_gpu if gpu_id < num_gpus - 1 else len(layers)

            for layer_idx in range(start_idx, end_idx):
                layers[layer_idx] = layers[layer_idx].to(f'cuda:{gpu_id}')
                layers[layer_idx] = LayerWithTransfer(layers[layer_idx], f'cuda:{gpu_id}', self)

        pl_module.model.model.embed_tokens.to('cuda:0')
        pl_module.model.model.norm.to(f'cuda:{num_gpus-1}')
        pl_module.model.lm_head.to(f'cuda:{num_gpus-1}')


    def on_train_start(self, pl_module):
        if torch.cuda.device_count() > 1:
            self._split_model_across_gpus(pl_module)
            move_perturbation_tensors_to_device(pl_module)

    def training_step(self, pl_module, batch, batch_idx):
        total_cycle = self.grad_accumulation_steps * self.compress_batches_every
        cycle_pos = batch_idx % total_cycle
        is_compressed_step = cycle_pos < self.grad_accumulation_steps

        if is_compressed_step:
            if cycle_pos == 0:
                compile_perturbations_into_weights(pl_module)
            self.compressed_batch(pl_module, batch, cycle_pos)

            if cycle_pos == self.grad_accumulation_steps - 1:
                initialize_scale_parameters(pl_module, self.lr)
                self._setup_scale_optimizer(pl_module)
            return None

        return self._optimize_scales(pl_module, batch)

    def _setup_scale_optimizer(self, pl_module):
        scale_params = [p for n, p in pl_module.model.named_parameters() if 'scale' in n]
        self.scale_optimizer = torch.optim.Adam(scale_params, lr=self.scale_lr)

    def _optimize_scales(self, pl_module, batch):
        outputs = pl_module.model(**batch)
        self.tensor_buffer = None
        loss = outputs.loss

        scale_params = [p for n, p in pl_module.model.named_parameters() if 'scale' in n]
        grads = torch.autograd.grad(loss, scale_params, retain_graph=False)

        for param, grad in zip(scale_params, grads):
            if param.grad is None:
                param.grad = grad / self.grad_accumulation_steps
            else:
                param.grad += grad / self.grad_accumulation_steps

        self.grad_accumulation_counter += 1

        if self.grad_accumulation_counter % self.grad_accumulation_steps == 0:
            self.scale_optimizer.step()
            self.scale_optimizer.zero_grad()
            self.grad_accumulation_counter = 0

        pl_module.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def compressed_batch(self, pl_module, batch, accumulation_step=0):
        batch_size = batch["input_ids"].size(0)
        total_loss = 0

        for i in range(batch_size):
            single_batch = {k: v[i:i+1] for k, v in batch.items()}

            outputs = pl_module.model(**single_batch)
            self.tensor_buffer = None
            loss = outputs.loss
            total_loss += loss.item()

            original_params_dict = {n: p for n, p in pl_module.model.named_parameters() if 'original' in n}
            grads = torch.autograd.grad(loss, list(original_params_dict.values()), retain_graph=False)
            grad_dict = dict(zip(original_params_dict.keys(), grads))

            slot_idx = accumulation_step * batch_size + i

            for name, module in pl_module.model.named_modules():
                if isinstance(module, ModifiedLinear):
                    weight_key = f"{name}.original_weight"
                    if weight_key in grad_dict:
                        module.weight_perturbations[slot_idx] = grad_dict[weight_key].data.to(torch.bfloat16)

                    if module.original_bias is not None:
                        bias_key = f"{name}.original_bias"
                        if bias_key in grad_dict:
                            module.bias_perturbations[slot_idx] = grad_dict[bias_key].data.to(torch.bfloat16)

        avg_loss = total_loss / batch_size
        pl_module.log("train/loss", avg_loss, on_step=True, on_epoch=False, prog_bar=True)
        return avg_loss

    def configure_optimizers(self, pl_module):
        pass

    def on_train_batch_end(self, pl_module, outputs, batch, batch_idx):
        import os

        from lightning.pytorch.callbacks import ModelCheckpoint

        checkpoint_callback = next((cb for cb in pl_module.trainer.callbacks if isinstance(cb, ModelCheckpoint)), None)
        if checkpoint_callback and checkpoint_callback._every_n_train_steps > 0:
            if (batch_idx + 1) % checkpoint_callback._every_n_train_steps == 0:
                os.makedirs(checkpoint_callback.dirpath, exist_ok=True)
                pl_module.trainer.save_checkpoint(os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.filename}.ckpt"))

class ModifiedLinear(nn.Module):
    def __init__(self, original_linear, batch_size):
        super().__init__()

        self.scale = nn.Parameter(torch.zeros(batch_size), requires_grad=True)

        self.original_weight = nn.Parameter(original_linear.weight.clone())
        self.weight_perturbations = torch.zeros(batch_size, *self.original_weight.shape, dtype=torch.bfloat16)

        self.original_bias = None
        if original_linear.bias is not None:
            self.original_bias = nn.Parameter(original_linear.bias.clone())
            self.bias_perturbations = torch.zeros(batch_size, *self.original_bias.shape, dtype=torch.bfloat16)


    def forward(self, x):
        weight = self.original_weight + torch.einsum('i,ijk->jk', self.scale, self.weight_perturbations)
        bias = None
        if self.original_bias is not None:
            bias = self.original_bias + torch.einsum('i,ij->j', self.scale, self.bias_perturbations)

        return F.linear(x, weight, bias)

def replace_linear_layers(pl_module, model, batch_size):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            linear = ModifiedLinear(module, batch_size)
            pl_module.linear_layers += 1
            setattr(model, name, linear)
        else:
            replace_linear_layers(pl_module, module, batch_size)


def move_perturbation_tensors_to_device(pl_module):
    for child in pl_module.model.modules():
        if isinstance(child, ModifiedLinear):
            layer_device = child.original_weight.device
            child.weight_perturbations = child.weight_perturbations.to(layer_device)
            child.scale = child.scale.to(layer_device)
            if child.original_bias is not None:
                child.bias_perturbations = child.bias_perturbations.to(layer_device)


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

class LayerWithTransfer(nn.Module):
    def __init__(self, original_layer, target_device, strategy):
        super().__init__()
        self.layer = original_layer
        self.target_device = torch.device(target_device)
        self.strategy = strategy

    def forward(self, hidden_states, *args, **kwargs):
        if self.strategy.tensor_buffer is None:
            self.strategy.tensor_buffer = (kwargs['position_embeddings'], kwargs['attention_mask'])

        if hidden_states.device == self.target_device:
            kwargs['position_embeddings'], kwargs['attention_mask'] = self.strategy.tensor_buffer
            return self.layer(hidden_states, *args, **kwargs)

        hidden_states = hidden_states.to(self.target_device)

        new_args = []
        for arg in args:
            if torch.is_tensor(arg):
                new_args.append(arg.to(self.target_device))
            else:
                new_args.append(arg)

        new_kwargs = {}
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                new_kwargs[k] = v.to(self.target_device)
            elif k == 'position_embeddings':
                new_kwargs[k] = (v[0].to(self.target_device), v[1].to(self.target_device))
            else:
                new_kwargs[k] = v

        self.strategy.tensor_buffer = (new_kwargs['position_embeddings'], new_kwargs['attention_mask'])
        output = self.layer(hidden_states, *new_args, **new_kwargs)
        return output
