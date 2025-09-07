import hydra_zen
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset
from lightning import LightningModule
from torch import nn
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.formatters import get_dataset_formatter
from sah.algorithms.llm_finetuning import NetworkConfig, TokenizerConfig


def get_scale_manager_wrap_policy():
    """Return FSDP auto wrap policy for ScaleManager modules only."""
    return ModuleWrapPolicy({ScaleManager})


class ScaleManager(nn.Module):
    """Dedicated module for managing scale parameters that can be FSDP-wrapped."""
    def __init__(self, batch_size, grad_accumulation_steps):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(batch_size * grad_accumulation_steps), requires_grad=True)


class MetaMath(Dataset):
    def __init__(self, tokenizer, block_size=512):
        self.tokenizer = tokenizer
        self.block_size = block_size

        print("Loading MetaMathQA dataset...")
        raw_dataset = load_dataset("meta-math/MetaMathQA", split="train[:20000]")
        print(f"Dataset loaded: {len(raw_dataset)} examples")
        formatter = get_dataset_formatter("meta-math/MetaMathQA")

        self.examples = []
        print("Processing examples...")
        for i, example in enumerate(raw_dataset):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(raw_dataset)} examples")
            formatted = formatter(example)
            # Tokenize the text
            tokenized = tokenizer(
                formatted["text"],
                truncation=True,
                padding=False,
                max_length=block_size,
                return_tensors="pt"
            )

            if len(tokenized["input_ids"][0]) > 1:
                self.examples.append({
                    "input_ids": tokenized["input_ids"][0],
                    "attention_mask": tokenized["attention_mask"][0]
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ModifiedLinear(nn.Module):
    def __init__(self, original_linear, batch_size, grad_accumulation_steps=1):
        super().__init__()

        self.original_weight = original_linear.weight
        setattr(self.original_weight, 'original', True)

        if original_linear.bias is not None:
            self.original_bias = original_linear.bias
            setattr(self.original_bias, 'original', True)
        else:
            self.original_bias = None

        # Store quantized perturbations as buffers (not parameters)
        self.register_buffer('weight_perturbation_quantized', torch.zeros(batch_size * grad_accumulation_steps, *self.original_weight.shape, dtype=torch.uint8))
        self.register_buffer('weight_perturbation_scale', torch.ones(batch_size * grad_accumulation_steps, dtype=torch.float32))
        self.register_buffer('weight_perturbation_zero_point', torch.zeros(batch_size * grad_accumulation_steps, dtype=torch.uint8))

        # Use ScaleManager for FSDP-wrapped scale parameters
        self.scale_manager = ScaleManager(batch_size, grad_accumulation_steps)

        if self.original_bias is not None:
            self.register_buffer('bias_perturbation_quantized', torch.zeros(batch_size * grad_accumulation_steps, *self.original_bias.shape, dtype=torch.uint8))
            self.register_buffer('bias_perturbation_scale', torch.ones(batch_size * grad_accumulation_steps, dtype=torch.float32))
            self.register_buffer('bias_perturbation_zero_point', torch.zeros(batch_size * grad_accumulation_steps, dtype=torch.uint8))

        self.activated = False

    def _quantize_perturbation(self, perturbation_tensor, slot_idx):
        """Quantize a perturbation tensor using int8 quantization."""
        # Calculate scale and zero point for the perturbation
        min_val = perturbation_tensor.min()
        max_val = perturbation_tensor.max()

        # Avoid division by zero
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            scale = (max_val - min_val) / 255.0  # 8-bit: 2^8 - 1 = 255
            zero_point = min_val

        # Quantize to 8-bit
        quantized = torch.clamp(
            torch.round((perturbation_tensor - zero_point) / scale),
            0, 255
        ).to(torch.uint8)

        return quantized, scale, zero_point

    def _dequantize_perturbations(self):
        """Dequantize weight and bias perturbations."""
        # Dequantize weight perturbations
        weight_perturbations = torch.zeros_like(self.weight_perturbation_quantized, dtype=torch.float32)
        for i in range(self.weight_perturbation_quantized.shape[0]):
            weight_perturbations[i] = (
                self.weight_perturbation_quantized[i].float() * self.weight_perturbation_scale[i] +
                self.weight_perturbation_zero_point[i].float()
            )

        bias_perturbations = None
        if self.original_bias is not None:
            bias_perturbations = torch.zeros_like(self.bias_perturbation_quantized, dtype=torch.float32)
            for i in range(self.bias_perturbation_quantized.shape[0]):
                bias_perturbations[i] = (
                    self.bias_perturbation_quantized[i].float() * self.bias_perturbation_scale[i] +
                    self.bias_perturbation_zero_point[i].float()
                )

        return weight_perturbations, bias_perturbations

    def forward(self, x):
        if not self.activated:
            return F.linear(x, self.original_weight, self.original_bias)

        # Dequantize perturbations for computation
        weight_perturbations, bias_perturbations = self._dequantize_perturbations()

        weight = self.original_weight + torch.einsum('i,ijk->jk', self.scale_manager.scale, weight_perturbations)
        bias = None
        if self.original_bias is not None:
            bias = self.original_bias + torch.einsum('i,ij->j', self.scale_manager.scale, bias_perturbations)

        return F.linear(x, weight, bias)

def replace_linear_layers(model, batch_size, grad_accumulation_steps=1, original_params=[]):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            linear = ModifiedLinear(module, batch_size, grad_accumulation_steps)
            setattr(model, name, linear)
        else:
            replace_linear_layers(module, batch_size, grad_accumulation_steps, original_params=original_params)
    return original_params

class CompressedFinetuneOnMetaMath(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        pretrained_config: NetworkConfig,
        batch_size: int = 8,
        block_size: int = 256,
        default_lr: float = 1e-4,
        compress_batches_every: int = 100,
        scale_lr: float = 1e-4,
        grad_accumulation_steps: int = 1
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = hydra_zen.instantiate(pretrained_config)
        self.batch_size = batch_size
        self.block_size = block_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.pretrained_config = pretrained_config

        self.tokenizer_config = tokenizer_config
        self.automatic_optimization = False
        self.default_lr = default_lr
        self.compress_batches_every = compress_batches_every
        self.scale_lr = scale_lr
        self.scale_optimizer = None
        self.grad_accumulation_counter = 0
        self.linear_layers_replaced = False

    def setup(self, stage=None):
        """Setup method called after model is moved to GPU."""
        if not self.linear_layers_replaced:
            replace_linear_layers(self.model, self.batch_size, self.grad_accumulation_steps)
            self.linear_layers_replaced = True

    def _set_modified_linear_activated(self, module, activated):
        """Recursively set activated flag for all ModifiedLinear layers."""
        for child in module.children():
            if isinstance(child, ModifiedLinear):
                child.activated = activated
            else:
                self._set_modified_linear_activated(child, activated)

    def _compile_perturbations_into_weights(self, module):
        """Compile perturbations into original weights and reset perturbations."""
        for child in module.children():
            if isinstance(child, ModifiedLinear):
                with torch.no_grad():
                    # Dequantize perturbations before compiling
                    weight_perturbations, bias_perturbations = child._dequantize_perturbations()

                    compiled_weight_update = torch.einsum('i,ijk->jk', child.scale_manager.scale, weight_perturbations)
                    child.original_weight.data += compiled_weight_update

                    if child.original_bias is not None:
                        compiled_bias_update = torch.einsum('i,ij->j', child.scale_manager.scale, bias_perturbations)
                        child.original_bias.data += compiled_bias_update

                    # Reset quantized perturbations (buffers)
                    child.weight_perturbation_quantized.zero_()
                    child.weight_perturbation_scale.fill_(1.0)
                    child.weight_perturbation_zero_point.zero_()
                    child.scale_manager.scale.data.zero_()

                    if child.original_bias is not None:
                        child.bias_perturbation_quantized.zero_()
                        child.bias_perturbation_scale.fill_(1.0)
                        child.bias_perturbation_zero_point.zero_()
            else:
                self._compile_perturbations_into_weights(child)

    def _setup_scale_optimizer(self):
        """Setup/reset the scale optimizer."""
        scale_params = [p for name, p in self.model.named_parameters() if 'scale' in name]
        self.scale_optimizer = torch.optim.Adam(scale_params, lr=self.scale_lr)

    def train_dataloader(self):
        self.tokenizer = hydra_zen.instantiate(self.tokenizer_config)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = MetaMath(self.tokenizer, self.block_size)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=4,
            persistent_workers=True,
        )

    def training_step(self, batch, batch_idx):
        scale_params = [p for name, p in self.model.named_parameters() if 'scale' in name]
        print('scale value', scale_params[0].mean())
        # breakpoint()
        # Check if this is a compressed batch step
        steps_since_last_meta_batch = batch_idx % self.compress_batches_every
        is_compressed_step = steps_since_last_meta_batch < self.grad_accumulation_steps

        if is_compressed_step:
            self._set_modified_linear_activated(self.model, False)
            self.compressed_batch(batch, accumulation_step=steps_since_last_meta_batch)

            # If this is the last accumulation step, activate and setup optimizer
            if steps_since_last_meta_batch == self.grad_accumulation_steps - 1:
                self._set_modified_linear_activated(self.model, True)
                self._setup_scale_optimizer()
            return

        outputs = self.model(**batch)
        loss = outputs.loss

        scale_params = [p for name, p in self.model.named_parameters() if 'scale' in name]
        grads = torch.autograd.grad(
            loss,
            scale_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )

        # Accumulate gradients
        for param, grad in zip(scale_params, grads):
            if param.grad is None:
                param.grad = grad / self.grad_accumulation_steps
            else:
                param.grad += grad / self.grad_accumulation_steps

        self.grad_accumulation_counter += 1

        # Only step optimizer every grad_accumulation_steps
        if self.grad_accumulation_counter % self.grad_accumulation_steps == 0:
            self.scale_optimizer.step()
            self.scale_optimizer.zero_grad()
            self.grad_accumulation_counter = 0

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def compressed_batch(self, batch, accumulation_step=0):
        device_rank = self.trainer.global_rank if self.trainer else 0
        world_size = self.trainer.world_size if self.trainer else 1

        self._compile_perturbations_into_weights(self.model)

        # Only rank 0 computes perturbations
        if device_rank == 0:
            batch_size = batch["input_ids"].size(0)
            for i in range(batch_size):
                single_batch = {
                    'input_ids': batch["input_ids"][i:i+1],
                    'attention_mask': batch["attention_mask"][i:i+1],
                    'labels': batch["labels"][i:i+1]
                }

                outputs = self.model(**single_batch)
                example_loss = outputs.loss

                original_params = [p for p in self.model.named_parameters() if  'original' in p[0]]
                grads = torch.autograd.grad(
                    example_loss,
                    [p[1] for p in original_params],
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False
                )

                with torch.no_grad():
                    for (name, param), grad in zip(original_params, grads):
                        slot_idx = accumulation_step * self.batch_size + i

                        # Find the corresponding ModifiedLinear module
                        module_path = name.split('.')[:-1]  # Remove 'original_weight' or 'original_bias'
                        target_module = self.model
                        for part in module_path:
                            target_module = getattr(target_module, part)

                        if 'original_weight' in name:
                            # Quantize weight perturbation
                            quantized, scale, zero_point = target_module._quantize_perturbation(grad.data, slot_idx)
                            target_module.weight_perturbation_quantized[slot_idx] = quantized
                            target_module.weight_perturbation_scale[slot_idx] = scale
                            target_module.weight_perturbation_zero_point[slot_idx] = zero_point

                        elif 'original_bias' in name:
                            # Quantize bias perturbation
                            quantized, scale, zero_point = target_module._quantize_perturbation(grad.data, slot_idx)
                            target_module.bias_perturbation_quantized[slot_idx] = quantized
                            target_module.bias_perturbation_scale[slot_idx] = scale
                            target_module.bias_perturbation_zero_point[slot_idx] = zero_point

                        # Set scale parameter via ScaleManager
                        with torch.no_grad():
                            target_module.scale_manager.scale.data[slot_idx] = self.default_lr

        # Broadcast perturbations from rank 0 to all devices
        if world_size > 1:
            self._broadcast_perturbations()

    def _broadcast_perturbations(self):
        """Broadcast perturbation parameters and buffers from rank 0 to all devices."""
        # Broadcast scale parameters
        for name, param in self.model.named_parameters():
            if 'scale' in name and 'perturbation' not in name:
                dist.broadcast(param.data, src=0)

        # Broadcast quantized perturbation buffers
        for name, buffer in self.model.named_buffers():
            if ('weight_perturbation' in name or 'bias_perturbation' in name):
                dist.broadcast(buffer, src=0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
