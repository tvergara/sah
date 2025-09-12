import os

import hydra_zen
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lightning import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.formatters import get_dataset_formatter
from sah.algorithms.llm_finetuning import NetworkConfig, TokenizerConfig


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
        raw_dataset = load_dataset("meta-math/MetaMathQA", split="train[:40000]")
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

        # Store perturbations as float16 tensors (not parameters or buffers)
        self.weight_perturbations = torch.zeros(batch_size * grad_accumulation_steps, *self.original_weight.shape, dtype=torch.float16)

        # Use ScaleManager for FSDP-wrapped scale parameters
        self.scale_manager = ScaleManager(batch_size, grad_accumulation_steps)

        if self.original_bias is not None:
            self.bias_perturbations = torch.zeros(batch_size * grad_accumulation_steps, *self.original_bias.shape, dtype=torch.float16)

        self.activated = False


    def forward(self, x):
        # if not self.activated:
        #     return F.linear(x, self.original_weight, self.original_bias)

        # Use perturbations directly (no quantization)
        weight = self.original_weight + torch.einsum('i,ijk->jk', self.scale_manager.scale, self.weight_perturbations.float())
        bias = None
        if self.original_bias is not None:
            bias = self.original_bias + torch.einsum('i,ij->j', self.scale_manager.scale, self.bias_perturbations.float())

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
        block_size: int = 512,
        default_lr: float = 1e-5,
        compress_batches_every: int = 1,#20,
        scale_lr: float = 0,#1e-5,
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

            # Setup pipeline parallelism if multiple GPUs are available
            if torch.cuda.device_count() > 1:
                self._setup_pipeline_parallelism()
            else:
                # Move perturbation tensors to device for single GPU
                self._move_perturbation_tensors_to_device()

    def _setup_pipeline_parallelism(self):
        """Setup manual pipeline parallelism by splitting model across all available devices."""
        # Find the number of transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            num_layers = len(layers)
            num_gpus = torch.cuda.device_count()

            # Calculate layers per GPU
            layers_per_gpu = num_layers // num_gpus
            remainder = num_layers % num_gpus

            print(f"Setting up manual pipeline parallelism across {num_gpus} GPUs")
            print(f"Total layers: {num_layers}, Base layers per GPU: {layers_per_gpu}")

            # Store split points for reference
            self._pipeline_split_points = []
            layer_idx = 0

            # Distribute layers across GPUs
            for gpu_id in range(num_gpus):
                # Calculate how many layers this GPU gets
                layers_for_this_gpu = layers_per_gpu + (1 if gpu_id < remainder else 0)
                start_layer = layer_idx
                end_layer = layer_idx + layers_for_this_gpu

                print(f"GPU {gpu_id}: Layers {start_layer}-{end_layer-1} ({layers_for_this_gpu} layers)")

                # Move layers to this GPU
                for i in range(start_layer, end_layer):
                    layers[i] = layers[i].to(f'cuda:{gpu_id}')

                # Store split point
                if gpu_id < num_gpus - 1:  # Not the last GPU
                    self._pipeline_split_points.append(end_layer)

                layer_idx = end_layer

            # Place embedding and output components
            # Embedding on first GPU
            if hasattr(self.model.model, 'embed_tokens'):
                self.model.model.embed_tokens = self.model.model.embed_tokens.to('cuda:0')
                print("Embedding on GPU 0")

            # Norm and head on last GPU
            last_gpu = num_gpus - 1
            if hasattr(self.model.model, 'norm'):
                self.model.model.norm = self.model.model.norm.to(f'cuda:{last_gpu}')
                print(f"Norm on GPU {last_gpu}")
            if hasattr(self.model, 'lm_head'):
                self.model.lm_head = self.model.lm_head.to(f'cuda:{last_gpu}')
                print(f"LM Head on GPU {last_gpu}")

            # Mark as pipeline model
            self._is_pipeline_model = True
            self._num_pipeline_gpus = num_gpus

        else:
            print("Warning: Could not find model layers for pipeline splitting")
            self._move_perturbation_tensors_to_device()

    def on_train_start(self):
        """Lightning hook called before training starts - move tensors to device."""
        if self.trainer.num_devices <= 1:
            self._move_perturbation_tensors_to_device()

    def _move_perturbation_tensors_to_device(self):
        """Move perturbation tensors to the same device as model parameters."""
        device = next(self.model.parameters()).device
        for child in self.model.modules():
            if isinstance(child, ModifiedLinear):
                child.weight_perturbations = child.weight_perturbations.to(device)

                if child.original_bias is not None:
                    child.bias_perturbations = child.bias_perturbations.to(device)

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
                    # Use perturbations directly (no dequantization needed)
                    compiled_weight_update = torch.einsum('i,ijk->jk', child.scale_manager.scale, child.weight_perturbations.float())
                    child.original_weight.data += compiled_weight_update

                    if child.original_bias is not None:
                        compiled_bias_update = torch.einsum('i,ij->j', child.scale_manager.scale, child.bias_perturbations.float())
                        child.original_bias.data += compiled_bias_update

                    # Reset perturbations
                    child.weight_perturbations.zero_()
                    child.scale_manager.scale.data.zero_()

                    if child.original_bias is not None:
                        child.bias_perturbations.zero_()
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
            # self.trainer.global_step += 1
            self.grad_accumulation_counter = 0

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def compressed_batch(self, batch, accumulation_step=0):
        self._compile_perturbations_into_weights(self.model)

        # Each GPU computes perturbations for its own layers
        batch_size = batch["input_ids"].size(0)

        loss = 0
        for i in range(batch_size):
            single_batch = {
                'input_ids': batch["input_ids"][i:i+1],
                'attention_mask': batch["attention_mask"][i:i+1],
                'labels': batch["labels"][i:i+1]
            }

            outputs = self.model(**single_batch)
            example_loss = outputs.loss
            loss += example_loss.item() / batch_size

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
                        # Store weight perturbation directly as float16
                        target_module.weight_perturbations[slot_idx] = grad.data.to(torch.float16)

                    elif 'original_bias' in name:
                        # Store bias perturbation directly as float16
                        target_module.bias_perturbations[slot_idx] = grad.data.to(torch.float16)

        # With pipeline parallelism, each GPU handles its own layers independently
        # No need for broadcasting perturbations
        self._initialize_scale_parameters()
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

    def _initialize_scale_parameters(self):
        """Initialize scale parameters on each rank's shard."""
        for child in self.model.modules():
            if isinstance(child, ModifiedLinear):
                with torch.no_grad():
                    child.scale_manager.scale.data.fill_(-self.default_lr)


    def configure_optimizers(self):
        pass

    def on_train_batch_end(self, outputs, batch, batch_idx=None):
        checkpoint_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break

        if checkpoint_callback is None:
            return

        save_every = checkpoint_callback._every_n_train_steps
        if save_every > 0 and (batch_idx + 1) % save_every == 0:
            os.makedirs(checkpoint_callback.dirpath, exist_ok=True)
            filepath = os.path.join(checkpoint_callback.dirpath, "last.ckpt")
            self.trainer.save_checkpoint(filepath)
