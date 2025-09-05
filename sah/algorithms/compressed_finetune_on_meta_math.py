import hydra_zen
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.formatters import get_dataset_formatter
from sah.algorithms.llm_finetuning import NetworkConfig, TokenizerConfig


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
    def __init__(self, original_linear, batch_size):
        super().__init__()

        self.original_weight = original_linear.weight
        setattr(self.original_weight, 'original', True)

        if original_linear.bias is not None:
            self.original_bias = original_linear.bias
            setattr(self.original_bias, 'original', True)
        else:
            self.original_bias = None

        self.weight_perturbation = nn.Parameter(torch.zeros(batch_size, *self.original_weight.shape), requires_grad=False)
        self.scale = nn.Parameter(torch.zeros(batch_size), requires_grad=True)

        if self.original_bias is not None:
            self.bias_perturbation = nn.Parameter(torch.zeros(batch_size, *self.original_bias.shape), requires_grad=False)

        self.activated = False

    def forward(self, x):
        if not self.activated:
            return F.linear(x, self.original_weight, self.original_bias)

        weight = self.original_weight + torch.einsum('i,ijk->jk', self.scale, self.weight_perturbation)
        bias = None
        if self.original_bias is not None:
            bias = self.original_bias + torch.einsum('i,ij->j', self.scale, self.bias_perturbation)

        return F.linear(x, weight, bias)

def replace_linear_layers(model, batch_size, original_params=[]):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            linear = ModifiedLinear(module, batch_size)
            setattr(model, name, linear)
        else:
            replace_linear_layers(module, batch_size, original_params=original_params)
    return original_params

class CompressedFinetuneOnMetaMath(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        pretrained_config: NetworkConfig,
        batch_size: int = 8,
        block_size: int = 512,
        default_lr: float = 1e-4,
        compress_batches_every: int = 100,
        scale_lr: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = hydra_zen.instantiate(pretrained_config)
        replace_linear_layers(self.model, batch_size)
        self.batch_size = batch_size
        self.block_size = block_size

        self.tokenizer_config = tokenizer_config
        self.automatic_optimization = False
        self.default_lr = default_lr
        self.compress_batches_every = compress_batches_every
        self.scale_lr = scale_lr
        self.scale_optimizer = None

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
                    compiled_weight_update = torch.einsum('i,ijk->jk', child.scale, child.weight_perturbation)
                    child.original_weight.data += compiled_weight_update

                    if child.original_bias is not None:
                        compiled_bias_update = torch.einsum('i,ij->j', child.scale, child.bias_perturbation)
                        child.original_bias.data += compiled_bias_update

                    child.weight_perturbation.data.zero_()
                    child.scale = nn.Parameter(torch.zeros_like(child.scale))
                    if child.original_bias is not None:
                        child.bias_perturbation.data.zero_()
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
        if batch_idx % self.compress_batches_every == 0:
            self._set_modified_linear_activated(self.model, False)
            self.compressed_batch(batch)
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

        self.scale_optimizer.zero_grad()
        for param, grad in zip(scale_params, grads):
            param.grad = grad
        self.scale_optimizer.step()

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def compressed_batch(self, batch):
        self._compile_perturbations_into_weights(self.model)

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
                    target_name = name.replace('original_weight', 'weight_perturbation').replace('original_bias', 'bias_perturbation')
                    target_param = dict(self.model.named_parameters())[target_name]
                    target_param.data[i] = grad.data.clone()
                    target_name = name.replace('original_weight', 'scale')
                    target_param = dict(self.model.named_parameters())[target_name]
                    target_param.data[i].copy_(self.default_lr)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
