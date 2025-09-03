
from dataclasses import dataclass

import hydra_zen
import torch
import torch.nn.functional as F
from datasets import load_dataset
from lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling

from sah.algorithms.formatters import get_dataset_formatter
from sah.algorithms.llm_finetuning import NetworkConfig, TokenizerConfig
from sah.algorithms.utils import load_weights_from_checkpoint


@dataclass(frozen=True, unsafe_hash=True)
class CheckpointConfig:
    path: str

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


class DistilOnMetaMath(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        pretrained_config: NetworkConfig,
        checkpoint_config: CheckpointConfig,
        batch_size: int = 8,
        block_size: int = 128,
        temperature: float = 4.0,
        alpha: float = 0.7,
        teacher_gpu: int = 0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = hydra_zen.instantiate(pretrained_config)
        self.teacher_model = hydra_zen.instantiate(pretrained_config)
        load_weights_from_checkpoint(self.teacher_model, checkpoint_config.path, model_name='model')

        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()

        self.batch_size = batch_size
        self.block_size = block_size
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_gpu = teacher_gpu
        self.teacher_device = None

        self.tokenizer_config = tokenizer_config

    def setup(self, stage=None):
        pass

    def on_train_start(self):
        # Set up teacher device and model
        self.teacher_device = torch.device(f"cuda:{self.teacher_gpu}")
        self.teacher_model = self.teacher_model.to(self.teacher_device)

        # Get available GPUs excluding teacher GPU
        num_gpus = torch.cuda.device_count()
        student_gpus = [i for i in range(num_gpus) if i != self.teacher_gpu]
        print(f"Total GPUs: {num_gpus}, Teacher GPU: {self.teacher_gpu}, Student GPUs: {student_gpus}")

        if len(student_gpus) > 1:
            # Use DataParallel for student model on remaining GPUs
            self.model = torch.nn.DataParallel(self.model, device_ids=student_gpus)
            # Move to primary student GPU
            self.model = self.model.to(f"cuda:{student_gpus[0]}")
        elif len(student_gpus) == 1:
            # Single GPU for student
            self.model = self.model.to(f"cuda:{student_gpus[0]}")

        print(f"Final devices - Teacher: {self.teacher_device}, Student: {next(self.model.parameters()).device}")

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
        outputs = self.model(**batch)

        teacher_batch = {k: v.to(self.teacher_device) for k, v in batch.items()}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_batch)

        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits.to(student_logits.device)

        hard_loss = outputs.loss

        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        mask = batch['labels'] != -100

        soft_loss = F.kl_div(
            student_log_probs[mask],
            teacher_probs[mask],
            reduction='batchmean'
        ) * (self.temperature ** 2)

        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/hard_loss", hard_loss, on_step=True, on_epoch=False)
        self.log("train/soft_loss", soft_loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer
