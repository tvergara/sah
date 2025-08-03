import pickle
from dataclasses import dataclass

import hydra_zen
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

from sah.algorithms.networks.transformer import TransformerConfig
from sah.algorithms.utils import (
    TokenizerConfig,
    load_weights_from_checkpoint,
)


@dataclass(frozen=True, unsafe_hash=True)
class CheckpointConfig:
    path: str

class SyntheticDataset(Dataset):
    def __init__(self, teacher_model, vocab_size: int, seq_length: int, dataset_size: int = 1000):
        self.teacher_model = teacher_model
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.dataset_size = dataset_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return idx


def synthetic_collate_fn(teacher_model, vocab_size: int, seq_length: int):
    def collate(batch_indices):
        batch_size = len(batch_indices)
        device = next(teacher_model.parameters()).device

        # Initialize batch with random starting tokens
        input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=device)
        input_ids[:, 0] = torch.randint(1, vocab_size - 1, (batch_size,), device=device)

        # Generate sequences token by token in parallel
        with torch.no_grad():
            for i in range(1, seq_length):
                # Get logits for all sequences in batch at position i-1
                outputs = teacher_model(input_ids[:, :i])[:, -1]  # [batch_size, vocab_size]

                # Sample next tokens for all sequences
                dist = torch.distributions.Categorical(logits=outputs)
                next_tokens = dist.sample()  # [batch_size]
                input_ids[:, i] = next_tokens

        # Get teacher labels for all sequences
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids)  # [batch_size, seq_length, vocab_size]

        return {
            'input_ids': input_ids,
            'teacher_logits': teacher_outputs
        }

    return collate


class DistilModel(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        transformer_config: TransformerConfig,
        pretrained_config: CheckpointConfig,
        finetuned_config: CheckpointConfig,
        seq_length: int = 100,
        dataset_size: int = 10000,
        batch_size: int = 64
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load tokenizer
        with open(tokenizer_config.out_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        # Initialize student model (pretrained)
        self.model = hydra_zen.instantiate(transformer_config, vocab_size=self.tokenizer.vocab_size)
        load_weights_from_checkpoint(self.model, pretrained_config.path, model_name='transformer')

        # Initialize teacher model (finetuned)
        self.teacher = hydra_zen.instantiate(transformer_config, vocab_size=self.tokenizer.vocab_size)
        load_weights_from_checkpoint(self.teacher, finetuned_config.path, model_name='transformer')
        self.teacher.eval()

        # Store hyperparameters
        self.seq_length = seq_length
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = SyntheticDataset(
            teacher_model=self.teacher,
            vocab_size=self.tokenizer.vocab_size,
            seq_length=self.seq_length,
            dataset_size=self.dataset_size
        )

        collate_fn = synthetic_collate_fn(
            teacher_model=self.teacher,
            vocab_size=self.tokenizer.vocab_size,
            seq_length=self.seq_length
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=0  # Keep 0 to avoid issues with model sharing across processes
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        teacher_logits = batch['teacher_logits']

        # Get student predictions
        student_logits = self.model(input_ids)

        # Compute KL divergence loss
        temperature = 1.0
        log_probs_student = F.log_softmax(student_logits / temperature, dim=-1)
        probs_teacher = F.softmax(teacher_logits / temperature, dim=-1)

        loss = F.kl_div(
            log_probs_student.view(-1, log_probs_student.size(-1)),
            probs_teacher.view(-1, probs_teacher.size(-1)),
            reduction="batchmean"
        ) * (temperature ** 2)

        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)
