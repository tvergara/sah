from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader, Dataset

from sah.algorithms.llm_finetuning import NetworkConfig, TokenizerConfig
from sah.algorithms.utils import (
    load_weights_from_checkpoint,
)


@dataclass(frozen=True, unsafe_hash=True)
class CheckpointConfig:
    path: str

class PreGeneratedDataset(Dataset):
    def __init__(self, data_dir: str, teacher_model, tokenizer):
        self.data_dir = Path(data_dir)
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer

        # Load all batch files
        self.batch_files = sorted(list(self.data_dir.glob("batch_*.txt")))
        if not self.batch_files:
            raise ValueError(f"No batch files found in {data_dir}")

        # Count total sequences across all files
        self.total_sequences = 0
        for batch_file in self.batch_files:
            with open(batch_file) as f:
                self.total_sequences += sum(1 for _ in f)

        print(f"Loaded {len(self.batch_files)} batch files with {self.total_sequences} total sequences")

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        # Find which file and line contains this index
        current_idx = 0
        for batch_file in self.batch_files:
            with open(batch_file) as f:
                lines = f.readlines()
                if current_idx + len(lines) > idx:
                    # Found the right file, get the sequence
                    line_idx = idx - current_idx
                    token_ids = [int(x) for x in lines[line_idx].strip().split()]
                    return torch.tensor(token_ids, dtype=torch.long)
                current_idx += len(lines)

        raise IndexError(f"Index {idx} out of range")


def pregenerated_collate_fn(teacher_model):
    def collate(batch_sequences):
        # batch_sequences is a list of tensors with token IDs
        device = next(teacher_model.parameters()).device

        # Stack the sequences into a batch
        input_ids = torch.stack(batch_sequences).to(device)

        # Get teacher labels for all sequences
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids)
            teacher_logits = teacher_outputs.logits  # [batch_size, seq_length, vocab_size]

        return {
            'input_ids': input_ids,
            'teacher_logits': teacher_logits
        }

    return collate


class DistilPretrainedModel(LightningModule):
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        pretrained_config: NetworkConfig,
        finetuned_config: NetworkConfig,
        checkpoint_config: CheckpointConfig,
        data_dir: str,
        batch_size: int = 16
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load tokenizer
        self.tokenizer = hydra_zen.instantiate(tokenizer_config)

        # Initialize student model (pretrained base model)
        self.model = hydra_zen.instantiate(pretrained_config, torch_dtype=torch.bfloat16)

        # initialize teacher model (finetuned chat model)
        self.teacher = hydra_zen.instantiate(finetuned_config, torch_dtype=torch.bfloat16)
        self.teacher.eval()

        # Ensure teacher doesn't get gradient updates
        for param in self.teacher.parameters():
            param.requires_grad = False

        # initialize teacher model (finetuned chat model)
        objective = deepcopy(self.teacher)
        load_weights_from_checkpoint(objective, checkpoint_config.path, model_name='model')

        diffs = [p_fine - p_pre for p_fine, p_pre in zip(objective.parameters(), self.model.parameters())]
        self.non_ignorable = [torch.abs(pd) > 0.005 for pd in diffs]
        self.non_ig = sum([p.float().sum() for p in self.non_ignorable])
        self.diffs = [p > 0 for p in diffs]
        # self.positives = [(pd * ig.float()) > 0 for pd, ig in zip(diffs, non_ignorable)]
        # self.positives = [p.to('cuda') for p in self.positives]
        self.base_model_copy = deepcopy(self.model)

        # Store hyperparameters
        self.data_dir = data_dir
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = PreGeneratedDataset(
            data_dir=self.data_dir,
            teacher_model=self.teacher,
            tokenizer=self.tokenizer
        )

        collate_fn = pregenerated_collate_fn(
            teacher_model=self.teacher
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
        student_outputs = self.model(input_ids)
        student_logits = student_outputs.logits

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
        diffs = [((p_fine - p_pre) > 0).to('cpu') for p_fine, p_pre in zip(self.model.parameters(), self.base_model_copy.parameters())]
        comparisons = [a == b for a, b in zip(diffs, self.diffs)]
        positives = sum([(a & b).sum() for a, b in zip(comparisons, self.non_ignorable)])
        self.log('positives', positives / self.non_ig)
        print('positives', positives.item() / self.non_ig)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer

    def state_dict(self):
        state = super().state_dict()
        teacher_keys = [key for key in state.keys() if key.startswith('teacher.')]
        for key in teacher_keys:
            del state[key]
        return state
