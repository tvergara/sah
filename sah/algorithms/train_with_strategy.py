
from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from sah.algorithms.networks.transformer import TransformerConfig
from sah.algorithms.ngram_model import get_ngram
from sah.algorithms.utils import GrammarConfig, GrammarDataset, TokenizerConfig, collate


@dataclass(frozen=True, unsafe_hash=True)
class GeneralConfig:
    data_variants_path: str
    pretraining_budget: int
    strategy: str

class TrainWithStrategy(LightningModule):
    def __init__(
        self,
        finetuning_config: GrammarConfig,
        tokenizer_config: TokenizerConfig,
        transformer_config: TransformerConfig,
        general_config: GeneralConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.grammar_config = finetuning_config
        self.tokenizer_config = tokenizer_config
        self.general_config = general_config
        self.dataset = hydra_zen.instantiate(self.grammar_config)
        self.tokenizer = self.dataset.tokenizer
        self.test_dataset = hydra_zen.instantiate(self.grammar_config, mode='test', tokenizer=self.tokenizer)
        self.pad  = self.dataset.pad_id
        self.tokenizer = self.dataset.tokenizer
        self.transformer = hydra_zen.instantiate(
            transformer_config,
            vocab_size = self.tokenizer.vocab_size,
        )
        self.build_pretraining_dataset()
        self.dataset = ConcatDataset([self.pretraining_dataset, self.dataset])

    def build_pretraining_dataset(self):
        variant_root = Path(self.general_config.data_variants_path)
        variant_paths = sorted(p for p in variant_root.iterdir() if p.is_dir())

        self.variant_datasets = [
            GrammarDataset(str(p), tokenizer=self.tokenizer)
            for p in variant_paths
        ]
        pt_dataset = ConcatDataset(self.variant_datasets)
        budget = self.general_config.pretraining_budget
        if self.general_config.strategy == 'ngram':
            ngram = get_ngram(self.dataset, device='cuda')
            def score(example):
                seq, mask = example
                toks = seq[mask.bool()].to(ngram.device)
                if toks.numel() < ngram.n:
                    return -float("inf")

                logL = ngram.score_batch(toks.unsqueeze(0)).item()  # (1,) → scalar
                return logL / toks.numel()

            scores = [score(pt_dataset[i]) for i in tqdm(range(len(pt_dataset)))]
            idx    = (torch.tensor(scores)
                           .topk(min(budget, len(scores))).indices.tolist())
        else:
            idx = torch.randperm(len(pt_dataset))[:budget].tolist()
        self.pretraining_dataset = Subset(pt_dataset, idx)

    def training_step(self, batch, batch_idx):
        x, mask = batch                 # x : (B, L)
        inputs  = x[:, :-1]
        targets = x[:, 1:]

        logits = self.transformer(inputs)
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.reshape(-1),
            reduction='mean',
            ignore_index=self.pad
        )

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, mask = batch                 # (B, L)
        inputs  = x[:, :-1]
        targets = x[:, 1:]
        logits = self.transformer(inputs)  # (B, L-1, V)
        target_mask = mask[:, 1:]

        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.reshape(-1),
            reduction='mean',
            ignore_index=self.pad
        )
        self.log("test/loss", loss, prog_bar=True)

        preds    = logits.argmax(-1)
        correct  = ((preds == targets) & (target_mask.bool())).sum()
        n_tokens = target_mask.sum()
        acc = correct.float() / n_tokens
        self.log("accuracy", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32, collate_fn=collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, collate_fn=collate)
