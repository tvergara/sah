import pickle
from dataclasses import dataclass

import hydra_zen
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import ConcatDataset, DataLoader, SequentialSampler, Subset

from sah.algorithms.networks.transformer import TransformerConfig
from sah.algorithms.utils import GrammarDataset, TokenizerConfig, collate


@dataclass(frozen=True, unsafe_hash=True)
class GrammarConfig:
    base_path: str
    grammars: list
    order_strategy: str
    max_length: int = 512

class CurriculumGrammarTrainer(LightningModule):
    def __init__(
        self,
        grammar_config: GrammarConfig,
        tokenizer_config: TokenizerConfig,
        transformer_config: TransformerConfig,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.grammar_config = grammar_config
        self.tokenizer_config = tokenizer_config


        self.tokenizer = None
        datasets = []
        for grammar in grammar_config.grammars:
            dataset = GrammarDataset(base_path=f"{grammar_config.base_path}-{grammar}", tokenizer=self.tokenizer)
            if self.tokenizer is None:
                self.tokenizer = dataset.tokenizer
                self.pad  = dataset.pad_id
            datasets.append(dataset)


        if grammar_config.order_strategy == 'pretrain':
            pretrain_dataset = ConcatDataset(datasets[:-1])
            perm = torch.randperm(len(pretrain_dataset))
            pretrain_dataset = Subset(pretrain_dataset, perm.tolist())
            self.dataset = ConcatDataset([pretrain_dataset, datasets[-1]])
        elif grammar_config.order_strategy == 'gradual':
            self.dataset = ConcatDataset(datasets)
        elif grammar_config.order_strategy == 'shuffle':
            self.dataset = ConcatDataset(datasets)
            perm = torch.randperm(len(self.dataset))
            self.dataset = Subset(self.dataset, perm.tolist())

        self.test_dataset = GrammarDataset(base_path=f"{grammar_config.base_path}-{grammar_config.grammars[-1]}", tokenizer=self.tokenizer, mode='test')

        self.transformer = hydra_zen.instantiate(
            transformer_config,
            vocab_size = self.tokenizer.vocab_size,
        )
        with open(tokenizer_config.out_path, "wb") as f:
            pickle.dump(self.tokenizer, f)


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
        return DataLoader(
            self.dataset,
            batch_size=32,
            collate_fn=collate,
            sampler=SequentialSampler(self.dataset)
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, collate_fn=collate)
