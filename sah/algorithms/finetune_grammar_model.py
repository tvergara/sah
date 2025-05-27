
import pickle

import hydra_zen
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.utils.data import DataLoader

from sah.algorithms.networks.transformer import Transformer
from sah.algorithms.utils import GrammarConfig, TokenizerConfig, collate


class GrammarFinetuner(LightningModule):
    def __init__(
        self,
        grammar_config: GrammarConfig,
        tokenizer_config: TokenizerConfig,
        vocab_size: int = 50,
        emb_dim: int = 32,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.save_hyperparameters()
        self.grammar_config = grammar_config
        with open(tokenizer_config.out_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.dataset = hydra_zen.instantiate(self.grammar_config, tokenizer=self.tokenizer)
        self.test_dataset = hydra_zen.instantiate(self.grammar_config, mode='test', tokenizer=self.tokenizer)
        self.pad  = self.dataset.pad_id
        self.transformer = Transformer(self.tokenizer.vocab_size)

    def training_step(self, batch, batch_idx):
        x, _ = batch                 # x : (B, L)
        inputs  = x[:, :-1]             # drop last token
        targets = x[:, 1:]              # predict this

        logits = self.transformer(inputs)
        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targets.reshape(-1),
            reduction='mean',
            ignore_index=self.pad
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch                 # (B, L)
        inp, targ = x[:, :-1], x[:, 1:]
        logits = self.transformer(inp)  # (B, L-1, V)

        loss = F.cross_entropy(
            logits.view(-1, self.tokenizer.vocab_size),
            targ.reshape(-1),
            reduction='sum',
            ignore_index=self.pad,
        )

        n_tokens = (targ != self.pad).sum()

        ce = loss / n_tokens
        self.log("test/loss", ce, on_epoch=True, prog_bar=True)

        self.log("perplexity", torch.exp(ce),
                 prog_bar=True, sync_dist=True)

        preds = logits.argmax(-1)
        correct = ((preds == targ) & (targ != self.pad)).sum()
        acc = correct.float() / n_tokens
        self.log("accuracy", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32,
                          collate_fn=lambda b: collate(b, self.pad))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32,
                          collate_fn=lambda b: collate(b, self.pad))
