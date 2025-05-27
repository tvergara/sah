import itertools
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from sah.algorithms.networks.transformer import Transformer


class TinyTokenizer:
    def __init__(self, counter: Counter):
        specials  = ["<PAD>", "<UNK>"]
        self.itos = specials + [tok for tok, c in counter.items() if c >= 1]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi["<PAD>"]
        self.vocab_size = len(self.itos)

    def encode(self, tokens):           # list[str] -> list[int]
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in tokens]

    def __len__(self): return len(self.itos)

class GrammarDataset(Dataset):
    def __init__(self, base_path: str, max_length: int = 512, mode='train'):
        fp = Path(base_path) / f"{mode}.txt"
        if not fp.exists():
            raise FileNotFoundError(fp)

        raw_sents   = [ln.strip().split()[:max_length] for ln in fp.read_text().splitlines()]
        vocab_ctr   = Counter(itertools.chain.from_iterable(raw_sents))
        self.tokenizer    = TinyTokenizer(vocab_ctr)
        self.pad_id = self.tokenizer.pad_id
        self.seqs   = [torch.tensor(self.tokenizer.encode(s)) for s in raw_sents]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]

def collate(batch, pad_id):
    xs = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    mask = (xs != pad_id)
    return xs, mask

@hydra_zen.hydrated_dataclass(
    target=GrammarDataset,
    frozen=True,
    unsafe_hash=True,
    populate_full_signature=True,
)
class GrammarConfig:
    base_path: str
    max_length: int = 512

@dataclass(frozen=True, unsafe_hash=True)
class TokenizerConfig:
    out_path: str

class GrammarTrainer(LightningModule):
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
        self.tokenizer_config = tokenizer_config
        self.dataset = hydra_zen.instantiate(self.grammar_config)
        self.test_dataset = hydra_zen.instantiate(self.grammar_config, mode='test')
        self.pad  = self.dataset.pad_id
        self.tokenizer = self.dataset.tokenizer
        self.transformer = Transformer(self.tokenizer.vocab_size)
        with open(tokenizer_config.out_path, "wb") as f:
            pickle.dump(self.tokenizer, f)


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
