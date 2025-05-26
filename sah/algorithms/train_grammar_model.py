import itertools
from collections import Counter
from pathlib import Path

import hydra_zen
import torch
from lightning import LightningModule
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, base_path: str, max_length: int = 512):
        fp = Path(base_path) / "train.txt"
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

class GrammarTrainer(LightningModule):
    def __init__(
        self,
        grammar_config: GrammarConfig,
        vocab_size: int = 50,
        emb_dim: int = 32,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.save_hyperparameters()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder   = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.grammar_config = grammar_config
        self.dataset = hydra_zen.instantiate(self.grammar_config)
        self.pad  = self.dataset.pad_id
        self.tokenizer = self.dataset.tokenizer
        self.embedding  = nn.Embedding(
            self.tokenizer.vocab_size, emb_dim, padding_idx=self.tokenizer.pad_id
        )

    def training_step(self, batch, batch_idx):
        x, _ = batch                 # x : (B, L)
        inp  = x[:, :-1]             # drop last token
        targ = x[:, 1:]              # predict this
        print(targ)

        emb = self.embedding(inp)
        loss = emb.sum()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch                 # x : (B, L)
        inp  = x[:, :-1]             # drop last token
        targ = x[:, 1:]              # predict this
        print(targ)
        emb = self.embedding(inp)
        loss = emb.sum()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=32,
                          collate_fn=lambda b: collate(b, self.pad))
