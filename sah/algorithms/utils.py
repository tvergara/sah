import itertools
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def load_weights_from_checkpoint(model, path, model_name='network'):
    ckpt = torch.load(path + '.ckpt', map_location=lambda storage, loc: storage)
    state_dict = ckpt["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace(f"{model_name}.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

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
    def __init__(self, base_path: str, max_length: int = 512, mode='train', tokenizer=None):
        fp = Path(base_path) / f"{mode}.txt"
        if not fp.exists():
            raise FileNotFoundError(fp)

        raw_sents   = [ln.strip().split()[:max_length] for ln in fp.read_text().splitlines()]

        if not tokenizer:
            vocab_ctr   = Counter(itertools.chain.from_iterable(raw_sents))
            self.tokenizer    = TinyTokenizer(vocab_ctr)
        else:
            self.tokenizer = tokenizer

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
    unsafe_hash=True,
    populate_full_signature=True,
)
class GrammarConfig:
    base_path: str
    max_length: int = 512

@dataclass(frozen=True, unsafe_hash=True)
class TokenizerConfig:
    out_path: str
