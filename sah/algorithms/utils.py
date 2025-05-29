import itertools
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import hydra_zen
import torch
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
    def __init__(self, counter: Counter, max_length=512):
        specials  = ["<PAD>", "<UNK>"]
        self.itos = specials + [tok for tok, c in counter.items() if c >= 1]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi["<PAD>"]
        self.unk_id = self.stoi["<UNK>"]
        self.vocab_size = len(self.itos)
        self.max_length = max_length

    def encode(self, tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens][:self.max_length]

        pad_needed = self.max_length - len(ids)
        if pad_needed:
            ids.extend([self.pad_id] * pad_needed)

        ids_tensor = torch.tensor(ids, dtype=torch.long)

        mask = torch.ones(self.max_length, dtype=torch.bool)
        if pad_needed:
            mask[-pad_needed:] = 0
        return ids_tensor, mask

    def __len__(self): return len(self.itos)

class GrammarDataset(Dataset):
    def __init__(self, base_path: str, max_length: int = 512, mode='train', tokenizer=None):
        fp = Path(base_path) / f"{mode}.txt"
        if not fp.exists():
            raise FileNotFoundError(fp)

        raw_sents   = [ln.strip().split()[:max_length] for ln in fp.read_text().splitlines()]

        if not tokenizer:
            vocab_ctr = Counter(itertools.chain.from_iterable(raw_sents))
            self.tokenizer = TinyTokenizer(vocab_ctr, max_length=max_length)
        else:
            self.tokenizer = tokenizer

        self.pad_id = self.tokenizer.pad_id
        self.data = [self.tokenizer.encode(s) for s in raw_sents]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def collate(batch):
    xs, masks = zip(*batch)
    return torch.stack(xs, 0), torch.stack(masks, 0)

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
