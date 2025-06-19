"""
gpu_ngram.py
============

Same API as your original code, but the model is held in one dense
tensor of shape (V**(n-1), V) – perfect for GPU batch scoring.

Usage
-----
>>> model = get_ngram(my_dataset, n=4, alpha=0.5, pad_id=-1, device="cuda")
>>> # sentence batch: (B, L) torch.LongTensor on the same device
>>> logL = model.score_batch(batch)          # (B,) total log-likelihoods
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["NGramModel", "get_ngram", "sequences_from_dataset"]

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
PAD_DEFAULT = -1


def context_indices(tokens, n: int, V: int):
    """
    tokens : 1-D list[int] **or** LongTensor
    returns: (row_idx, targets)  — both 1-D LongTensors of length L-n+1
    """
    if not torch.is_tensor(tokens):
        tokens = torch.tensor(tokens, dtype=torch.long)
    L = tokens.size(0)
    if L < n:
        return None, None                       # sequence too short

    # (L-n+1, n-1) sliding windows of previous tokens
    ctx = tokens.unfold(0, n - 1, 1)            # (L-n+1, n-1)

    bases = (V ** torch.arange(n - 2, -1, -1, device=tokens.device)).long()
    rows  = (ctx * bases).sum(-1)               # (L-n+1,)

    targets = tokens[n - 1 :]                   # y_t  (L-n+1,)
    return rows, targets


# --------------------------------------------------------------------------
# Dense, GPU-friendly model
# --------------------------------------------------------------------------
class NGramModel:
    """Add-α smoothed *n*-gram stored as one (contexts × V) tensor of **log-probs**."""

    def __init__(self, table_log: torch.Tensor, n: int, vocab_size: int):
        """
        Parameters
        ----------
        table_log : float32 tensor (V**(n-1), V) on desired device
        """
        self.table_log = table_log  # row = context, col = next-token
        self.n = n
        self.V = vocab_size
        self.device = table_log.device

    # --- single prediction -------------------------------------------------
    @torch.no_grad()
    def predict_log_probs(self, context: Sequence[int]) -> torch.Tensor:
        """Return log P(· | context)  – shape (V,)."""
        ctx = context[-(self.n - 1) :]  # keep last n-1 tokens
        idx = 0
        for tok in ctx:
            idx = idx * self.V + tok
        return self.table_log[idx]

    def predict_probs(self, context):  # API parity
        return self.predict_log_probs(context).exp()

    # --- batched scoring ---------------------------------------------------
    @torch.no_grad()
    def score_batch(self, batch_tok: torch.Tensor) -> torch.Tensor:
        """
        batch_tok : (B, L) LongTensor on self.device
        Returns    : (B,) log-likelihood of each sentence (ignores first n-1 tokens)
        """
        B, L = batch_tok.shape
        if L < self.n:
            raise ValueError("Sentence shorter than n-gram order.")
        # build context indices with pure tensor arithmetic
        V = self.V
        idx = torch.zeros((B, L - self.n + 1), device=batch_tok.device, dtype=torch.long)
        for k in range(self.n - 1):
            idx = idx * V + batch_tok[:, k : L - self.n + 1 + k]
        log_cond = self.table_log[idx]                         # (B, L-n+1, V)
        targets = batch_tok[:, self.n - 1 :].unsqueeze(-1)     # (B, L-n+1, 1)
        log_p = log_cond.gather(-1, targets).squeeze(-1)       # (B, L-n+1)
        return log_p.sum(-1)                                   # (B,)

    # --- I/O helpers -------------------------------------------------------
    def to(self, device):  # keeps chaining behaviour from original class
        self.table_log = self.table_log.to(device)
        self.device = self.table_log.device
        return self

    # --- training ----------------------------------------------------------
    @classmethod
    def from_sequences(
        cls,
        sequences: list[list[int]],
        n: int = 4,
        alpha: float = 1.0,
        vocab_size: int | None = None,
        device: str | torch.device = "cpu",
    ):
        if vocab_size is None:
            vocab_size = max(tok for seq in sequences for tok in seq) + 1
        V = vocab_size
        # counts + α smoothing  →  float32
        table = torch.full((V ** (n - 1), V), alpha, dtype=torch.float32)

        for seq in sequences:
            rows, tgt = context_indices(seq, n, V)      # tensors or (None, None)
            if rows is None:
                continue                                # skipped short sequence
            # simple Python loop is fine at training time
            for r, y in zip(rows.tolist(), tgt.tolist()):
                table[r, y] += 1
        table /= table.sum(-1, keepdim=True)  # normalise
        return cls(table.log().to(device), n, V)


# --------------------------------------------------------------------------
# Dataset integration (same as before, lightly trimmed)
# --------------------------------------------------------------------------
def sequences_from_dataset(dataset, pad_id: int = PAD_DEFAULT) -> list[list[int]]:
    """Extract unpadded sequences from a (seq, mask) dataset."""
    from torch.utils.data import DataLoader

    seqs: list[list[int]] = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for seq, mask in loader:
        seq = seq.squeeze(0)
        mask = mask.squeeze(0)
        seqs.append(seq[mask.bool()].tolist())
    return seqs


def get_ngram(
    dataset,
    n: int = 4,
    alpha: float = 1.0,
    pad_id: int = PAD_DEFAULT,
    device: str | torch.device = "cpu",
) -> NGramModel:
    """High-level helper exactly like before."""
    seqs = sequences_from_dataset(dataset, pad_id)
    V = max(tok for s in seqs for tok in s) + 1
    return NGramModel.from_sequences(seqs, n=n, alpha=alpha, vocab_size=V, device=device)


# --------------------------------------------------------------------------
# Quick sanity test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    toy = [[1, 2, 3, 4, 5, 2, 3, 6]]
    mdl = NGramModel.from_sequences(toy, n=4, alpha=1.0)
    sent = torch.tensor([[1, 2, 3, 4, 5, 2, 3, 6]])
    print("log-L:", mdl.score_batch(sent))
