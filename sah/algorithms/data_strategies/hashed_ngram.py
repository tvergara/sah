# build_hashed_ngram_counts_uint8.py  ——  counts table, final dtype = uint8
from __future__ import annotations

from collections.abc import Sequence

import torch


def get_hashed_ngram(
    sequences: Sequence[Sequence[int]],
    n: int = 4,
    entries: int = 2000,
    outputs: int = 22,
    device: str | torch.device = "cpu",
    vocab_size: int = 22
) -> torch.Tensor:              # returns (entries, outputs) uint8 on device
    # --- infer vocab ------------------------------------------------------
    V = vocab_size
    E = entries
    out = outputs

    # --- INT32 scratch table ---------------------------------------------
    counts32 = torch.zeros((E, out), dtype=torch.int32, device=device)

    # pre-compute polynomial bases once
    bases = (V ** torch.arange(n - 2, -1, -1, device=device)).long()

    for seq_py, mask in sequences:
        if len(seq_py) < n:
            continue
        seq = torch.as_tensor(seq_py, dtype=torch.long, device=device)

        # sliding windows (vectorised)
        win   = seq.unfold(0, n, 1)            # (L-n+1, n)
        ctx   = win[:, :-1]
        targ  = win[:, -1]                     # (L-n+1,)

        ctx_hash  = (ctx * bases).sum(-1) % E
        targ_hash = targ % out
        flat_idx  = ctx_hash * out + targ_hash   # (L-n+1,)

        # accumulate via 1-D bincount
        binc = torch.bincount(flat_idx, minlength=E * out).view(E, out)
        counts32 += binc

    # --- clamp + down-cast -------------------------------------------------
    counts8 = counts32.to(torch.uint8)
    return counts8
