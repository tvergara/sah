from math import inf
from random import shuffle

import torch
from tqdm.auto import tqdm

from sah.algorithms.data_strategies.hashed_ngram import get_hashed_ngram


def taylor_scores(batch_ngrams: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """DL ≈ ⟨∇_Q L , ΔQ⟩  with  ΔQ = batch_ngrams        (element-wise product + sum) The more
    *negative* the score, the larger the expected decrease in loss.

    batch_ngrams : (B, E, O)
    grad         : (E, O)
    returns      : (B,)
    """
    return (batch_ngrams * grad).sum(dim=(1, 2))          # (B,)


def add_examples(Q: torch.Tensor, chosen: torch.Tensor):
    # chosen is (k, E, O)   – in-place update to keep graph intact
    Q.data.add_(chosen.sum(dim=0))


# ───────────────────────────────────────── main loop ─────────────────────────
def get_idx_with_proposed_strategy(
    ft_dataset,
    pt_dataset,
    budget=20_000,
    vocab_size=22,
    k_per_iter=8,
    iterations=800,
    batch_size=2048,
    alpha=10.0,
    tau=0.1,
    device="cuda",
):
    # 0) reference counts
    ref_ngram = get_hashed_ngram(ft_dataset, vocab_size=vocab_size).to(device)

    # 1) pre-compute all candidate n-gram tensors once
    pt_ngrams = []
    for ex in tqdm(pt_dataset, desc="hashing PT examples"):
        pt_ngrams.append(get_hashed_ngram([ex], vocab_size=vocab_size))

    N = len(pt_dataset)
    chosen_mask = torch.zeros(N, dtype=torch.bool, device=device)
    chosen_ids  = []

    current_ngram = torch.zeros_like(ref_ngram, dtype=torch.float32,
                                     requires_grad=True)

    pt_matrix = torch.stack(pt_ngrams, 0)
    pt_matrix = pt_matrix.to(device, dtype=torch.uint8)

    for it in tqdm(range(iterations)):
        # ── 1. compute loss & gradient wrt current_ngram ────────────────────
        loss = compute_metric(current_ngram, ref_ngram, alpha)
        current_ngram.grad = None          # faster than zero_grad() on 1 tensor
        loss.backward()
        print('loss', loss)
        grad = current_ngram.grad          # (E, O)

        # ── 2. score *all unchosen* examples in batches ─────────────────────
        best_ids    = torch.full((k_per_iter,), -1, dtype=torch.long, device=device)
        neg_inf = -float("inf")       # handy constant

        best_util  = torch.empty(0, device=device)          # perturbed scores seen so far
        best_ids   = torch.empty(0, dtype=torch.long,
                                 device=device)             # corresponding indices

        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)

            batch = pt_matrix[start:end]
            mask  = chosen_mask[start:end]          # already-picked examples in this batch
            if mask.all():
                continue

            scores = taylor_scores(batch, grad)                   # (B,)
            scores[mask] = inf                                    # sentinel: never pick

            gumbel = -torch.log(-torch.log(torch.rand_like(scores)))

            util   = (-scores / tau) + gumbel                       # (B,)
            util[mask] = neg_inf                                  # keep masked out

            top_util, top_local = torch.topk(util, k_per_iter, largest=True)

            merged_util = torch.cat([best_util,  top_util])
            merged_ids  = torch.cat([best_ids,   top_local + start])

            best_util, keep = torch.topk(merged_util, k_per_iter, largest=True)
            best_ids         = merged_ids[keep]

        with torch.no_grad():
            current_ngram.add_(torch.stack([pt_ngrams[i].to(device) for i in best_ids]).sum(dim=0))
            # current_ngram.clamp_max_(255)

        chosen_mask[best_ids] = True
        chosen_ids.extend(best_ids.tolist())

        if len(chosen_ids) >= budget:
            break

    # in case we stopped early, trim to exactly `budget`
    shuffle(chosen_ids)
    return chosen_ids[:budget]


def compute_metric(
    current_ngram: torch.Tensor,   # (E, O)   counts  – “model”
    ngram:         torch.Tensor,   # (E, O)   counts  – “reference”
    alpha: float = 1.0,            # balances the terms
    eps: float = 1e-9,             # additive smoothing to avoid log(0)
):
    freq = ngram.to(torch.float32).sum(dim=-1)
    freq /= freq.sum()

    Q = current_ngram + eps
    P = ngram.to(torch.float32)       + eps

    Q = Q / Q.sum(dim=-1, keepdim=True)             # (E, O)
    P = P / P.sum(dim=-1, keepdim=True)

    H_Q = -(Q * Q.log()).sum(dim=-1)                # (E,)
    H_PQ = -(P * Q.log()).sum(dim=-1)               # (E,)

    return ((H_PQ -  alpha * H_Q) * freq).sum()
