#!/usr/bin/env python
"""Compare predictive entropy and cross‑entropy between two text corpora using *n*-gram language
models instead of ground‑truth PFSAs.

The script mirrors the previous PFSA‑based implementation but now:
  • Loads **train.txt** from the *original* (A) and *variant* (B) folders
    and learns separate *n*-gram models (order *n* is configurable).
  • Uses those models to estimate next‑token probability distributions
    for each position in **test.txt** of the *variant* corpus.
  • Reports
        H(A)   = average predictive entropy under model A
        H(B, A) = average cross‑entropy of B‑model predictions evaluated
                  under A‑model log‑probs
        Δ       = H(B, A) − H(A)
  • Optionally appends the metrics to a CSV (one row per experiment).

Dependencies: *omegaconf*, *hydra‑core*, *torch* (used only for numeric
convenience – no GPU required).  All heavy lifting is plain‑Python.
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


# ─── Hydra config ────────────────────────────────────────────────────────────
@dataclass
class AutomataCfg:  # kept for path semantics only
    variant_path: str = "modified-automaton"   # B
    original_path: str = "initial-automaton"   # A (baseline)

@dataclass
class EvalCfg:
    device: str = "cpu"        # n‑gram math is CPU‑friendly
    eps: float = 1e-12

@dataclass
class NgramCfg:
    n: int = 3                  # n‑gram order (e.g. 3 ⇒ trigram)
    alpha: float = 1.0          # add‑α (Laplace) smoothing

@dataclass
class ResultCfg:
    file: str = "results.csv"  # set to "" to disable writing
    run_id: str = "exp000"     # unique identifier for this run

@dataclass
class Config:
    automaton: AutomataCfg = field(default_factory=AutomataCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    ngram: NgramCfg = field(default_factory=NgramCfg)
    result: ResultCfg = field(default_factory=ResultCfg)

# ─── data helpers ────────────────────────────────────────────────────────────

def read_corpus(txt_path: Path) -> list[list[str]]:
    """Return a list of token sequences (split on whitespace)."""
    return [ln.split() for ln in txt_path.read_text().splitlines() if ln]


def build_vocab(*corpora: list[list[str]]) -> list[str]:
    """Return a *sorted* vocabulary covering all corpora."""
    vocab = sorted({tok for corpus in corpora for seq in corpus for tok in seq})
    return vocab


# ─── n‑gram model -----------------------------------------------------------

Context = tuple[int, ...]  # hashable representation of previous tokens


def train_ngram(seqs: list[list[int]], n: int) -> list[dict[Context, Counter]]:
    """Return counts for orders 0…n‑1 (index == context length)."""
    counts_by_order: list[dict[Context, Counter]] = [defaultdict(Counter) for _ in range(n)]

    for seq in tqdm(seqs):
        for t, y in enumerate(seq):
            max_back = min(t, n - 1)  # context length cannot exceed n‑1
            for k in range(max_back + 1):  # 0 … max_back inclusive
                ctx = tuple(seq[t - k : t])  # last *k* tokens (k==0 → empty ctx)
                counts_by_order[k][ctx][y] += 1
    return counts_by_order


def predict_dist(
    counts_by_order: list[dict[Context, Counter]],
    context: Context,
    vocab_size: int,
    alpha: float,
    device: torch.device,
):
    """Return *torch.float64* probability vector of size *vocab_size*.

    Uses *add‑alpha* smoothing and backs off by progressively shortening the context until counts
    are found (or ultimately to unigrams).  Assumes α > 0.
    """
    k = len(context)
    while k >= 0:
        ctx = context[-k:] if k else ()  # take last *k* tokens
        counter = counts_by_order[k].get(ctx)
        if counter:
            break
        k -= 1
    # If still no counts (unlikely), fall back to uniform distribution
    counter = counter or {}

    probs = torch.full((vocab_size,), alpha, dtype=torch.float64, device=device)
    for tok_id, cnt in counter.items():
        probs[tok_id] += cnt
    probs /= probs.sum()
    return probs


# ─── metrics computation ----------------------------------------------------

@torch.no_grad()
def entropy_and_cross(
    ids: torch.Tensor,
    model_A: list[dict[Context, Counter]],
    model_B: list[dict[Context, Counter]],
    n: int,
    vocab_size: int,
    alpha: float,
    device="cpu",
    eps=1e-12,
):
    """Compute predictive entropy H(A) and cross‑entropy H(B, A)."""
    ids = ids.to(device)
    PAD = -1  # identical to previous script
    mask = ids.ne(PAD)

    ent_A = cross_BA = tot = 0.0

    # iterate over *every* non‑PAD token in the batch
    B, L = ids.shape
    for b in tqdm(range(B)):
        seq_ids = ids[b]
        for t in range(L):
            if not mask[b, t]:
                continue
            # build (n‑1)-length context, ignoring PADs on the left
            left = max(0, t - (n - 1))
            ctx_ids = [seq_ids[i].item() for i in range(left, t) if seq_ids[i] != PAD]
            context: Context = tuple(ctx_ids)

            p_A = predict_dist(model_A, context, vocab_size, alpha, device)
            p_B = predict_dist(model_B, context, vocab_size, alpha, device)

            # entropy and cross‑entropy (vector form)
            log_p_A = (p_A + eps).log()
            ent_A += (-(p_A * log_p_A).sum()).item()
            cross_BA += (-(p_B * log_p_A).sum()).item()
            tot += 1

    return ent_A / tot, cross_BA / tot


# ─── glue logic -------------------------------------------------------------

def tokens_to_ids(seqs: list[list[str]], tok2id: dict[str, int]) -> list[list[int]]:
    return [[tok2id[t] for t in seq] for seq in seqs]


def pad_stack(seqs: list[list[int]], pad: int = -1) -> torch.Tensor:
    L = max(map(len, seqs)) if seqs else 0
    t = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return t


def run(cfg: Config):
    var_dir, ori_dir = Path(cfg.automaton.variant_path), Path(cfg.automaton.original_path)

    # ── 1. Load corpora ------------------------------------------------------
    train_B_tok = read_corpus(var_dir / "train.txt")
    train_A_tok = read_corpus(ori_dir / "train.txt")

    test_tok = read_corpus(var_dir / "test.txt")

    # ── 2. Build shared vocabulary -----------------------------------------
    vocab = build_vocab(train_B_tok, train_A_tok, test_tok)
    tok2id = {t: i for i, t in enumerate(vocab)}
    V = len(vocab)

    # ── 3. Convert to ids ----------------------------------------------------
    train_B_ids = tokens_to_ids(train_B_tok, tok2id)
    train_A_ids = tokens_to_ids(train_A_tok, tok2id)
    test_ids    = pad_stack(tokens_to_ids(test_tok, tok2id))  # (B_test, L)

    # ── 4. Train n‑gram models ---------------------------------------------
    print(f"Training {cfg.ngram.n}-gram models … (|V| = {V})")
    model_B = train_ngram(train_B_ids, cfg.ngram.n)
    model_A = train_ngram(train_A_ids, cfg.ngram.n)

    # ── 5. Metrics ----------------------------------------------------------
    H_A, H_BA = entropy_and_cross(
        test_ids,
        model_A,
        model_B,
        n=cfg.ngram.n,
        vocab_size=V,
        alpha=cfg.ngram.alpha,
        device=cfg.eval.device,
        eps=cfg.eval.eps,
    )

    # ── 6. Report -----------------------------------------------------------
    print(
        f"\nAverage predictive entropy  H(A)            : {H_A:.4f} nats "
        f"({H_A / math.log(2):.4f} bits)"
    )
    print(
        f"Average cross‑entropy      H(B, A) = H_BA  : {H_BA:.4f} nats "
        f"({H_BA / math.log(2):.4f} bits)"
    )
    print(
        f"Δ (H_BA − H_A)                                : {H_BA - H_A:+.4f} nats "
        f"({(H_BA - H_A) / math.log(2):+.4f} bits)"
    )

    # ── 7. Save to CSV ------------------------------------------------------
    if cfg.result.file:
        out_path = Path(cfg.result.file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["id", "n", "H_A", "H_BA"])
            writer.writerow([cfg.result.run_id, cfg.ngram.n, H_A, H_BA])
        print(f"Results appended to {out_path.resolve()}")


# ─── CLI entrypoint (Hydra) -------------------------------------------------
@hydra.main(version_base=None, config_path=None)
def main(cfg: Config):  # type: ignore[arg-type]
    print("Hydra cfg:\n" + OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()
