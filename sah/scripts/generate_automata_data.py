from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import yaml
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AutomatonCfg:
    n_states: int = 10
    vocab_size: int = 10
    alpha_tokens: float = 0.1   # Dirichlet conc. for token rows
    p_self_loop: float = 0.2    # min prob of staying in same state
    seed: int | None = 42


@dataclass
class DatasetCfg:
    n: int = 1_000
    seq_len: int = 10
    test_split: float = 0.2
    seed: int = 42
    save_dir: str = "initial-automaton"


@dataclass
class Config:
    automaton: AutomatonCfg = field(default_factory=AutomatonCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)

# ──────────────────────────────────────────────────────────────────────────────
# PFSA factory
# ──────────────────────────────────────────────────────────────────────────────

def make_random_pfsa(cfg: AutomatonCfg):
    """Return (vocab, token_probs, next_state_cond).

    token_probs      : (S, V)
    next_state_cond  : (S, V, S)
    """
    rng = np.random.default_rng(cfg.seed)
    S, V = cfg.n_states, cfg.vocab_size
    vocab = [f"t{i}" for i in range(V)]

    # P(token | state)
    token_probs = rng.dirichlet(np.full(V, cfg.alpha_tokens), size=S)

    # P(next_state | state, token)
    next_state_cond = np.empty((S, V, S))
    for i in range(S):
        for j in range(V):
            base = rng.dirichlet(np.ones(S))           # random row
            base *= 1.0 - cfg.p_self_loop             # leave room for self-loop
            base[i] += cfg.p_self_loop
            next_state_cond[i, j] = base               # row sums to 1
    return vocab, token_probs, next_state_cond


def sample_sentence(
    *,
    vocab: list[str],
    token_probs: np.ndarray,
    next_state_cond: np.ndarray,
    seq_len: int,
    rng: np.random.Generator,
):
    S, V = token_probs.shape
    state = 0                 # q₀
    sent: list[str] = []
    for _ in range(seq_len):
        token = rng.choice(V, p=token_probs[state])
        sent.append(vocab[token])
        state = rng.choice(S, p=next_state_cond[state, token])
    return sent

# ──────────────────────────────────────────────────────────────────────────────
# Dataset pipeline
# ──────────────────────────────────────────────────────────────────────────────

def create_dataset(cfg: Config):
    random.seed(cfg.dataset.seed)
    rng = np.random.default_rng(cfg.dataset.seed)

    vocab, token_probs, next_state_cond = make_random_pfsa(cfg.automaton)

    data = [
        " ".join(
            sample_sentence(
                vocab=vocab,
                token_probs=token_probs,
                next_state_cond=next_state_cond,
                seq_len=cfg.dataset.seq_len,
                rng=rng,
            )
        )
        for _ in tqdm(range(cfg.dataset.n), desc="Generating sequences")
    ]

    train, test = train_test_split(
        data,
        test_size=cfg.dataset.test_split,
        random_state=cfg.dataset.seed,
        shuffle=True,
    )

    out = Path(cfg.dataset.save_dir)
    out.mkdir(parents=True, exist_ok=True)

    # write dataset
    (out / "train.txt").write_text("\n".join(train))
    (out / "test.txt").write_text("\n".join(test))
    (out / "literals.txt").write_text("\n".join(vocab))

    # snapshot automaton & config
    yaml.dump(
        {
            "vocab": vocab,
            "token_probs": token_probs.tolist(),
            "next_state_cond": next_state_cond.tolist(),
        },
        (out / "automaton.yaml").open("w"),
    )
    (out / "hyperparams.yaml").write_text(OmegaConf.to_yaml(cfg))

    print(
        f"\n✨ Saved dataset to {out.resolve()}\n"
        f"   ├─ train.txt        ({len(train)} lines)\n"
        f"   ├─ test.txt         ({len(test)} lines)\n"
        f"   ├─ automaton.yaml   (token-dep. transition tables)\n"
        f"   └─ hyperparams.yaml"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path=None)
def main(cfg: Config):  # type: ignore[arg-type]
    print("Hydra configuration (active):\n" + OmegaConf.to_yaml(cfg))
    create_dataset(cfg)


if __name__ == "__main__":
    main()
