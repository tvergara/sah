# modify_automaton_dataset.py  (state-count removal)
# ---------------------------------------------------------------------------
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import yaml
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ════════════════════════════════════════════════════════════════════════════
# Utility helpers  (same as before – renorm_rows, exponentiate_row ...)
# ════════════════════════════════════════════════════════════════════════════
def renorm_rows(mat: np.ndarray) -> np.ndarray:
    out = mat.copy()
    sums = out.sum(-1, keepdims=True)
    sums[sums == 0.0] = 1.0
    return out / sums

def exponentiate_row(row: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return row
    row = np.power(row, gamma)
    return row / row.sum()


# ════════════════════════════════════════════════════════════════════════════
# Hydra configuration
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class ModCfg:
    add_states: int = 0            # ⊕   how many new nodes to append
    remove_states: int = 0         # ⊖   exactly how many existing nodes to delete (≠ q0)
    gamma_token: float = 1.0       # spikiness for token rows
    gamma_trans: float = 1.0       # spikiness for transition rows
    new_state_noise: float = 0.05  # noise when cloning templates
    seed: int | None = 123         # RNG for modification phase

@dataclass
class DatasetCfg:
    n: int = 1_000
    seq_len: int = 10
    test_split: float = 0.2
    seed: int = 42
    out_dir: str = "modified-automaton"

@dataclass
class InitialAutomaton:
    path: str = "initial-automaton"

@dataclass
class Config:
    initial_automaton: InitialAutomaton = field(default_factory=InitialAutomaton)
    mod: ModCfg = field(default_factory=ModCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)


# ════════════════════════════════════════════════════════════════════════════
# Load + transform PFSA
# ════════════════════════════════════════════════════════════════════════════
def load_automaton(src: Path):
    d = yaml.safe_load((src / "automaton.yaml").read_text())
    vocab = d["vocab"]
    tp = np.asarray(d["token_probs"], dtype=np.float64)          # (S, V)
    nc = np.asarray(d["next_state_cond"], dtype=np.float64)      # (S, V, S)
    return vocab, tp, nc

def drop_and_add_states(
    token_probs: np.ndarray,
    next_state_cond: np.ndarray,
    cfg: ModCfg,
    rng: np.random.Generator,
):
    S, V = token_probs.shape
    if cfg.remove_states >= S:
        raise ValueError(
            f"remove_states={cfg.remove_states} is ≥ total states ({S})."
        )

    # ── 1.  remove exactly K states (except q0) ────────────────────────────
    removable = list(range(1, S))
    rng.shuffle(removable)
    to_remove = set(removable[: cfg.remove_states])
    keep_idx = [i for i in range(S) if i not in to_remove]

    token_probs = token_probs[keep_idx]                 # (S₁, V)
    next_state_cond = next_state_cond[keep_idx]         # (S₁, V, S)
    next_state_cond = next_state_cond[:, :, keep_idx]   # (S₁, V, S₁)
    next_state_cond = renorm_rows(next_state_cond)

    # ── helper to append one brand-new state cleanly ───────────────────────
    def append_state(tp: np.ndarray, nc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S_curr = tp.shape[0]
        S_new = S_curr + 1

        # choose a template state to clone
        t_idx = rng.integers(S_curr)
        tok_row = tp[t_idx]
        trans_rows = nc[t_idx]                          # (V, S_curr)

        # (a) new token row = noisy clone
        tok_uniform = np.full_like(tok_row, 1 / V)
        tok_row_new = (
            (1 - cfg.new_state_noise) * tok_row + cfg.new_state_noise * tok_uniform
        )
        tok_row_new /= tok_row_new.sum()
        tp = np.vstack([tp, tok_row_new[None, :]])      # (S_new, V)

        # (b) grow nc to (S_new, V, S_new)
        #     step-1: pad existing tensor with a zero column (new target)
        nc = np.pad(nc, ((0, 0), (0, 0), (0, 1)))       # (S_curr, V, S_new)

        #     step-2: give every (state,token) a small prob to jump to new state
        nc[:, :, -1] = cfg.new_state_noise
        nc[:, :, :-1] *= (1 - cfg.new_state_noise)
        nc = renorm_rows(nc)

        #     step-3: create new state's own (V, S_new) transition matrix
        trans_uniform = np.full((V, S_new), 1 / S_new)
        trans_rows_new = (
            (1 - cfg.new_state_noise) * np.hstack([trans_rows, trans_uniform[:, :1]])
            + cfg.new_state_noise * trans_uniform
        )
        trans_rows_new = renorm_rows(trans_rows_new)

        nc = np.concatenate([nc, trans_rows_new[None, :, :]], axis=0)  # (S_new, V, S_new)
        return tp, nc

    # ── 2.  append cfg.add_states new nodes ────────────────────────────────
    for _ in range(cfg.add_states):
        token_probs, next_state_cond = append_state(token_probs, next_state_cond)

    return token_probs, next_state_cond



def adjust_spikiness(tp: np.ndarray, nc: np.ndarray, γ_tok: float, γ_tr: float):
    tp = np.apply_along_axis(exponentiate_row, 1, tp, γ_tok)
    S, V, _ = nc.shape
    for i in range(S):
        for j in range(V):
            nc[i, j] = exponentiate_row(nc[i, j], γ_tr)
    return tp, nc


# ════════════════════════════════════════════════════════════════════════════
# Sampling (unchanged)
# ════════════════════════════════════════════════════════════════════════════
def sample_sentence(*, vocab, token_probs, next_state_cond, seq_len, rng):
    S, V = token_probs.shape
    state = 0
    out = []
    for _ in range(seq_len):
        tok = rng.choice(V, p=token_probs[state])
        out.append(vocab[tok])
        state = rng.choice(S, p=next_state_cond[state, tok])
    return out


# ════════════════════════════════════════════════════════════════════════════
# Core pipeline
# ════════════════════════════════════════════════════════════════════════════
def create_dataset(cfg: Config):
    src_dir = Path(cfg.initial_automaton.path)
    vocab, tp, nc = load_automaton(src_dir)

    # deterministic seed from full config
    digest = hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()
    rng = np.random.default_rng(int(digest, 16) % 2**32)

    tp, nc = drop_and_add_states(tp, nc, cfg.mod, rng)
    tp, nc = adjust_spikiness(tp, nc, cfg.mod.gamma_token, cfg.mod.gamma_trans)

    data = [
        " ".join(
            sample_sentence(
                vocab=vocab,
                token_probs=tp,
                next_state_cond=nc,
                seq_len=cfg.dataset.seq_len,
                rng=rng,
            )
        )
        for _ in tqdm(range(cfg.dataset.n), desc="Generating sequences")
    ]

    train, test = train_test_split(
        data, test_size=cfg.dataset.test_split, random_state=cfg.dataset.seed, shuffle=True
    )

    out = Path(cfg.dataset.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("\n".join(train))
    (out / "test.txt").write_text("\n".join(test))
    (out / "literals.txt").write_text("\n".join(vocab))
    yaml.safe_dump(
        {"vocab": vocab, "token_probs": tp.tolist(), "next_state_cond": nc.tolist()},
        (out / "automaton.yaml").open("w"),
    )
    (out / "hyperparams.yaml").write_text(OmegaConf.to_yaml(cfg))

    print(
        f"\n✨ Saved to {out.resolve()}\n"
        f"   ├─ train.txt      ({len(train)} lines)\n"
        f"   ├─ test.txt       ({len(test)} lines)\n"
        f"   ├─ automaton.yaml (modified PFSA)\n"
        f"   └─ hyperparams.yaml"
    )


# ════════════════════════════════════════════════════════════════════════════
@hydra.main(version_base=None, config_path=None)
def main(cfg: Config):  # type: ignore[arg-type]
    print("Active Hydra config:\n" + OmegaConf.to_yaml(cfg))
    create_dataset(cfg)

if __name__ == "__main__":
    main()
