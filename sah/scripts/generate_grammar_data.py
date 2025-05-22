from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from nltk import CFG
from nltk.grammar import Nonterminal
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CFG utilities
# ──────────────────────────────────────────────────────────────────────────────

def random_sentence(grammar: CFG, symbol: Nonterminal | None = None, *, depth: int = 0, max_depth: int = 10):
    """Return a single randomly generated sentence from *grammar*."""
    if symbol is None:
        symbol = grammar.start()

    if depth > max_depth:
        return []

    if isinstance(symbol, Nonterminal):
        productions = grammar.productions(lhs=symbol)
        prod = random.choice(productions)
        words: list[str] = []
        for sym in prod.rhs():
            words.extend(random_sentence(grammar, sym, depth=depth + 1, max_depth=max_depth))
        return words
    else:
        return [symbol]


# ──────────────────────────────────────────────────────────────────────────────
# Grammar factory
# ──────────────────────────────────────────────────────────────────────────────

def make_random_cfg(
    *,
    n_nonterms: int = 10,
    n_terms: int = 10,
    n_prods_per_nonterm: int = 3,
    avg_branch: int = 4,
    p_recursive: float = 0.6,
    min_len: int = 1,
    seed: int | None = None,
):
    """Return an *NLTK‑parsable* CFG string that produces ≥ `min_len` tokens.

    The strategy: build a base random grammar (start symbol `N0`), then
    prepend a new start symbol `S`.  `S` expands to a sequence of
    *exactly* `min_len` occurrences of `N0` followed by an *optional*
    `N0`.  Because each `N0` yields ≥1 terminal, the whole sentence
    length is lower‑bounded by `min_len`.
    """
    if seed is not None:
        random.seed(seed)
    if min_len < 1:
        raise ValueError("min_len must be ≥ 1")

    # 1) generate the *base* grammar (start symbol N0)
    nonterms = [f"N{i}" for i in range(n_nonterms)]
    terms = [f"t{i}" for i in range(n_terms)]

    productions: list[str] = []
    for A in nonterms:
        # lexical rule (mandatory)
        productions.append(f"{A} -> '{random.choice(terms)}'")
        # synthetic rules
        for _ in range(n_prods_per_nonterm - 1):
            rhs: list[str] = []
            k = max(1, int(random.expovariate(1 / (avg_branch - 1))) + 1)
            for _ in range(k):
                rhs.append(random.choice(nonterms) if random.random() < p_recursive else f"'{random.choice(terms)}'")
            productions.append(f"{A} -> {' '.join(rhs)}")

    # 2) wrap with a new start symbol S enforcing ≥ min_len tokens
    # `S -> N0 N0 ... N0`  (min_len times) optionally followed by another N0
    start_rule = "S -> " + " ".join(["N0"] * min_len)
    productions.insert(0, start_rule)  # make S the first rule (hence start)

    return "\n".join(productions)

# ──────────────────────────────────────────────────────────────────────────────
# Hydra configuration schema
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GrammarCfg:
    n_nonterms: int = 10
    n_terms: int = 10
    n_prods_per_nonterm: int = 3
    avg_branch: int = 4
    p_recursive: float = 0.6
    min_len: int = 1  # ← new!
    seed: int | None = 42


@dataclass
class DatasetCfg:
    n: int = 1_000
    max_depth: int = 20
    test_split: float = 0.2
    seed: int = 42
    save_dir: str = "initial-grammar"


@dataclass
class Config:
    grammar: GrammarCfg = field(default_factory=GrammarCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)

# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────────────

def create_dataset(cfg: Config):
    random.seed(cfg.dataset.seed)

    grammar_text = make_random_cfg(**OmegaConf.to_container(cfg.grammar, resolve=True))
    cfg_obj = CFG.fromstring(grammar_text)

    data: list[str] = []
    for _ in tqdm(range(cfg.dataset.n), desc="Generating sentences"):
        data.append(" ".join(random_sentence(cfg_obj, max_depth=cfg.dataset.max_depth)))

    train_data, test_data = train_test_split(
        data,
        test_size=cfg.dataset.test_split,
        random_state=cfg.dataset.seed,
        shuffle=True,
    )

    out_dir = Path(cfg.dataset.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) write dataset
    (out_dir / "grammar.txt").write_text(grammar_text)
    (out_dir / "train.txt").write_text("\n".join(train_data))
    (out_dir / "test.txt").write_text("\n".join(test_data))

    # 2) snapshot active config (YAML‑only)
    (out_dir / "hyperparams.yaml").write_text(OmegaConf.to_yaml(cfg))

    print(
        f"\n✨ Saved dataset to {out_dir.resolve()} \n   ├─ train.txt  ({len(train_data)} lines)\n   ├─ test.txt   ({len(test_data)} lines)\n   ├─ grammar.txt (CFG rules)\n   └─ hyperparams.yaml"
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
