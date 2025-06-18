from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path

import hydra
from nltk import CFG, Production
from nltk.grammar import Nonterminal
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import random_sentence


def is_lexical(prod: Production) -> bool:
    """True iff production is purely terminal (all RHS symbols are strings)."""
    return all(isinstance(s, str) for s in prod.rhs())

def trim_grammar(
    g: CFG,
    *,
    keep_prob: float = 0.8,
    seed: int | None = None,
    ensure_lexical: bool = True,
) -> CFG:
    """Return a pruned copy of *g* where each production is kept with probability *keep_prob*.

    Guarantees that:
    • every LHS keeps at least one production,
    • start rule is always kept,
    • (optional) each non-terminal keeps ≥1 lexical rule if it had one.
    """
    kept: dict[Nonterminal, list[Production]] = {lhs: [] for lhs in g._lhs_index}
    start_lhs = g.start()

    for p in g.productions():
        lhs = p.lhs()
        if lhs == start_lhs:                      # never drop start rule
            kept[lhs].append(p)
            continue
        if random.random() < keep_prob:
            kept[lhs].append(p)

    # enforce the guarantees
    for lhs, prods in kept.items():
        if not prods:                             # keep at least one prod
            prods.append(random.choice(g.productions(lhs=lhs)))
        if ensure_lexical:
            has_lex = any(is_lexical(p) for p in prods)
            if not has_lex:                       # re-add one lexical rule
                lexical_candidates = [p for p in g.productions(lhs=lhs) if is_lexical(p)]
                if lexical_candidates:
                    prods.append(random.choice(lexical_candidates))

    pruned_prods = [p for lst in kept.values() for p in lst]
    return CFG(start=start_lhs, productions=pruned_prods)

# ──────────────────────────────────────────────────────────────────────────────
# Hydra configuration schema
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TrimCfg:
    keep_prob: float = 0.8
    ensure_lexical: bool = True
    seed: int | None = 123

@dataclass
class DatasetCfg:
    n: int = 1_000
    max_depth: int = 20
    test_split: float = 0.2
    seed: int = 42
    out_dir: str = "trimmed-grammar"

@dataclass
class InitialGrammar:
    path: str = "initial-grammar"

@dataclass
class Config:
    initial_grammar: InitialGrammar = field(default_factory=InitialGrammar)
    trim: TrimCfg = field(default_factory=TrimCfg)
    dataset: DatasetCfg = field(default_factory=DatasetCfg)

# ──────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ──────────────────────────────────────────────────────────────────────────────
def create_dataset(cfg: Config):
    # ––– Load original grammar –––
    src = Path(cfg.initial_grammar.path) / "grammar.txt"
    if not src.exists():
        raise FileNotFoundError(src)
    orig_grammar = CFG.fromstring(src.read_text())

    # ––– Trim productions –––
    pruned = trim_grammar(
        orig_grammar,
        keep_prob=cfg.trim.keep_prob,
        seed=cfg.trim.seed,
    )


    yaml_str = OmegaConf.to_yaml(cfg)
    digest = hashlib.sha256(yaml_str.encode("utf-8")).hexdigest()
    derived_seed = int(digest, 16) % (2 ** 32)

    # ––– Generate sentences –––
    random.seed(derived_seed)
    data: list[str] = []
    for _ in tqdm(range(cfg.dataset.n), desc="Generating sentences"):
        data.append(" ".join(random_sentence(pruned, max_depth=cfg.dataset.max_depth)))

    train, test = train_test_split(
        data,
        test_size=cfg.dataset.test_split,
        random_state=cfg.dataset.seed,
        shuffle=True,
    )

    # ––– Persist –––
    out = Path(cfg.dataset.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grammar.txt").write_text("\n".join(map(str, pruned.productions())))
    (out / "train.txt").write_text("\n".join(train))
    (out / "test.txt").write_text("\n".join(test))
    (out / "hyperparams.yaml").write_text(OmegaConf.to_yaml(cfg))

    print(
        f"\n✨ Saved to {out.resolve()}\n"
        f"   ├─ train.txt   ({len(train)} lines)\n"
        f"   ├─ test.txt    ({len(test)} lines)\n"
        f"   ├─ grammar.txt (pruned rules)\n"
        f"   └─ hyperparams.yaml"
    )

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path=None)
def main(cfg: Config):  # type: ignore[arg-type]
    print("Active Hydra config:\n" + OmegaConf.to_yaml(cfg))
    create_dataset(cfg)

if __name__ == "__main__":
    main()
