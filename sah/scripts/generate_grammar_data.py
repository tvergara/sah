import random

from nltk import CFG
from nltk.grammar import Nonterminal  # handy alias
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def random_sentence(grammar, symbol=None, depth=0, max_depth=10):
    """Return a single randomly-generated sentence from *grammar*."""
    if symbol is None:                 # start symbol
        symbol = grammar.start()

    if depth > max_depth:
        return []

    if isinstance(symbol, Nonterminal):
        productions = grammar.productions(lhs=symbol)
        prod = random.choice(productions)
        words = []
        for sym in prod.rhs():
            words.extend(random_sentence(grammar, sym, depth+1, max_depth))
        return words
    else:
        return [symbol]


def make_random_cfg(
    n_nonterms: int = 10,
    n_terms: int = 10,
    n_prods_per_nonterm: int = 3,
    avg_branch: int = 4,
    p_recursive: float = 0.6,
    seed: int | None = None,
):
    """Build a random context-free grammar string that NLTK can parse.

    Returns (grammar_string, start_symbol)
    """
    if seed is not None:
        random.seed(seed)

    # Define symbol inventories
    nonterms = [f"N{i}" for i in range(n_nonterms)]
    terms    = [f"t{i}" for i in range(n_terms)]
    start    = nonterms[0]          # S ≡ N0

    prods = []

    for A in nonterms:
        # --- 1. obligatory lexical rule: A → 'ti'
        t = random.choice(terms)
        prods.append(f"{A} -> '{t}'")

        # --- 2. additional rules
        for _ in range(n_prods_per_nonterm - 1):
            rhs = []
            # decide RHS length (≥1)
            k = max(1, int(random.expovariate(1 / (avg_branch - 1))) + 1)
            for _ in range(k):
                if random.random() < p_recursive:
                    rhs.append(random.choice(nonterms))
                else:
                    rhs.append(f"'{random.choice(terms)}'")
            prods.append(f"{A} -> {' '.join(rhs)}")

    grammar_str = "\n".join(prods)
    return grammar_str, start


def create_dataset(
    n=1_000,
    max_depth=20,
    test_split=0.2,
    seed=42,
):
    random.seed(seed)
    grammar_text, S = make_random_cfg(seed=seed)
    cfg = CFG.fromstring(grammar_text)

    data = []
    for _ in tqdm(range(n)):
        data.append(' '.join(random_sentence(cfg, max_depth=max_depth)))

    train_test_split(
        data,
        test_size=test_split,
        random_state=seed,
        shuffle=True,
    )



if __name__ == "__main__":
    create_dataset()
