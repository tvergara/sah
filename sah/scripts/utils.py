import random

from nltk import CFG
from nltk.grammar import Nonterminal


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
