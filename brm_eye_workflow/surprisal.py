"""Split-aware character/word n-gram surprisal utilities."""

from __future__ import annotations

from collections import Counter
from math import log
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

from .io_utils import safe_literal_list


def train_ngram_counts(sequences: Iterable[Sequence[str]], n: int = 3) -> tuple[Counter, Counter, set]:
    """Train add-k-smoothed n-gram counts from token sequences."""
    ngram_counts: Counter = Counter()
    context_counts: Counter = Counter()
    vocab = set()
    bos = ["<s>"] * (n - 1)
    for seq in sequences:
        toks = list(map(str, seq))
        vocab.update(toks)
        padded = bos + toks + ["</s>"]
        for i in range(n - 1, len(padded)):
            context = tuple(padded[i - n + 1:i])
            token = padded[i]
            ngram_counts[(context, token)] += 1
            context_counts[context] += 1
    vocab.add("</s>")
    return ngram_counts, context_counts, vocab


def sequence_surprisal(
    tokens: Sequence[str],
    ngram_counts: Counter,
    context_counts: Counter,
    vocab: set,
    n: int = 3,
    alpha: float = 0.1,
) -> List[float]:
    """Compute token-level negative log probabilities under add-alpha smoothing."""
    out: List[float] = []
    bos = ["<s>"] * (n - 1)
    padded = bos + list(map(str, tokens))
    v = max(len(vocab), 1)
    for i in range(n - 1, len(padded)):
        context = tuple(padded[i - n + 1:i])
        token = padded[i]
        num = ngram_counts[(context, token)] + alpha
        den = context_counts[context] + alpha * v
        out.append(-log(num / den))
    return out


def add_split_aware_surprisal(
    df: pd.DataFrame,
    token_list_col: str = "IA_LABEL",
    split_col: str = "split_TEXT",
    train_label: str = "train",
    n: int = 3,
    alpha: float = 0.1,
    output_col: str = "surprisal",
) -> pd.DataFrame:
    """Add split-aware n-gram surprisal without using held-out text to train.

    The language model is trained only on rows in ``split_col == train_label``.
    """
    train_sequences = [safe_literal_list(x) for x in df.loc[df[split_col] == train_label, token_list_col]]
    counts, ctx, vocab = train_ngram_counts(train_sequences, n=n)
    out = df.copy()
    out[output_col] = [sequence_surprisal(safe_literal_list(x), counts, ctx, vocab, n=n, alpha=alpha)
                       for x in out[token_list_col]]
    return out
