"""Domain-separated splitting utilities.

The split labels follow the workflow in the paper: TEXT (held-out texts),
READER (held-out readers), and RXT (joint held-out readers and texts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .io_utils import read_table, write_table


def _partition_ids(ids: Sequence, ratios=(0.6, 0.2, 0.2), seed: int = 13) -> Dict[object, str]:
    ids = np.array(sorted(pd.unique(pd.Series(ids).dropna())))
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    labels = {}
    for x in ids[:n_train]:
        labels[x] = "train"
    for x in ids[n_train:n_train + n_val]:
        labels[x] = "val"
    for x in ids[n_train + n_val:]:
        labels[x] = "test"
    return labels


def make_domain_splits(
    input_path: str | Path,
    output_path: str | Path,
    reader_col: str = "reader_id",
    text_col: str = "text_id",
    ratios=(0.6, 0.2, 0.2),
    seed: int = 13,
) -> pd.DataFrame:
    """Add TEXT/READER/RXT split labels to a token-level table.

    RXT uses independently partitioned readers and texts and keeps only rows
    whose reader and text labels agree. This creates strict joint held-out
    partitions while preserving a simple reproducible rule.
    """
    df = read_table(input_path)
    if reader_col not in df.columns or text_col not in df.columns:
        raise ValueError(f"Input must contain {reader_col!r} and {text_col!r}.")
    reader_map = _partition_ids(df[reader_col], ratios, seed)
    text_map = _partition_ids(df[text_col], ratios, seed + 1)
    df = df.copy()
    df["split_TEXT"] = df[text_col].map(text_map)
    df["split_READER"] = df[reader_col].map(reader_map)
    rxt = []
    for _, row in df.iterrows():
        a, b = row["split_READER"], row["split_TEXT"]
        rxt.append(a if a == b else "unused")
    df["split_RXT"] = rxt
    write_table(df, output_path)
    return df


def summarize_split_counts(
    df: pd.DataFrame,
    split_cols: Sequence[str] = ("split_TEXT", "split_READER", "split_RXT"),
) -> pd.DataFrame:
    """Return row counts for each split column."""
    records = []
    for col in split_cols:
        if col not in df.columns:
            continue
        for label, n in df[col].value_counts(dropna=False).items():
            records.append({"split_type": col, "partition": label, "n_rows": int(n)})
    return pd.DataFrame(records)
