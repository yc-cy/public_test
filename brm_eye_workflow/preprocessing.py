"""Preprocessing helpers for interest-area eye-movement tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from .io_utils import read_table, safe_literal_list, write_table


def combine_interest_area_rows(
    input_path: str | Path,
    output_path: str | Path,
    group_cols: Sequence[str] = ("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
) -> pd.DataFrame:
    """Group token-level rows into trial-level rows with list-valued cells.

    This is the open-source-safe version of the original ``IP_dataset_combine``
    idea. All file paths and grouping columns are configurable.
    """
    df = read_table(input_path)
    missing = set(group_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing grouping columns: {sorted(missing)}")
    out = df.groupby(list(group_cols), dropna=False).agg(lambda x: list(x)).reset_index()
    write_table(out, output_path)
    return out


def expand_list_valued_ia_table(
    input_path: str | Path,
    output_path: str | Path,
    id_cols: Sequence[str] = ("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
    list_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Expand trial-level list-valued IA columns into token-level rows.

    Each row is expanded by position. Rows with inconsistent list lengths are
    skipped and reported in a ``_skipped_reason`` column in the returned log.
    """
    df = read_table(input_path)
    if list_cols is None:
        list_cols = [c for c in df.columns if c not in id_cols]
    output_rows: List[dict] = []
    skipped: List[dict] = []
    for row_idx, row in df.iterrows():
        parsed = {col: safe_literal_list(row[col]) for col in list_cols}
        lengths = {col: len(vals) for col, vals in parsed.items()}
        nonzero_lengths = [v for v in lengths.values() if v > 0]
        if not nonzero_lengths or len(set(nonzero_lengths)) != 1:
            skipped.append({"row_index": row_idx, "_skipped_reason": f"lengths={lengths}"})
            continue
        n = nonzero_lengths[0]
        id_values = {col: row[col] for col in id_cols if col in df.columns}
        for pos in range(n):
            out_row = dict(id_values)
            out_row["IA_POSITION"] = pos
            for col, vals in parsed.items():
                out_row[col] = vals[pos] if pos < len(vals) else np.nan
            output_rows.append(out_row)
    out = pd.DataFrame(output_rows)
    write_table(out, output_path)
    if skipped:
        log_path = Path(output_path).with_suffix(".skipped_rows.csv")
        pd.DataFrame(skipped).to_csv(log_path, index=False)
    return out


def complete_missing_values(
    input_path: str | Path,
    output_path: str | Path,
    numeric_cols: Sequence[str],
    strategy: str = "mean",
) -> pd.DataFrame:
    """Replace dots/blank strings and impute numeric missing values.

    Parameters
    ----------
    strategy:
        ``mean``, ``median``, or ``zero``.
    """
    df = read_table(input_path).replace({".": np.nan, "": np.nan})
    for col in numeric_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if strategy == "mean":
            fill_value = df[col].mean()
        elif strategy == "median":
            fill_value = df[col].median()
        elif strategy == "zero":
            fill_value = 0
        else:
            raise ValueError("strategy must be one of: mean, median, zero")
        df[col] = df[col].fillna(fill_value)
    write_table(df, output_path)
    return df


def summarize_background(
    background_path: str | Path,
    group_col: str,
    numeric_cols: Sequence[str],
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Summarize participant-background variables by group."""
    df = read_table(background_path)
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    records = []
    for group, part in df.groupby(group_col, dropna=False):
        rec = {group_col: group, "N": len(part)}
        for col in numeric_cols:
            if col not in part.columns:
                continue
            x = pd.to_numeric(part[col], errors="coerce")
            rec[f"{col}_mean"] = x.mean()
            rec[f"{col}_sd"] = x.std()
            rec[f"{col}_min"] = x.min()
            rec[f"{col}_max"] = x.max()
        records.append(rec)
    out = pd.DataFrame(records)
    if output_path is not None:
        write_table(out, output_path)
    return out


def merge_word_features(
    ia_path: str | Path,
    word_feature_path: str | Path,
    output_path: str | Path,
    token_col: str = "IA_LABEL",
    feature_token_col: str = "word",
) -> pd.DataFrame:
    """Merge token-level IA rows with lexical features by word/token."""
    ia = read_table(ia_path)
    feats = read_table(word_feature_path)
    if token_col not in ia.columns or feature_token_col not in feats.columns:
        raise ValueError("Token columns are missing from one of the input tables.")
    out = ia.merge(feats, left_on=token_col, right_on=feature_token_col, how="left")
    write_table(out, output_path)
    return out
