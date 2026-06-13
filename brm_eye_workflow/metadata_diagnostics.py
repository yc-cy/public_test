"""Metadata-leakage diagnostics and reader-level permutation controls."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


def reader_level_permute(
    df: pd.DataFrame,
    reader_col: str,
    metadata_cols: Sequence[str],
    seed: int = 13,
    suffix: str = "_perm",
) -> pd.DataFrame:
    """Permute metadata at reader level while preserving within-reader consistency."""
    rng = np.random.default_rng(seed)
    reader_meta = df[[reader_col, *metadata_cols]].drop_duplicates(reader_col).reset_index(drop=True)
    permuted = reader_meta.copy()
    permuted[metadata_cols] = reader_meta[metadata_cols].iloc[rng.permutation(len(reader_meta))].to_numpy()
    permuted = permuted.rename(columns={c: f"{c}{suffix}" for c in metadata_cols})
    return df.merge(permuted, on=reader_col, how="left")


def centered_kernel_alignment(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered kernel alignment (CKA)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    y = y - np.nanmean(y, axis=0, keepdims=True)
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    k = linear_kernel(x)
    l = linear_kernel(y)
    hsic = np.sum(k * l)
    norm = np.sqrt(np.sum(k * k) * np.sum(l * l))
    return float(hsic / norm) if norm > 0 else np.nan


def cka_metadata_identity_test(
    reader_metadata: pd.DataFrame,
    reader_col: str,
    metadata_cols: Sequence[str],
    n_perm: int = 500,
    seed: int = 13,
) -> dict:
    """Compare metadata representation with one-hot reader identity using CKA."""
    meta = reader_metadata[[reader_col, *metadata_cols]].drop_duplicates(reader_col).copy()
    z = pd.get_dummies(meta[metadata_cols], dummy_na=True).to_numpy(dtype=float)
    u = pd.get_dummies(meta[reader_col]).to_numpy(dtype=float)
    observed = centered_kernel_alignment(z, u)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        null.append(centered_kernel_alignment(z[rng.permutation(len(z))], u))
    null_arr = np.asarray(null)
    p = (np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1)
    return {
        "observed_cka": observed,
        "p_value_upper": float(p),
        "null_mean": float(np.mean(null_arr)),
        "null_ci_low": float(np.quantile(null_arr, 0.025)),
        "null_ci_high": float(np.quantile(null_arr, 0.975)),
    }


def compute_feature_gain_table(
    results: pd.DataFrame,
    baseline_config: str = "word",
    all_config: str = "all",
    perm_config: str = "all-permute",
    metric_col: str = "R2",
    keys: Sequence[str] = ("split", "model", "target"),
) -> pd.DataFrame:
    """Compute all-vs-word and permuted-vs-word gains from a result table."""
    index_cols = list(keys)
    wide = results.pivot_table(index=index_cols, columns="config", values=metric_col, aggfunc="mean").reset_index()
    if baseline_config not in wide:
        raise ValueError(f"Missing baseline config: {baseline_config}")
    wide["delta_all"] = wide.get(all_config, np.nan) - wide[baseline_config]
    wide["delta_perm"] = wide.get(perm_config, np.nan) - wide[baseline_config]
    return wide
